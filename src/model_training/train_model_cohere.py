# Databricks notebook source
# MAGIC %pip install -U --quiet databricks-sdk databricks-agents mlflow-skinny mlflow mlflow[gateway] databricks-vectorsearch langchain langchain_core langchain_community databricks-api databricks-feature-engineering
# MAGIC %pip install --quiet cohere
# MAGIC # version inconsistence between cohere and deprecate in typing_extensions
# MAGIC %pip install --quiet typing_extensions==4.7.1 --upgrade
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# Library
import pyspark.sql.functions as F
from pyspark.ml.feature import Tokenizer
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, StringType

# Client
import mlflow.deployments
client = mlflow.deployments.get_deploy_client("databricks")

from databricks.vector_search.client import VectorSearchClient

# vector search endpoint name
VECTOR_SEARCH_ENDPOINT_NAME = "twinllm-cohere-vectorembed"
# Where we want to store our index
gitrepo_vs_index_fullname = "portfolio.twinllm.gitrepo_index"
pdf_vs_index_fullname = "portfolio.twinllm.pdf_index"
vsc = VectorSearchClient(disable_notice=True)

# COMMAND ----------


df = spark.read.table("portfolio.twinllm.gold_substackpost_table")


# COMMAND ----------

def add_topic_to_query(topic):

    question = """Write technical blog that demonstrates knowledge and best practice for "TOPIC" and its technical advantage and describes code snippest from the piece of information in the text bellow: """

    question = question.replace("TOPIC", topic)
    return question

def search_text_from_topic(topic):

    #similarity search to given topic
    embeddings_response = client.predict(
        endpoint="cohere-embed-endpoint",
        inputs={
            "input": [topic],
            "input_type": 'search_query'
        },
    )

    question_vector = embeddings_response['data'][0]['embedding']

    results = vsc.get_index(VECTOR_SEARCH_ENDPOINT_NAME, gitrepo_vs_index_fullname).similarity_search(
        query_vector=question_vector,
        columns=["text"],
        num_results=7)
    docs_gitrepo = results.get('result', {}).get('data_array', [])

    # Concatenate all strings in the list of lists which is results of vector search where 1st entry contains text and second entry is similarity score in a list
    gitrepo_text = " ".join([item[0] for item in docs_gitrepo if item[1] > 0.5])

    results = vsc.get_index(VECTOR_SEARCH_ENDPOINT_NAME, pdf_vs_index_fullname).similarity_search(
        query_vector=question_vector,
        columns=["text"],
        num_results=7)
    docs_pdf = results.get('result', {}).get('data_array', [])

    # Concatenate all strings in the list of lists which is results of vector search where 1st entry contains text and second entry is similarity score in a list
    pdf_text = " ".join([item[0] for item in docs_pdf if item[1] > 0.5])

    text = gitrepo_text + " #" + pdf_text

    return text

add_topic_to_query_udf = udf(add_topic_to_query, StringType())
search_text_from_topic_udf = udf(search_text_from_topic, StringType())


# COMMAND ----------

df = (df.withColumn("topic", F.concat_ws(" ", F.col("title"), F.col("subtitle")))
        .withColumn("topic_question", add_topic_to_query_udf("topic"))
        .withColumn("search_result_text", search_text_from_topic_udf("topic_question"))
        .withColumn("query", F.concat(F.col("topic_question"), F.col("search_result_text")))
)


# COMMAND ----------

# Initialize the Tokenizer
tokenizer_target = Tokenizer(inputCol="text", outputCol="text_tokens")
tokenizer_query = Tokenizer(inputCol="query", outputCol="query_tokens")

# Transform the DataFrame to tokenize each row
df = tokenizer_target.transform(df)
df = tokenizer_query.transform(df)

# Count the number of tokens in each row
# create label
df = (df.withColumn("target_token_count", F.size(df["text_tokens"]))
        .withColumn("query_token_count", F.size(df["query_tokens"]))
        .withColumn("total_token_count", F.col("target_token_count") + F.col("query_token_count"))
        .filter(F.col("total_token_count") < 8000)  # Ensure the “System” preamble is under 4096 tokens and each conversation turn is within 8192 tokens to avoid being dropped from the dataset.
    )

# COMMAND ----------

display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create feature store to store training data

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup

fe = FeatureEngineeringClient()

table_name = "portfolio.twinllm.train-dataset"

# COMMAND ----------

feature_df = df.select("post_id", "query")
fe.create_table(
    name=table_name,
    primary_keys=["post_id"],
    df=feature_df,
    schema=feature_df.schema,
    description="training dataset from substack blog and text data searched from github repository and pdf materials"
)

# COMMAND ----------

# Update the feature table
#fe.write_table(
#    name=table_name,
#    df=df,
#    mode='merge'
#)

# Drop the future table
#fe.drop_table(name=table_name)

# COMMAND ----------

model_feature_lookups = [FeatureLookup(table_name=table_name, lookup_key="post_id", feature_names=["query"])]
training_set = fe.create_training_set(df=df.select("post_id", "text"), feature_lookups=model_feature_lookups, label="text", exclude_columns=[])

# COMMAND ----------

display(training_set.load_df())

# COMMAND ----------

import json

def create_finetune_data(rows):
    data = []
    for row in rows:
        data.append(
            {"messages": [
                    {
                    "role": "System",
                    "content": "You are a technical blog writer. You will write a blog post about a specific technology.",
                    },
                    {
                    "role": "User",
                    "content": row["query"],
                    },
                    {"role": "Chatbot",
                    "content": row["text"],
                    }
                ]
            }
        )
    # Write to file
    with open('/Workspace/Users/PATH/TwinLLM/src/data/cohere_finetune_data.jsonl', 'w') as file:
        for entry in data:
            json.dump(entry, file)
            file.write('\n')

# COMMAND ----------

# Iterate over DataFrame and call the function
training_data = training_set.load_df().toPandas()
create_finetune_data(training_data.to_dict(orient='records'))

# COMMAND ----------

import cohere

# instantiate the Cohere client
api_key = dbutils.secrets.get(scope="your_scope", key="your_cohere-production_key")
co = cohere.Client(api_key)  

# create cohere finetune tarining dataset through api
chat_dataset = co.datasets.create(name="twinllm-dataset",
                                   data=open("/Workspace/Users/PATH/TwinLLM/src/data/cohere_finetune_data.jsonl", "rb"),
                                   type="chat-finetune-input")
print(co.wait(chat_dataset))

# create cohere finetune tarining dataset with evaluation data through api                               
#chat_dataset_with_eval = co.datasets.create(name="chat-dataset-with-eval",
#                                           data=open("path/to/train.jsonl, "rb"),
#                                           eval_data=open("path/to/eval.jsonl, "rb"),
#                                           type="chat-finetune-input")
#print(co.wait(chat_dataset_with_eval))


# COMMAND ----------

from cohere.finetuning import (
    BaseModel,
    FinetunedModel,
    Hyperparameters,
    Settings,
    WandbConfig
)

hp = Hyperparameters(
    early_stopping_patience=3,
    early_stopping_threshold=0.001,
    train_batch_size=2,
    train_epochs=10,
    learning_rate=0.01,
)

finetuned_model = co.finetuning.create_finetuned_model(
    request=FinetunedModel(
        name="cohere-twinllm-finetuned-model",
        settings=Settings(
            base_model=BaseModel(
                base_type="BASE_TYPE_CHAT",
            ),
            dataset_id="your_dataset_id",
            hyperparameters=hp,
        ),
    )
)
print(finetuned_model)


# COMMAND ----------


