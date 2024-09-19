# Databricks notebook source
#%pip install --quiet cohere
# version inconsistence between cohere and deprecate in typing_extensions
#%pip install --quiet typing_extensions==4.7.1 --upgrade
#%pip install langchain
#%pip install langchain-community
#%pip install --quiet langchain_experimental
%pip install --quiet mlflow-skinny mlflow mlflow[gateway]
dbutils.library.restartPython()

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType

import mlflow.deployments
client = mlflow.deployments.get_deploy_client("databricks")

#from pyspark.sql.functions import udf

def distil_file(text):
    message = """## Instruction
    Given the text which could be a code snippet, summarise the text. The summary is detailed and explains technical terms and implementation.
    The summary should include examples and code snippets. Before summary context, the output must include header of the text and keywords and technical terms. The output should only contain header, keywords and summary.

    ## Given Text
    {text}
    """
    chat_response = client.predict(
            endpoint="cohere-chat-endpoint",
            inputs={
                "messages": [{"role": "user", "content": message.format(text=text)}],
                "stream": False,
                "prompt_truncation": "AUTO"
            },
        )

    return chat_response['choices'][0]['message']['content']

def precompute_embed(text):

    inputs = [text]

    embeddings_response = client.predict(
        endpoint="cohere-embed-endpoint",
        inputs={
            "input": inputs,
            "input_type": 'search_query'
        },
    )

    return embeddings_response['data'][0]['embedding']

# Register the UDF but can not serialize
#distil_file_udf = udf(distil_file, StringType())
#spark.udf.register("distil_file_udf", distil_file_udf)
#precompute_embed_udf = udf(precompute_embed, StringType())
#spark.udf.register("precompute_embed_udf", precompute_embed_udf)

def batch_upsert(microBatchDF, batchId):

    # udf is slower for overhead, so we can use pandas
    microBatchPandasDF = microBatchDF.toPandas()
    microBatchPandasDF["text"] = microBatchPandasDF["content"].apply(distil_file)
    microBatchPandasDF["text_vector"] = microBatchPandasDF["text"].apply(precompute_embed)
    microBatchDF = spark.createDataFrame(microBatchPandasDF)

    (microBatchDF#.withColumn("text", distil_file_udf(F.col("content")))
                 #.withColumn("text_vector", precompute_embed_udf(F.col("text")))
                .write.format("delta")
                 .mode("append")
                 .option("overwriteSchema", "true")
                 .saveAsTable("portfolio.twinllm.silver_gitrepo_table")
                 )

# COMMAND ----------

query = (spark.readStream.table("portfolio.twinllm.bronze_gitrepo_table")
         .writeStream
         .foreachBatch(batch_upsert)
         .option("checkpointLocation", "dbfs:/twinllm/checkpoint/silver/gitrepos")
         .trigger(availableNow=True)
         .start()
)

query.awaitTermination()


# COMMAND ----------

# MAGIC %sql
# MAGIC DESCRIBE TABLE portfolio.twinllm.silver_gitrepo_table;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT text, file_name, repo FROM portfolio.twinllm.silver_gitrepo_table;

# COMMAND ----------

#%sql
#DROP TABLE portfolio.twinllm.silver_gitrepo_table

# COMMAND ----------

#dbutils.fs.rm("dbfs:/twinllm/checkpoint/silver/gitrepos", True)
