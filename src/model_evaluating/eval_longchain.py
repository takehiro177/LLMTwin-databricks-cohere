# Databricks notebook source
# MAGIC %md
# MAGIC Use the Mosaic AI Agent Evaluation to evaluate your RAG applications

# COMMAND ----------

# MAGIC %pip install -U --quiet databricks-sdk==0.28.0 databricks-agents mlflow-skinny mlflow mlflow[gateway] databricks-vectorsearch langchain langchain-databricks langchain-cohere langchain_community
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# Library
import os
from langchain_community.llms import Databricks
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatDatabricks
from langchain.schema.runnable import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers import EnsembleRetriever
from langchain.prompts import PromptTemplate
from langchain_databricks.vectorstores import DatabricksVectorSearch
from langchain_cohere import CohereEmbeddings
from operator import itemgetter

import mlflow

# COMMAND ----------

chain_config = {
    "llm_model_serving_endpoint_name": "cohere-finetuned-chat-endpoint",  # the foundation model we want to use
    "vector_search_endpoint_name": "twinllm-cohere-vectorembed",  # the endoint we want to use for vector search
    "gitrepo_vector_search_index": "portfolio.twinllm.gitrepo_index",
    "pdf_vector_search_index": "portfolio.twinllm.pdf_index",
    "llm_prompt_template": """Write technical blog that demonstrates knowledge and best practice for {input} and its technical advantage and describes code snippest from the piece of information in the text bellow: {context} """,
}

# COMMAND ----------

## Enable MLflow Tracing
mlflow.langchain.autolog()

## Load the chain's configuration
model_config = mlflow.models.ModelConfig(development_config=chain_config)

## Turn the Vector Search index into a LangChain retriever
if not os.getenv("COHERE_API_KEY"):
    os.environ["COHERE_API_KEY"] = dbutils.secrets.get(scope="", key="")
    
embeddings = CohereEmbeddings(
    model="embed-english-v3.0",
)

gitrepo_vector_search = DatabricksVectorSearch(
    endpoint=model_config.get("vector_search_endpoint_name"),
    index_name=model_config.get("gitrepo_vector_search_index"),
    text_column="text",
    embedding=embeddings,
)

pdf_vector_search = DatabricksVectorSearch(
    endpoint=model_config.get("vector_search_endpoint_name"),
    index_name=model_config.get("pdf_vector_search_index"),
    text_column="text",
    embedding=embeddings,
)

# Create the retriever
gitrepo_retriever = gitrepo_vector_search.as_retriever(search_kwargs={"k": 4})
pdf_retriever = pdf_vector_search.as_retriever(search_kwargs={"k": 4})

# Initialize the LLM from databricks serving endpoint
llm = Databricks(endpoint_name=model_config.get("llm_model_serving_endpoint_name"))

# Create a chat prompt template
prompt = ChatPromptTemplate.from_messages(
    [  
        ("system", "You are a technical blog writer. You will write a blog post about a specific technology."), # Contains the instructions from the configuration
        ("user", model_config.get("llm_prompt_template")) #user's questions
    ]
)

# initialize the ensemble retriever to combine the two retrievers for aquiring different sources information based on the Reciprocal Rank Fusion algorithm.
ensemble_retriever = EnsembleRetriever(
    retrievers=[gitrepo_retriever, pdf_retriever], weights=[0.5, 0.5]
)

# Our foundation model answering the final prompt
model = ChatDatabricks(
    endpoint=model_config.get("llm_model_serving_endpoint_name"),
)

# Query the chain for a topic that you want llm to generate
topic = "DANets with Pytorch Lightning implementation: a flexible framework for integrating custom loss functions and metrics and a robust and adaptable training strategy aimed at achieving accuracy beyond traditional Gradient Boosting models like LGBM, XGBoost, and CatBoost."

# RAG Chain
# Method to format the docs returned by the retriever into the prompt (keep only the text from chunks)
def format_context(docs):
    chunk_contents = [f"Passage: {d.page_content}\n" for d in docs]
    return "".join(chunk_contents)

def extract_user_query_string(chat_messages_array):
    return chat_messages_array[-1]["content"]

chain = (
    {
        "input": itemgetter("messages") | RunnableLambda(extract_user_query_string),
        "context": itemgetter("messages") | RunnableLambda(extract_user_query_string) | ensemble_retriever | RunnableLambda(format_context),
    }
    | prompt
    | model
    | StrOutputParser()
)


# COMMAND ----------

# Let's give it a try:
input_example = {"messages": [{"role": "user", "content": topic}]}
answer = chain.invoke(input_example)
print(answer)

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")

# Log the model to MLflow
with mlflow.start_run(run_name="twinllm_cohere_rag_bot"):
  logged_chain_info = mlflow.langchain.log_model(
          #Note: In classical ML, MLflow works by serializing the model object.  In generative AI, chains often include Python packages that do not serialize.  Here, we use MLflow's new code-based logging, where we saved our chain under the chain notebook and will use this code instead of trying to serialize the object.
          lc_model=os.path.join(os.getcwd(), 'chain/chain.py'),  # Chain code file e.g., /path/to/the/chain.py 
          model_config=chain_config, # Chain configuration 
          artifact_path="chain", # Required by MLflow, the chain's code/config are saved in this directory
          input_example=input_example,
      )

MODEL_NAME = "twinllm_cohere_rag"
MODEL_NAME_FQN = f"portfolio.twinllm.{MODEL_NAME}"
# Register to Unity Catalog
uc_registered_model_info = mlflow.register_model(model_uri=logged_chain_info.model_uri, name=MODEL_NAME_FQN)

# COMMAND ----------

from databricks import agents
# Deploy to enable the Review APP and create an API endpoint
# Note: scaling down to zero will provide unexpected behavior for the chat app. Set it to false for a prod-ready application.
deployment_info = agents.deploy(MODEL_NAME_FQN, model_version=uc_registered_model_info.version, scale_to_zero=True)
instructions_to_reviewer = f"""## Instructions for Testing the Databricks Documentation Assistant chatbot

The inputs are invaluable for the development. By providing detailed feedback and corrections, it helps llm fix issues and improve the overall quality of the application."""

# Add the user-facing instructions to the Review App
agents.set_review_instructions(MODEL_NAME_FQN, instructions_to_reviewer)



# COMMAND ----------

print(f"\n\nReview App URL to share with your stakeholders: {deployment_info.review_app_url}")

# COMMAND ----------

eval_dataset = spark.table("portfolio.twinllm.twinllm_cohere_rag_payload").limit(10).toPandas()
display(eval_dataset)

# COMMAND ----------

from databricks.agents import destroy

# Destroy the deployment
destroy(MODEL_NAME_FQN, uc_registered_model_info.version)

# COMMAND ----------

# evaluate llm accuracy from feedback
eval_dataset = "NONE"

# COMMAND ----------

with mlflow.start_run(run_id=logged_chain_info.run_id):
    # Evaluate the logged model
    eval_results = mlflow.evaluate(
        data=eval_dataset,
        model=logged_chain_info.model_uri,
        model_type="databricks-agent",
    )

# COMMAND ----------


