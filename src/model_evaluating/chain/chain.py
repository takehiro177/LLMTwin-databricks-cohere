# Library
import os
os.environ["DATABRICKS_HOST"] = "YOUR_HOST_URL_HERE"
os.environ["DATABRICKS_TOKEN"] = "YOUR_DATABRICKS_TOKEN_HERE"

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
mlflow.set_registry_uri("databricks-uc")  # enable accessing unity catalog registered models

chain_config = {
    "llm_model_serving_endpoint_name": "cohere-finetuned-chat-endpoint",  # the foundation model we want to use
    "vector_search_endpoint_name": "twinllm-cohere-vectorembed",  # the endoint we want to use for vector search
    "gitrepo_vector_search_index": "portfolio.twinllm.gitrepo_index",
    "pdf_vector_search_index": "portfolio.twinllm.pdf_index",
    "llm_prompt_template": """Write technical blog that demonstrates knowledge and best practice for {input} and its technical advantage and describes code snippest from the piece of information in the text bellow: {context} """,
}

## Enable MLflow Tracing
mlflow.langchain.autolog()

## Load the chain's configuration
model_config = mlflow.models.ModelConfig(development_config=chain_config)

## Turn the Vector Search index into a LangChain retriever
if not os.getenv("COHERE_API_KEY"):
    os.environ["COHERE_API_KEY"] = "YOUR_COHERE_API_KEY_HERE"
    
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

# Tell MLflow logging where to find your chain.
mlflow.models.set_model(model=chain)