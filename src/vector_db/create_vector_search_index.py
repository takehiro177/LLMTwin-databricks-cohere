# Databricks notebook source
# MAGIC %pip install -U --quiet databricks-sdk databricks-agents mlflow-skinny mlflow mlflow[gateway] databricks-vectorsearch langchain langchain_core langchain_community databricks-api
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# Library
import time

# COMMAND ----------

# utils
def endpoint_exists(client, endpoint_name):
    try:
        client.get_endpoint(endpoint_name)
        return True
    except Exception as e:
        if "not found" in str(e).lower():
            return False
        else:
            raise e

def index_exists(vsc, endpoint_name, index_name):
    try:
        indexes = vsc.list_indexes(name=endpoint_name)['vector_indexes']
        return any(index['name'] == index_name for index in indexes)
    except Exception as e:
        print(f"Error checking if index exists: {e}")
        return False
    
def wait_for_index_to_be_ready(vsc, endpoint_name, index_name, timeout=3600, interval=30):
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            vsc.wait_for_endpoint(name=endpoint_name)
            return True
        except Exception as e:
            print(f"Error checking index status: {e}")
        time.sleep(interval)
    print(f"Timeout waiting for index {index_name} to be ready.")
    return False


# COMMAND ----------

# create vector search endpoint

from databricks.vector_search.client import VectorSearchClient

VECTOR_SEARCH_ENDPOINT_NAME = "twinllm-cohere-vectorembed"

vsc = VectorSearchClient(disable_notice=True)
if not endpoint_exists(vsc, VECTOR_SEARCH_ENDPOINT_NAME):
    vsc.create_endpoint(name=VECTOR_SEARCH_ENDPOINT_NAME, endpoint_type="STANDARD")
    time.sleep(15)
    vsc.wait_for_endpoint(name=VECTOR_SEARCH_ENDPOINT_NAME)
    
print(f"Endpoint named {VECTOR_SEARCH_ENDPOINT_NAME} is ready.")

# COMMAND ----------

#%sql
#-- change data feed must be enabled for vector search index table
#ALTER TABLE portfolio.twinllm.gold_gitrepo_table
#SET TBLPROPERTIES (delta.enableChangeDataFeed = true);

# COMMAND ----------

#%sql
#-- change data feed must be enabled for vector search index table
#ALTER TABLE portfolio.twinllm.gold_pdf_table
#SET TBLPROPERTIES (delta.enableChangeDataFeed = true);

# COMMAND ----------

#The table we'd like to index
source_table_fullname = "portfolio.twinllm.gold_gitrepo_table"
# Where we want to store our index
vs_index_fullname = "portfolio.twinllm.gitrepo_index"

if not index_exists(vsc, VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname):
  print(f"Creating index {vs_index_fullname} on endpoint {VECTOR_SEARCH_ENDPOINT_NAME}...")
  vsc.create_delta_sync_index(
    endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME,
    index_name=vs_index_fullname,
    source_table_name=source_table_fullname,
    pipeline_type="TRIGGERED",
    primary_key="gitrepo_file_id",
    embedding_dimension=1024,
    embedding_vector_column="text_vector"
  )
  #Let's wait for the index to be ready and all our embeddings to be created and indexed
  wait_for_index_to_be_ready(vsc, VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname)
else:
  #Trigger a sync to update our vs content with the new data saved in the table
  wait_for_index_to_be_ready(vsc, VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname)
  vsc.get_index(VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname).sync()

print(f"index {vs_index_fullname} on table {source_table_fullname} is ready")

# COMMAND ----------

#The table we'd like to index
source_table_fullname = "portfolio.twinllm.gold_pdf_table"
# Where we want to store our index
vs_index_fullname = "portfolio.twinllm.pdf_index"

if not index_exists(vsc, VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname):
  print(f"Creating index {vs_index_fullname} on endpoint {VECTOR_SEARCH_ENDPOINT_NAME}...")
  vsc.create_delta_sync_index(
    endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME,
    index_name=vs_index_fullname,
    source_table_name=source_table_fullname,
    pipeline_type="TRIGGERED",
    primary_key="pdf_chunk_id",
    embedding_dimension=1024,
    embedding_vector_column="text_vector"
  )
  #Let's wait for the index to be ready and all our embeddings to be created and indexed
  wait_for_index_to_be_ready(vsc, VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname)
else:
  #Trigger a sync to update our vs content with the new data saved in the table
  wait_for_index_to_be_ready(vsc, VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname)
  vsc.get_index(VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname).sync()

print(f"index {vs_index_fullname} on table {source_table_fullname} is ready")

# COMMAND ----------

TEST = False

# COMMAND ----------

if TEST:
    import mlflow.deployments

    client = mlflow.deployments.get_deploy_client("databricks")

    # test query similarity search
    #question = "what is KAN in deep learning and what is benefit of it over MLP?"
    question = "what is Causal ML and what is the use case of it?"
    embeddings_response = client.predict(
        endpoint="cohere-embed-endpoint",
        inputs={
            "input": [question],
            "input_type": 'search_query'
        },
    )

# COMMAND ----------

if TEST:
  question_vector = embeddings_response['data'][0]['embedding']

  results = vsc.get_index(VECTOR_SEARCH_ENDPOINT_NAME, "portfolio.twinllm.gitrepo_index").similarity_search(
    query_vector=question_vector,
    columns=["text"],
    num_results=3)
  docs = results.get('result', {}).get('data_array', [])
  print(docs)

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT *
# MAGIC FROM system.billing.usage
# MAGIC WHERE billing_origin_product = 'VECTOR_SEARCH'
# MAGIC   AND usage_metadata.endpoint_name IS NOT NULL

# COMMAND ----------

# delete vector serch index
#vsc.delete_index(endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME, index_name="portfolio.twinllm.gitrepo_index")
#vsc.delete_index(endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME, index_name="portfolio.twinllm.pdf_index")
# wait until index deletion completed
#time.sleep(30)
# delete vector serch endpoint
#vsc.delete_endpoint(name=VECTOR_SEARCH_ENDPOINT_NAME)
