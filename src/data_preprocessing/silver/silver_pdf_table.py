# Databricks notebook source
# MAGIC %pip install unstructured[pdf]
# MAGIC %pip install --quiet mlflow-skinny mlflow mlflow[gateway]
# MAGIC %pip install -qU langchain-cohere
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# Retrieve the secret from the secret scope
storage_account_key = dbutils.secrets.get(scope="", key="")
spark.conf.set("fs.azure.account.key.CONTAINER_LOCATION.dfs.core.windows.net", storage_account_key)

import getpass
import os

if not os.getenv("COHERE_API_KEY"):
    os.environ["COHERE_API_KEY"] = dbutils.secrets.get(scope="", key="")

# COMMAND ----------

# Library
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, StringType

from unstructured.partition.pdf import partition_pdf

import mlflow.deployments
client = mlflow.deployments.get_deploy_client("databricks")

# COMMAND ----------

## longchain chunker to extract embedding
from langchain_cohere import CohereEmbeddings
from langchain_experimental.text_splitter import SemanticChunker

cohere_embeddings = CohereEmbeddings(model="embed-english-light-v3.0")
text_splitter = SemanticChunker(cohere_embeddings, breakpoint_threshold_type="gradient")

# COMMAND ----------


def merge_small_chunks(chunks, min_length=1000):
    merged_chunks = []
    current_chunk = ""
    for chunk in chunks:
        if len(current_chunk) + len(chunk) < min_length:
            current_chunk += " " + chunk
        else:
            if current_chunk:
                merged_chunks.append(current_chunk.strip())
            current_chunk = chunk
    if current_chunk:
        merged_chunks.append(current_chunk.strip())
    return merged_chunks

def split_text_with_semantic_segment(chunks):

    text = ""
    for chunk in chunks:
        text += " " + chunk

    new_chunks = text_splitter.split_text(text)

    return new_chunks

def extract_text_from_pdf(binary_content):
    """
    Function to extract text from a PDF file.
    """
    from io import BytesIO
    import os
    temp_pdf_path = "/tmp/temp_pdf.pdf"
    with open(temp_pdf_path, "wb") as f:
        f.write(binary_content)
    elements = partition_pdf(temp_pdf_path)
    elements = [str(element) for element in elements]
    os.remove(temp_pdf_path)  # Clean up the temporary file

    #merge_chunks = merge_small_chunks(elements)
    merge_chunks = split_text_with_semantic_segment(elements)
    return merge_chunks

extract_text_udf = udf(extract_text_from_pdf, ArrayType(StringType()))

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

def batch_upsert(microBatchDF, batchId):

    # udf is not available since it can not to pickle api call
    microBatchDF = microBatchDF.toPandas()
    microBatchDF["text_chunks"] = microBatchDF["content"].apply(extract_text_from_pdf)
    microBatchDF = spark.createDataFrame(microBatchDF)    

    # Assign row numbers
    window_spec = Window.partitionBy("ingestion_time", "path").orderBy("text_chunk")

    microBatchDF = (microBatchDF.withColumn("ingestion_time", F.current_timestamp())
                 #.withColumn("text_chunks", extract_text_udf(F.col("content")))  # for only functional text split
                 .select(
                     F.col("ingestion_time"),
                     F.col("text_chunks"),
                     F.col("path"),
                 )
                 .withColumn("text_chunk", F.explode(F.col("text_chunks"))).drop("text_chunks")
                 .withColumn("row_number", F.row_number().over(window_spec))
                 )
    # udf is slower for small overhead, so we can use pandas
    microBatchDF = microBatchDF.toPandas()
    microBatchDF["text_vector"] = microBatchDF["text_chunk"].apply(precompute_embed)
    microBatchDF = spark.createDataFrame(microBatchDF)    

    (microBatchDF.write.format("delta")
                 .mode("append")
                 .option("overwriteSchema", "true")
                 .saveAsTable("portfolio.twinllm.silver_pdf_table")
                 )

# COMMAND ----------

query = (spark.readStream.table("portfolio.twinllm.bronze_pdf_table")
                        .writeStream
                         .foreachBatch(batch_upsert)
                         .option("checkpointLocation", "dbfs:/twinllm/checkpoint/silver/pdfs")
                         .trigger(availableNow=True)
                         .start()
)

query.awaitTermination()

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM portfolio.twinllm.silver_pdf_table

# COMMAND ----------

#%sql
#DROP TABLE portfolio.twinllm.silver_pdf_table

# COMMAND ----------

#dbutils.fs.rm("dbfs:/twinllm/checkpoint/silver/pdfs", True)
