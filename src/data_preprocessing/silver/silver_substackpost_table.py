# Databricks notebook source
# MAGIC %pip install --quiet beautifulsoup4
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# Retrieve the secret from the secret scope
storage_account_key = dbutils.secrets.get(scope="", key="")
spark.conf.set("fs.azure.account.key.CONTAINER_ACCOUNTS.dfs.core.windows.net", storage_account_key)

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
from azure.storage.blob import BlobServiceClient
from bs4 import BeautifulSoup

def html_to_text(html_path):
    # Azure Blob Storage connection details
    storage_account_name = "your_account_name"
    container_name = "substack-datasource"
    
    # Create BlobServiceClient
    blob_service_client = BlobServiceClient(account_url=f"https://{storage_account_name}.blob.core.windows.net", credential=storage_account_key)
    
    # Get the blob client
    relative_path = html_path.split(".net/")[1]
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=relative_path)
    
    # Download the blob content
    blob_content = blob_client.download_blob().readall()
    
    # Parse HTML content using BeautifulSoup
    soup = BeautifulSoup(blob_content, "html.parser")
    text = soup.get_text()
    
    return text

# Register the UDF but can not serialize
html_to_text_udf = udf(html_to_text, StringType())
spark.udf.register("html_to_text_udf", html_to_text_udf)

def batch_upsert(microBatchDF, batchId):

    (microBatchDF.filter(F.col("is_published").isNotNull())
                 .withColumn("folderPath", F.regexp_replace(F.col("filePath"), "/[^/]+$", ""))
                 .withColumn("html_path", F.concat_ws('', F.col("folderPath"), F.lit("/posts/"), F.col("post_id"), F.lit(".html")))
                 .withColumn("text", html_to_text_udf(F.col("html_path")))
                 .withColumn("ingestion_time", F.current_timestamp())
                 .select(
                     F.col("post_id"),
                     F.col("post_date"),
                     F.col("text"),
                     F.col("is_published"),
                     F.col("email_sent_at"),
                     F.col("inbox_sent_at"),
                     F.col("type"),
                     F.col("audience"),
                     F.col("title"),
                     F.col("subtitle"),
                     F.col("html_path"),
                     F.col("ingestion_time"),
                 )
                .write.format("delta")
                 .mode("append")
                 .option("overwriteSchema", "true")
                 .saveAsTable("portfolio.twinllm.silver_substackpost_table")
                 )

# COMMAND ----------

query = (spark.readStream.table("portfolio.twinllm.bronze_substackpost_table")
         .writeStream
         .foreachBatch(batch_upsert)
         .option("checkpointLocation", "dbfs:/twinllm/checkpoint/silver/substackposts")
         .trigger(availableNow=True)
         .start()
)

query.awaitTermination()


# COMMAND ----------

# MAGIC %sql
# MAGIC DESCRIBE TABLE portfolio.twinllm.silver_substackpost_table;

# COMMAND ----------

#%sql
#DROP TABLE portfolio.twinllm.silver_substackpost_table

# COMMAND ----------

#dbutils.fs.rm("dbfs:/twinllm/checkpoint/silver/substackposts", True)
