# Databricks notebook source
# Retrieve the secret from the secret scope
storage_account_key = dbutils.secrets.get(scope="", key="")
spark.conf.set("fs.azure.account.key.CONTAINER_LOCATION.dfs.core.windows.net", storage_account_key)

# COMMAND ----------

query = (spark.readStream.format("cloudFiles")
                         .option("cloudFiles.format", "binaryFile")
                         .option("pathGlobfilter", "*.pdf")
                         .option("cloudFiles.schemaLocation", "dbfs:/twinllm/checkpoint/bronze/pdfs")
                         .load("abfss://CONTAINER_LOCATION.dfs.core.windows.net/raw_pdf/")
                        .writeStream
                         .format("delta")
                         .outputMode("append")
                         .option("checkpointLocation", "dbfs:/twinllm/checkpoint/bronze/pdfs")
                         .trigger(availableNow=True)
                         .table("portfolio.twinllm.bronze_pdf_table")
)

query.awaitTermination()

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM portfolio.twinllm.bronze_pdf_table

# COMMAND ----------

#%sql
#DROP TABLE portfolio.twinllm.bronze_pdf_table

# COMMAND ----------

#dbutils.fs.rm("dbfs:/twinllm/checkpoint/bronze/pdfs", True)
