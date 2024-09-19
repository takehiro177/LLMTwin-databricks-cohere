# Databricks notebook source
# Retrieve the secret from the secret scope
storage_account_key = dbutils.secrets.get(scope="", key="")
spark.conf.set("fs.azure.account.key.CONTAINER_LOCATION.dfs.core.windows.net", storage_account_key)

# COMMAND ----------

from pyspark.sql.functions import input_file_name

# Configure Auto Loader to read from your Azure Blob Storage Gen2 with folder structure
query = (spark.readStream
            .format("cloudFiles")
            .option("cloudFiles.format", "csv")
            .option("cloudFiles.schemaLocation", "dbfs:/twinllm/checkpoint/bronze/substackposts")
            .option("header", "true") 
            .option("pathGlobFilter", "*.csv")
            .load("abfss://substack-datasource@CONTAINER_LOCATION.dfs.core.windows.net/*/")
            .withColumn("filePath", input_file_name())
      .writeStream
            .format("delta")
            .outputMode("append")
            .option("checkpointLocation", "dbfs:/twinllm/checkpoint/bronze/substackposts")
            .trigger(availableNow=True)
            .table("portfolio.twinllm.bronze_substackpost_table")
      )

query.awaitTermination()

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM portfolio.twinllm.bronze_substackpost_table

# COMMAND ----------

#%sql
#DROP TABLE portfolio.twinllm.bronze_substackpost_table

# COMMAND ----------

#dbutils.fs.rm("dbfs:/twinllm/checkpoint/bronze/substackposts", True)
