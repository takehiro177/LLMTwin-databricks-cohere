# Databricks notebook source
input_path = "dbfs:/twinllm/raw/gitrepos"

query = (spark.readStream.format("cloudFiles")
            .option("cloudFiles.format", "json")
            .option("multiline", "true")
            .option("cloudFiles.schemaLocation", "dbfs:/twinllm/checkpoint/bronze/gitrepos")
            .option("cloudFiles.inferColumnTypes", "True")
            .load(input_path)
        .writeStream
            .format("delta")
            .outputMode("append")
            .option("checkpointLocation", "dbfs:/twinllm/checkpoint/bronze/gitrepos")
            .trigger(availableNow=True)
            .table("portfolio.twinllm.bronze_gitrepo_table")
    )

query.awaitTermination()

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM portfolio.twinllm.bronze_gitrepo_table

# COMMAND ----------

#%sql
#DROP TABLE portfolio.twinllm.bronze_gitrepo_table

# COMMAND ----------

#dbutils.fs.rm("dbfs:/twinllm/checkpoint/bronze/gitrepos", True)
