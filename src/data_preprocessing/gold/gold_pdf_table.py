# Databricks notebook source
# Library
from pyspark.sql import functions as F

# COMMAND ----------

(spark.read.table("portfolio.twinllm.silver_pdf_table")
            .withColumn("pdf_chunk_id", F.concat_ws("_", F.col("path"), F.col("row_number")))
            .select(
               F.col("pdf_chunk_id"),
               F.col("text_chunk").alias("text"),
               F.col("text_vector"),
            )
         .write
            .format("delta")
            .mode("overwrite")
            .option("overwriteSchema", "True")
            .saveAsTable("portfolio.twinllm.gold_pdf_table")
)

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM portfolio.twinllm.gold_pdf_table

# COMMAND ----------

#%sql
#DROP TABLE portfolio.twinllm.gold_pdf_table
