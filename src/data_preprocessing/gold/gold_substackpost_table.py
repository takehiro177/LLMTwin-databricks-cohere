# Databricks notebook source
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# COMMAND ----------

window = Window.partitionBy("post_id").orderBy(F.col("ingestion_time"))

(spark.read.table("portfolio.twinllm.silver_substackpost_table")
            .withColumn("rank", F.rank().over(window))
            .filter(F.col("rank") == 1)
            .drop("rank")
            .select(
               F.col("post_id"),
               F.col("ingestion_time"),
               F.col("text"),
               F.col("title"),
               F.col("subtitle")
            )
            .dropna(subset=["title"])
         .write
            .format("delta")
            .mode("overwrite")
            .option("overwriteSchema", "True")
            .saveAsTable("portfolio.twinllm.gold_substackpost_table")
)

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM portfolio.twinllm.gold_substackpost_table

# COMMAND ----------

# MAGIC %sql
# MAGIC DESCRIBE TABLE portfolio.twinllm.gold_substackpost_table;

# COMMAND ----------

# MAGIC %sql
# MAGIC DESCRIBE EXTENDED portfolio.twinllm.gold_substackpost_table;

# COMMAND ----------

# MAGIC %sql
# MAGIC DESCRIBE DETAIL portfolio.twinllm.gold_substackpost_table;

# COMMAND ----------

#%sql
#DROP TABLE gold_substackpost_table
