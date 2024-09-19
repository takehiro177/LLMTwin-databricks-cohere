# Databricks notebook source
# Library
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# COMMAND ----------

lowQuality_data_list = [""]

window = Window.partitionBy("gitrepo_file_id").orderBy(F.col("timestamp"))

(spark.read.table("portfolio.twinllm.silver_gitrepo_table")
            .withColumn("timestamp", F.to_timestamp(F.col("timestamp"), "yyyyMMddHHmmss"))
            .withColumn("gitrepo_file_id", F.concat_ws("-", F.col("file_name"), F.col("repo"), F.col("timestamp").cast("string")))
            .withColumn("rank", F.rank().over(window))
            .filter(F.col("rank") == 1)  # filter only latest version
            .drop("rank")
            .filter(~F.col("gitrepo_file_id").isin(lowQuality_data_list)) # remove low quality data
            .select(
               F.col("content"),
               F.col("file_name"),
               F.col("repo"),
               F.col("timestamp"),
               F.col("text"),
               F.col("text_vector"),
               F.col("gitrepo_file_id"),
            )
         .write
            .format("delta")
            .mode("overwrite")
            .option("overwriteSchema", "True")
            .saveAsTable("portfolio.twinllm.gold_gitrepo_table")
)


# COMMAND ----------

# MAGIC %sql
# MAGIC DESCRIBE TABLE portfolio.twinllm.gold_gitrepo_table;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT text, repo, file_name, gitrepo_file_id FROM portfolio.twinllm.gold_gitrepo_table;

# COMMAND ----------

#%sql
#DROP TABLE gold_gitrepo_table
