# Databricks notebook source
# MAGIC %pip install -U --quiet mlflow-skinny mlflow mlflow[gateway]
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

## create mosic ai model serving for cohere embedding

import mlflow.deployments

client = mlflow.deployments.get_deploy_client("databricks")

client.create_endpoint(
    name="cohere-embed-endpoint",
    config={
        "served_entities": [
            {
                "name": "test",
                "external_model": {
                    "name": "embed-english-v3.0",
                    "provider": "cohere",
                    "task": "llm/v1/embeddings",
                    "cohere_config": {
                        "cohere_api_key": "{{secrets/}}",
                    }
                }
            }
        ]
    }
)

# COMMAND ----------

## create mosic ai model serving for cohere chat

import mlflow.deployments

client = mlflow.deployments.get_deploy_client("databricks")

client.create_endpoint(
    name="cohere-chat-endpoint",
    config={
        "served_entities": [
            {
                "name": "test",
                "external_model": {
                    "name": "command-nightly",
                    "provider": "cohere",
                    "task": "llm/v1/chat",
                    "cohere_config": {
                        "cohere_api_key": "{{secrets/}}",
                    }
                }
            }
        ]
    }
)

# COMMAND ----------

## create mosic ai model serving for cohere finetune chat model

import mlflow.deployments

client = mlflow.deployments.get_deploy_client("databricks")

client.create_endpoint(
    name="cohere-finetuned-chat-endpoint",
    config={
        "served_entities": [
            {
                "name": "test",
                "external_model": {
                    "name": "YOUR_FINETUNE_MODEL_NAME",
                    "provider": "cohere",
                    "task": "llm/v1/chat",
                    "cohere_config": {
                        "cohere_api_key": "{{secrets/}}",
                    }
                }
            }
        ]
    }
)
