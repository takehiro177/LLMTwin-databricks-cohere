# TwinLLM-databricks-cohere
## Overview
This repository provides an end-to-end LLMOps deployment fo Cohere LLM on Databricks, leveraging Cohere on Mosaic AI. The primary use case of this project is to create a TwinLLM, which learns users’ technical knowledge by finetuning from various resources such as GitHub repositories, PDFs, and more. The TwinLLM is designed to assist users by performing technical blog writing tasks.

The data is collecting through streaming process and maintained as workflow for vector database and search as well as finetuning Cohere LLM model.
For further detail, please vist this blog post in substack.


https://github.com/user-attachments/assets/b0ae69a8-1eca-4016-b77b-15924ffbff3e


## Features
* End-to-End Deployment: Seamless integration and deployment of LLMOps with Cohere on Databricks.
* Cohere Integration: Utilizes Cohere’s language models for advanced and high quality cost effective LLM NLP capabilities.
* Mosaic AI: Leverages Mosaic AI for efficient model training and deployment leveraging Cohere LLM models.
* Knowledge Extraction: Learns from various technical resources such as GitHub repositories, PDFs, and Substack.
* Technical Blog Writing: Generates high-quality technical blog posts based on the retrieve knowledge from the resource and finetuned on original writing in Substack post by finetuning Cohere chat model.

## Architecture
The architecture of the TwinLLM project includes the following components:

* Data Ingestion: Collects data from GitHub repositories, PDFs, and other technical resources through structured streaming process.
* Data Processing: Preprocesses the collected data for vector search and creating vector index on Databricks for RAG.
* LLM model finetuning: Utilizes Cohere LLM Platform and finetune Cohere’s language models and Mosaic AI for serving TwinLLM as endpoint.
* Deployment: Deploys the finetuned LLM model on Databricks for real-time and chat agent Application usage in Databricks.
* Blog Generation: Generates technical blog posts based on user queries and the learned knowledge.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

### DISCRAMER
This work does not resprent any parties nor take any responsibilities on its output and result. Please note that using contents in this repository is at your own risk.
