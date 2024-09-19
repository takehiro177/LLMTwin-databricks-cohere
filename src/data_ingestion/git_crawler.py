# Databricks notebook source
# MAGIC %pip install selenium
# MAGIC %pip install nbconvert
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# Library
import os
import shutil
from datetime import datetime
import subprocess
import tempfile
import json
import requests

from selenium import webdriver
from selenium.webdriver.chrome.options import Options

import nbconvert
from bs4 import BeautifulSoup


# COMMAND ----------

class GithubCrawler():
    def __init__(self, ignore=(".git", ".toml", ".lock", ".png", ".jpg", ".csv", ".pyc", ".pdf")) -> None:
        super().__init__()
        self._ignore = ignore

    def extract(self, link: str, **kwargs) -> None:

        repo_name = link.rstrip("/").split("/")[-1]

        local_temp = tempfile.mkdtemp()

        try:
            os.chdir(local_temp)
            subprocess.run(["git", "clone", link])

            repo_path = os.path.join(local_temp, os.listdir(local_temp)[0])

            tree = {}
            for root, dirs, files in os.walk(repo_path):
                dir = root.replace(repo_path, "").lstrip("/")
                if dir.startswith(self._ignore):
                    continue

                for file in files:
                    if file.endswith(self._ignore):
                        continue

                    if file.endswith(".ipynb"):
                        command = ["jupyter", "nbconvert", "--to", "html", root + "/" + file]
                        result = subprocess.run(command, capture_output=True, text=True)
                        py_file = os.path.join(root, file.replace(".ipynb", ".html"))
                        with open(os.path.join(root, py_file), "r", errors="ignore") as f:
                            html_content = f.read()
                            soup = BeautifulSoup(html_content, 'html.parser')
                            tree[file] = soup.get_text()

                    if file.endswith(".py"):
                        with open(os.path.join(root, file), "r", errors="ignore") as f:
                            tree[file] = f.read().replace(" ", "")

                    if file.endswith(".md"):
                        with open(os.path.join(root, file), "r", errors="ignore") as f:
                            tree[file] = f.read().replace(" ", "")

                    if file.endswith(".yaml"):
                        file_path = os.path.join(dir, file)
                        with open(os.path.join(root, file), "r", errors="ignore") as f:
                            tree[file_path] = f.read().replace(" ", "")
                            
                    if file.endswith(".sh"):
                        file_path = os.path.join(dir, file)
                        with open(os.path.join(root, file), "r", errors="ignore") as f:
                            tree[file_path] = f.read().replace(" ", "")

                    if file == "Dockerfile":
                        file_path = os.path.join(dir, file)
                        with open(os.path.join(root, file), "r", errors="ignore") as f:
                            tree[file_path] = f.read().replace(" ", "")

            return tree

        except Exception:
            raise
        finally:
            shutil.rmtree(local_temp)


# COMMAND ----------

directory = '/dbfs/twinllm/raw/gitrepos/'
# Check if the directory exists
if not os.path.exists(directory):
    # Create the directory
    os.makedirs(directory)
    print("Directory created successfully!")
else:
    print("Directory already exists!")
# executed timestamp
timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

# COMMAND ----------

# Replace 'your_username' with your GitHub username
username = 'your_username'

# GitHub API URL to fetch user repositories
url = f'https://api.github.com/users/{username}/repos'

# Send a GET request to the GitHub API
response = requests.get(url)

# list of repository URLs to be crawled
latest_updated_repo_rul_dict = {}
# Check if the request was successful
if response.status_code == 200:
    repos = response.json()
    for repo in repos:
        if repo['private'] == False:
            repo_name = repo['name']
            repo_url = repo['html_url']
            updated_at = datetime.strptime(repo['updated_at'], '%Y-%m-%dT%H:%M:%SZ')
            # Convert to datetime object
            execution_time = datetime.strptime(timestamp, '%Y%m%d%H%M%S')
            difference = execution_time - updated_at
            if difference.days < 30:
                latest_updated_repo_rul_dict[repo_name] = repo_url
else:
    raise Exception(f"Failed to fetch repositories: {response.status_code}")

# COMMAND ----------

latest_updated_repo_rul_dict

# COMMAND ----------

if len(latest_updated_repo_rul_dict) != 0:
    gitcrawl = GithubCrawler()
    json_data = []
    for name, url in latest_updated_repo_rul_dict.items():
        text_contents = gitcrawl.extract(link=url)
        text_to_json = [{"file_name": key, "content": value, "timestamp": timestamp, "repo": name} for key, value in text_contents.items()]
        json_data += text_to_json

    file_name = directory + f"/data_{timestamp}.json"
    with open(file_name, 'w') as file:
        json.dump(json_data, file, indent=4)
    print("json file saved successfully!")

# COMMAND ----------

# MAGIC %fs ls dbfs:/twinllm/raw/gitrepos

# COMMAND ----------

#dbutils.fs.rm("dbfs:/twinllm/raw/gitrepos", True)

# COMMAND ----------


