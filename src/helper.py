
import os
from git import Repo
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import LanguageParser
from langchain.text_splitter import Language
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings



# Clone git hub repository 

def repo_ingestion(repo_url):
    os.makedirs("repo",exist_ok=True)
    repo_path="repo/"
    Repo.clone_from(repo_url,to_path=repo_path)




#Loading repositories 
def load_repo(repo_path):
    loader = GenericLoader.from_filesystem(repo_path,
                                        glob = "**/*",
                                       suffixes=[".py"],
                                       parser = LanguageParser(language=Language.PYTHON, parser_threshold=500))
                                       
    documents = loader.load()
    return documents

# Splitting the text     
def text_splitter(documents):
    splitter = RecursiveCharacterTextSplitter.from_language(language=Language.PYTHON,chunk_size=500,chunk_overlap=20)
    docs_chunk = splitter.split_documents(documents=documents)    

    return docs_chunk

#Loading embeddings

def load_embedding():
    embeddings = OpenAIEmbeddings(disallowed_special=())
    return embeddings
