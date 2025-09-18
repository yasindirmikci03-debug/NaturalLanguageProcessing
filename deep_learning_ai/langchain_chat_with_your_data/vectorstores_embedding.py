import os
import openai
import sys
sys.path.append('../..')

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key  = os.environ['OPENAI_API_KEY']

from langchain.document_loaders import PyPDFLoader

loaders = [
    PyPDFLoader("docs/cs229_lectures/MachineLearning-Lecture01.pdf"),
    PyPDFLoader("docs/cs229_lectures/MachineLearning-Lecture01.pdf"),
    PyPDFLoader("docs/cs229_lectures/MachineLearning-Lecture02.pdf"),
    PyPDFLoader("docs/cs229_lectures/MachineLearning-Lecture03.pdf")
]

docs = []

for loader in loaders:
    docs.extend(loader.load())

# Split
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1500,chunk_overlap = 150)

splits = text_splitter.split_documents(docs)


# Embeddings

from langchain.embeddings.openai import OpenAIEmbeddings

embedding = OpenAIEmbeddings()

sentence1 = "i like dogs"
sentence2 = "i like canines"
sentence3 = "the weather is ugly outside"

embedding1 = embedding.embed_query(sentence1)
embedding2 = embedding.embed_query(sentence2)
embedding3 = embedding.embed_query(sentence3)

import numpy as np

np.dot(embedding1,embedding2)
np.dot(embedding1,embedding3)
np.dot(embedding2,embedding3)


# Vectorstores

from langchain.vectorstores import chroma

persist_directory = "docs/chroma"

vectordb = chroma.from_documents(documents = splits,embedding = embedding,persist_directory = persist_directory)

# Similarity Search

question = "is there an email i can ask for help"

docs = vectordb.similarity_search(question,k=3)

print(docs[0].page_content)

# Save this

vectordb.persist()



