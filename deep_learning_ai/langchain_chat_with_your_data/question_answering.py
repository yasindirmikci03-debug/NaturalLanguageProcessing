import os
import openai
import sys
sys.path.append('../..')

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key  = os.environ['OPENAI_API_KEY']


from langchain.vectorstores import chroma
from langchain.embeddings.openai import OpenAIEmbeddings

persist_directory = "docs/chroma"
embedding = OpenAIEmbeddings()
vectordb = chroma(persist_directory= persist_directory,embedding_function = embedding)

question = "What are major topics for this class?"
docs = vectordb.similarity_search(question,k=3)

from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(model_name = "gpt-3.5-turbo",temperature = 0)

from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(llm,retriever = vectordb.as_retriever())

result = qa_chain({"query" : question})

print(result["result"])

from langchain.prompts import PromptTemplate

# build prompt
template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. 
{context}
Question: {question}
Helpful Answer:"""

QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

qa_chain = RetrievalQA.from_chain_type(llm,retriever = vectordb.as_retriever(),return_source_documents = True,chain_type_kwargs={"prompt" : QA_CHAIN_PROMPT})

question = "Is probability a class topic?"

result = qa_chain({"query" : question})

print(result["result"])

print(result["source_documents"][0])


qa_chain_mr = RetrievalQA.from_chain_type(llm,retriever = vectordb.as_retriever(),chain_type="map_reduce")

result = qa_chain_mr({"query":question})

print(result[result])

#import os
#os.environ["LANGCHAIN_TRACING_V2"] 
#os.environ["LANGCHAIN_ENDPOINT"] 
#os.environ["LANGCHAIN_API_KEY"] 

qa_chain_mr = RetrievalQA.from_chain_type(llm,retriever = vectordb.as_retriever(),chain_type="refine")

result = qa_chain_mr({"query" : question})

print(result[result])

