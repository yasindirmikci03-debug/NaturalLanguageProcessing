import os

from dotenv import load_dotenv,find_dotenv
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.indexes import VectorstoreIndexCreator
#from IPython.display import display, Markdown
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings

_ = load_dotenv(find_dotenv())

file = "OutdoorClothingCatalog_1000.csv"
loader = CSVLoader(file_path = file)

index = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch
).from_loaders([loader])

query ="Please list all your shirts with sun protection \
in a table in markdown and summarize each one."

llm_replacement_model = OpenAI(temperature=0, 
                               model='gpt-3.5-turbo-instruct')

response = index.query(query, 
                       llm = llm_replacement_model)

# display(Markdown(query))

##### Step by step

docs = loader.load()

embeddings = OpenAIEmbeddings()

embed = embeddings.embed_query("Hi my name is Yasin")

db = DocArrayInMemorySearch.from_documents(docs,embeddings)

query = "Please sugget a shirt with sunblocking"

docs = db.similarity_search(query)

retriever = db.as_retriever()

llm = ChatOpenAI(temperature = 0.0)

qdocs = "".join([docs[i].page_content for i in range(len(docs))])

response = llm.call_as_llm(f"{qdocs} Question: Please list all your \
shirts with sun protection in a table in markdown and summarize each one.") 

# display(Markdown(response))

qa_stuff = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever = retriever,
    verbose = True,
)

query =  "Please list all your shirts with sun protection in a table \
in markdown and summarize each one."

response = qa_stuff.run(query)

# display(Markdown(response))

response = index.query(query,llm=llm)

index = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch,
    embedding=embeddings,
).from_loaders([loader])