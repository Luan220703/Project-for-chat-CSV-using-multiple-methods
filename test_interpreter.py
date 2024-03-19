# nhập các thư viện 
import streamlit as st
import pandas as pd
from langchain.document_loaders.csv_loader import CSVLoader
import openai 
import os 
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

load_dotenv()
os.environ['OPENAI_API_KEY'] = ''
openai.api_key = os.environ['OPENAI_API_KEY']

# Loading tập data
loader = CSVLoader(r'C:\Users\Administrator\Downloads\train.csv')
df = loader.load() 



#embedding vector 
embedding = OpenAIEmbeddings()


#vector store
db = Chroma.from_documents(df,embedding)
query = "How many females and males in this data?"
docs = db.similarity_search(query)
docs[0].page_content









