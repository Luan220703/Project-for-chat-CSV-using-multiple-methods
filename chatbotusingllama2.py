# ta sẽ tạo chatbot bằng cách sử dụng llama2 thay vì openai
# sau đó sẽ thay thế các hàm của openai bằng llama2 và visualize các câu lệnh bằng cách sử dụng thư viện lida

# nhập vào các thư viện cần thiết
from langchain.llms import CTransformers
import torch
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import sys
from lida import Manager, TextGenerationConfig , llm
import lida
import os
import streamlit as st
from PIL import Image
from io import BytesIO
import base64
import pandas as pd

# hàm chuyển đổi base64 thành hình ảnh
def base64_to_image(base64_string):
    # Decode the base64 string
    byte_data = base64.b64decode(base64_string)
    
    # Use BytesIO to convert the byte data to image
    return Image.open(BytesIO(byte_data))

# tạo một config cho model của llama2
model_name = "TheBloke/Llama-2-7B-GGML"
text_gen = llm(provider="hf", model=model_name, device_map="auto")
lida = Manager(llm=text_gen)


menu = st.sidebar.selectbox("Choose an Option", ["Summarize", "Question based Graph","ChatBot"]) # lựa chọn menu

# nếu chọn menu là Question based Graph thì sẽ hiển thị ra màn hình một subheader là Query your Data to Generate Graph
if menu == "Question based Graph":
    st.subheader("Query your Data to Generate Graph")
    file_uploader = st.file_uploader("Upload your CSV", type="csv") # dạng .csv
    if file_uploader is not None:
        path_to_save = "filename1.csv" # tên file
        with open(path_to_save, "wb") as f: # mở file
            f.write(file_uploader.getvalue()) # ghi file
        df = pd.read_csv('filename1.csv') # đọc file csv
        st.write(df) # hiển thị ra màn hình
        text_area = st.text_area("Query your Data to Generate Graph", height=200)
        if st.button("Generate Graph"):
            if len(text_area) > 0:
                st.info("Your Query: " + text_area) # hiển thị ra màn hình query
                lida = Manager(text_gen = llm)
                textgen_config = TextGenerationConfig(n=1, temperature=0.2, use_cache=True)
                summary = lida.summarize("filename1.csv", summary_method="default", textgen_config=textgen_config)
                user_query = text_area
                charts = lida.visualize(summary=summary, goal=user_query, textgen_config=textgen_config)
                charts[0]
                image_base64 = charts[0].raster
                img = base64_to_image(image_base64)
                st.image(img)
else:
    st.subheader("ChatCSV")
    file_uploader = st.file_uploader("Upload your CSV", type="csv")
    if file_uploader is not None:
        df = pd.read_csv(file_uploader)
        st.write(df)
        lida = Manager(text_gen = llm)
        textgen_config = TextGenerationConfig(n=1, temperature=0.5, model="TheBloke/Llama-2-7B-GGML", use_cache=True)
        st.text("Enter your message")
        message = st.text_input("Your message")
        if st.button("Send"):
            response = lida.get_response(message, textgen_config=textgen_config)
            st.write(response)






