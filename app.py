import streamlit as st 
from pandasai.llm.openai import OpenAI
from dotenv import load_dotenv
import os
import pandas as pd
from pandasai import SmartDataframe
import openai
load_dotenv()
os.environ['OPENAI_API_KEY'] = ''
openai.api_key = os.environ['OPENAI_API_KEY']
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TKAgg')

st.set_page_config(layout='wide')
st.title("ChatCSV - Chat with your CSV files")

input_csv = st.file_uploader("Upload your CSV file here", type=['csv'])

if input_csv is not None:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.info("CSV Uploaded Successfully")
        data = pd.read_csv(input_csv)
        st.dataframe(data, use_container_width=True)

    with col2:
        st.info("Chat Below")
        input_text = st.text_area("Enter your query")

        if input_text is not None:
            if st.button("Done"):
                st.info("Your Query: " + input_text)
                # Initialize OpenAI and PandasAI
                llm = OpenAI(api_token=openai.api_key)
                sdf = SmartDataframe(data,config={'llm':llm})
                # Run chat_with_csv function
                result = sdf.chat(input_text)
                st.success(result)
                
