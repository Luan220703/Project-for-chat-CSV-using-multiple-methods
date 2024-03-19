
import streamlit as st 
from lida import Manager, TextGenerationConfig , llm  
from dotenv import load_dotenv
import os
import openai
from PIL import Image
from io import BytesIO
import base64
from langchain.chat_models import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
import pandas as pd

load_dotenv()
os.environ['OPENAI_API_KEY'] = ''
openai.api_key = os.environ['OPENAI_API_KEY']

def base64_to_image(base64_string):
    # Decode the base64 string
    byte_data = base64.b64decode(base64_string)
    
    # Use BytesIO to convert the byte data to image
    return Image.open(BytesIO(byte_data))


lida = Manager(text_gen = llm("openai"))
textgen_config = TextGenerationConfig(n=1, temperature=0.5, model="gpt-3.5-turbo-0301", use_cache=True) # lệnh này để tạo ra một config cho model của openAI

menu = st.sidebar.selectbox("Choose an Option", ["Summarize", "Question based Graph","ChatBot","Handling Missing Values"]) # lựa chọn menu


# nếu chọn menu là Summarize thì sẽ hiển thị ra màn hình một subheader là Summarization of your Data
if menu == "Summarize":
    st.subheader("Summarization of your Data")
    file_uploader = st.file_uploader("Upload your CSV", type="csv") # dạng .csv
    if file_uploader is not None:
        path_to_save = "filename.csv" # tên file
        with open(path_to_save, "wb") as f: # mở file
            f.write(file_uploader.getvalue()) # ghi file
        summary = lida.summarize("filename.csv", summary_method="default", textgen_config=textgen_config) # tóm tắt file csv bằng gọi hàm summarize
        st.write(summary) # hiển thị ra màn hình
        goals = lida.goals(summary, n=2, textgen_config=textgen_config) # gọi hàm goals để tạo ra một list các mục tiêu
        for goal in goals:
            st.write(goal) # hiển thị ra màn hình
        i = 0 # gán i = 0
        library = "seaborn" # thư viện để vẽ biểu đồ
        textgen_config = TextGenerationConfig(n=1, temperature=0.2, use_cache=True) # tạo ra một config cho model của openAI
        charts = lida.visualize(summary=summary, goal=goals[i], textgen_config=textgen_config, library=library)   # gọi hàm visualize để tạo ra biểu đồ
        img_base64_string = charts[0].raster # chuyển biểu đồ thành dạng base64
        img = base64_to_image(img_base64_string) # chuyển base64 thành hình ảnh
        st.image(img) # hiển thị hình ảnh ra màn hình

# nếu chọn menu là Question based Graph thì sẽ hiển thị ra màn hình một subheader là Query your Data to Generate Graph        
elif menu == "Question based Graph":
    st.subheader("Query your Data to Generate Graph")
    file_uploader = st.file_uploader("Upload your CSV", type="csv") # dạng .csv
    if file_uploader is not None:
        df = pd.read_csv(file_uploader)
        if st.checkbox('Show data'): # đọc file csv
            st.write(df)
        path_to_save = "filename1.csv" # tên file
        with open(path_to_save, "wb") as f: # mở file
            f.write(file_uploader.getvalue()) # ghi file
         # hiển thị ra màn hình
        text_area = st.text_area("Query your Data to Generate Graph", height=200) # tạo ra một text area để nhập query
        if st.button("Generate Graph"): # nếu nhấn nút Generate Graph
            if len(text_area) > 0: 
                st.info("Your Query: " + text_area) # hiển thị ra màn hình query
                lida = Manager(text_gen = llm("openai"))  # tạo ra một manager
                textgen_config = TextGenerationConfig(n=1, temperature=0.2, use_cache=True) # tạo ra một config cho model của openAI
                summary = lida.summarize("filename1.csv", summary_method="default", textgen_config=textgen_config) 
                user_query = text_area
                charts = lida.visualize(summary=summary, goal=user_query, textgen_config=textgen_config)   # gọi hàm visualize để tạo ra biểu đồ
                charts[0] 
                image_base64 = charts[0].raster # chuyển biểu đồ thành dạng base64
                img = base64_to_image(image_base64) # chuyển base64 thành hình ảnh
                st.image(img) # hiển thị hình ảnh ra màn hình
elif menu == "ChatBot":
    st.title("ChatCSV  💬")
    file_uploader = st.file_uploader("Upload a CSV file", type="csv")
    if file_uploader is not None:
        df = pd.read_csv(file_uploader)
        if st.checkbox('Show data'): # đọc file csv
            st.write(df)
        path_to_save = "filename1.csv" # tên file
        with open(path_to_save, "wb") as f: # mở file
            f.write(file_uploader.getvalue()) # ghi file
            
        user_input = st.text_input(
                "Ask any statistical question related to your CSV"
            )
        llm = llm = ChatOpenAI(model_name='gpt-3.5-turbo-0613', openai_api_key=os.environ['OPENAI_API_KEY']) # tạo ra một model chatbot
        agent = create_pandas_dataframe_agent(llm, df, verbose=True, agent_type=AgentType.OPENAI_FUNCTIONS)
        if user_input:
            response = agent.run(user_input)
            st.write(response)

else:
    # sử dụng kỹ thuật prompt để điền vào các giá trị còn thiếu
    st.subheader("Handling missing values")
    file_uploader = st.file_uploader("Upload your CSV", type="csv")
    if file_uploader is not None:
        df = pd.read_csv(file_uploader)
        if st.checkbox('Show data'):
            st.write(df)
        st.write(df.isnull().sum())
        # sử dụng kỹ thuật prompt để điền vào bằng openai
        prompt = st.text_area("Enter the prompt", height=200)
        llm = llm = ChatOpenAI(model_name='gpt-3.5-turbo-0613', openai_api_key=os.environ['OPENAI_API_KEY']) # tạo ra một model chatbot
        agent = create_pandas_dataframe_agent(llm, df, verbose=True, agent_type=AgentType.OPENAI_FUNCTIONS,agent_executor_kwargs={"handle_parsing_errors": True})
        response = agent.run(prompt)
        # nếu agent đã fill the missing values thì xuất ra file
        if response:
            st.write(response)
            st.write("Do you want to save the filled data?")
            if st.button("Save"):
                df.to_csv("filled_data.csv", index=False)
                st.write("Data saved successfully")
                
            st.write("Do you want to see the filled data?")
            if st.button("Yes"):
                st.write(df)
            # xuất ra file csv mới
            st.write("Do you want to download the filled data?")
            if st.button("Download"):
                csv = df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="filled_data.csv">Download csv file</a>'
                st.markdown(href, unsafe_allow_html=True)




