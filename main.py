import os
import streamlit as st
import time
from dotenv import load_dotenv
import langchain
import langchain_google_genai
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS

load_dotenv()

st.title("Research Tool")
st.sidebar.title("Article URLs")

urls=[]
file_path = "faiss_index"

for i in range(3):
    url = st.sidebar.text_input(f"URL{i+1}")
    urls.append(url)

process_url_click = st.sidebar.button("Process URLs")

main_place_folder = st.empty()
llm = ChatGoogleGenerativeAI(
    model = "gemini-2.0-flash-001",temperature=0.9, max_tokens=500
)
embeddings= GoogleGenerativeAIEmbeddings(model= "models/embedding-001")

if process_url_click:
    # load data
    loader = UnstructuredURLLoader(urls= urls)
    main_place_folder.text("Loading Data...")
    data = loader.load()

    # split data
    text_split = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', ' '],
        chunk_size=1000
    )
    main_place_folder.text("Splitting Text...")
    docs = text_split.split_documents(data)

    # create embeddings
    vector_index = FAISS.from_documents(data, embeddings)
    main_place_folder.text("Embedding vector...")
    time.sleep(2)

    # save locally
    vector_index.save_local(file_path)

query = main_place_folder.text_input("Question:")
if query:
    if os.path.exists(file_path):
        vector = FAISS.load_local(file_path, embeddings, allow_dangerous_deserialization=True)
        chain= RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever= vector.as_retriever())
        result = chain({"question":query}, return_only_outputs= True)
        st.subheader("Answer:")
        st.write(result["answer"])

        # display sources, if available
        source = result.get("sources")
        if source:
            st.subheader("Sources:")
            source_list = source.split("\n")
            for s in source_list:
                st.write(s)