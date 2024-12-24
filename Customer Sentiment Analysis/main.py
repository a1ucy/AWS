import os
import streamlit as st
from langchain_community.llms import Bedrock
from langchain_community.embeddings import BedrockEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import JSONLoader

os.environ["AWS_ACCESS_KEY_ID"] = "xxx"
os.environ["AWS_SECRET_ACCESS_KEY"] = "xxx"
os.environ["AWS_DEFAULT_REGION"] = "us-east-1"

def get_llm():
    model_kwargs = {
        "maxTokens": 1024,
        "temperature": 0,
        "topP": 0.5,
        "stopSequences": [],
        "countPenalty": {"scale": 0},
        "presencePenalty": {"scale": 0},
        "frequencyPenalty": {"scale": 0}
    }

    llm = Bedrock(
        model_id="ai21.j2-ultra-v1",
        model_kwargs=model_kwargs
    )

    return llm

def get_sentiment_analysis(review):
    llm = get_llm()
    prompt = f"{review}\n\nAnalyze the sentiment of the above review."
    sentiment = llm.invoke(prompt)
    return sentiment

st.set_page_config(layout="wide")

st.markdown("""
<style>
.reportview-container {
    margin-top: -2em;
}
#MainMenu {
    visibility: hidden;
}
.stDeployButton {
    display: none;
}
footer {
    visibility: hidden;
}
#stDecoration {
    display: none;
}
</style>
""", unsafe_allow_html=True)

st.title("Customer Review Sentiment Analysis")

input_text = st.text_input("Enter a customer review:")
go_button = st.button("Analyze", type="primary")
if go_button:
    with st.spinner("Analyzing..."):
        sentiment = get_sentiment_analysis(review=input_text)
        st.markdown(f"### Review: {input_text}")
        st.write(f"Sentiment: {sentiment}")
