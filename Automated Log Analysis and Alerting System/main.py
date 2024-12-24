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
        "countPenalty": {"scale":0},
        "presencePenalty": {"scale":0},
        "frequencyPenalty": {"scale":0}

    }

    llm = Bedrock(
        model_id="ai21.j2-ultra-v1",
        region_name="us-east-1",
        model_kwargs=model_kwargs
    )

    return llm

def get_index():
    embeddings = BedrockEmbeddings()
    loader = JSONLoader(
        file_path="logs.json",
        jq_schema='.[]',
        content_key='log',
        metadata_func=lambda record, metadata: metadata.update(record) or metadata
    )

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " "],
        chunk_size=8000,
        chunk_overlap=0
    )

    index_creator = VectorstoreIndexCreator(
        vectorstore_cls=FAISS,
        embedding=embeddings,
        text_splitter=text_splitter
    )

    index_from_loader = index_creator.from_loaders([loader])

    return index_from_loader

def analyze_log(log):
    llm = get_llm()
    prompt = f"{log}\n\nAnalyze the log and determine if there are any issues. If so, provide a summary and suggest actions."
    analysis = llm.invoke(prompt)
    return analysis

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

st.title("Automated Log Analysis and Alerting System")

if 'vector_index' not in st.session_state:
    with st.spinner("Indexing logs..."):
        st.session_state.vector_index = get_index()

input_log = st.text_area("Enter a log entry:")
analyze_button = st.button("Analyze", type="primary")
if analyze_button:
    with st.spinner("Analyzing..."):
        analysis = analyze_log(log=input_log)
        st.markdown(f"### Log Entry: {input_log}")
        st.write(f"Analysis: {analysis}")
