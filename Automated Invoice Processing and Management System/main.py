import os
import json
import streamlit as st
from langchain_community.llms import Bedrock
from langchain_community.embeddings import BedrockEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import JSONLoader
import boto3

os.environ["AWS_ACCESS_KEY_ID"] = "xxx"
os.environ["AWS_SECRET_ACCESS_KEY"] = "xxx"
os.environ["AWS_DEFAULT_REGION"] = "us-east-1"



def get_index():
    embeddings = BedrockEmbeddings()
    loader = JSONLoader(
        file_path="invoices.json",
        jq_schema='.[]',
        content_key='invoice_id',
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

def analyze_invoice(invoice):
    # Prepare the prompt
    prompt = f"Invoice Details:\n{invoice}\n\nExtract key information and provide a summary."

    # Create a Bedrock runtime client
    client = boto3.client("bedrock-runtime", region_name=os.environ["AWS_DEFAULT_REGION"])

    # Define the model parameters
    payload = {
        "prompt": prompt,
        "maxTokens": 1024,
        "temperature": 0,
        "topP": 0.5,
        "stopSequences": [],
        "countPenalty": {"scale": 0},
        "presencePenalty": {"scale": 0},
        "frequencyPenalty": {"scale": 0}
    }

    # Invoke the model
    response = client.invoke_model(
        modelId="ai21.j2-ultra-v1",
        contentType="application/json",
        accept="application/json",
        body=json.dumps(payload)
    )

    # Read and parse the response
    response_body = response["body"].read().decode("utf-8")
    result = json.loads(response_body)

    # Extract the model's completion text
    completion_text = result.get("completion", "")
    return completion_text

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

st.title("Automated Invoice Processing and Management System")

if 'vector_index' not in st.session_state:
    with st.spinner("Indexing invoices..."):
        st.session_state.vector_index = get_index()

input_invoice = st.text_area("Enter an invoice entry:")
analyze_button = st.button("Analyze", type="primary")
if analyze_button:
    with st.spinner("Analyzing..."):
        analysis = analyze_invoice(invoice=input_invoice)
        st.markdown(f"### Invoice Entry: {input_invoice}")
        st.write(f"Analysis: {analysis}")
