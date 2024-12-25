import os
import streamlit as st
from langchain_community.embeddings import BedrockEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders.csv_loader import CSVLoader
from botocore.exceptions import NoCredentialsError, PartialCredentialsError

# Set AWS credentials and region
os.environ["AWS_ACCESS_KEY_ID"] = "your_access_key_id"
os.environ["AWS_SECRET_ACCESS_KEY"] = "your_secret_access_key"
os.environ["AWS_REGION"] = "us-east-1"

# Streamlit page configuration
st.set_page_config(page_title="Employee Knowledge Base Search", layout="wide")

# Hide Streamlit style
hide_streamlit_style = """
<style>
div[data-testid="stToolbar"] {
    visibility: hidden;
    height: 0%;
    position: fixed;
}
div[data-testid="stDecoration"] {
    visibility: hidden;
}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

def get_index():
    """Creates and returns an in-memory vector store to be used in the application"""
    try:
        embeddings = BedrockEmbeddings(region_name=os.getenv("AWS_REGION"))
        loader = CSVLoader(file_path="knowledge_base.csv")
        index_creator = VectorstoreIndexCreator(
            vectorstore_cls=FAISS,
            embedding=embeddings,
            text_splitter=CharacterTextSplitter(chunk_size=300, chunk_overlap=0),
        )
        index_from_loader = index_creator.from_loaders([loader])
        return index_from_loader
    except (NoCredentialsError, PartialCredentialsError) as e:
        st.error("Could not load credentials to authenticate with AWS client. Please check that credentials in the specified profile name are valid.")
        raise e
    except Exception as e:
        st.error(f"An error occurred: {e}")
        raise e

def get_similarity_search_results(index, question):
    results = index.vectorstore.similarity_search_with_score(question)
    flattened_results = [{"content": res[0].page_content, "score": res[1]} for res in results]
    return flattened_results

def get_embedding(text):
    try:
        embeddings = BedrockEmbeddings(region_name=os.getenv("AWS_REGION"))
        return embeddings.embed_query(text)
    except (NoCredentialsError, PartialCredentialsError) as e:
        st.error("Could not load credentials to authenticate with AWS client. Please check that credentials in the specified profile name are valid.")
        raise e
    except Exception as e:
        st.error(f"An error occurred: {e}")
        raise e

st.title("Employee Knowledge Base Search")

if 'vector_index' not in st.session_state:
    with st.spinner("Indexing document..."):
        st.session_state.vector_index = get_index()

input_text = st.text_input("Ask a question about the company:")
go_button = st.button("Go", type="primary")

if go_button:
    with st.spinner("Working..."):
        response_content = get_similarity_search_results(index=st.session_state.vector_index, question=input_text)
        st.table(response_content)
        raw_embedding = get_embedding(input_text)
        with st.expander("View question embedding"):
            st.json(raw_embedding)
