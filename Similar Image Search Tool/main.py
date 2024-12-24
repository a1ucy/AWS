import os
import boto3
import json
import base64
from io import BytesIO
import streamlit as st
from langchain_community.vectorstores import FAISS
# Calls Bedrock to get a vector from either an image, text, or both

def get_multimodal_vector(input_image_base64=None, input_text=None):
    session = boto3.Session(
        aws_access_key_id='xxx',
        aws_secret_access_key='xxx',
        region_name='us-east-1'
    )

    bedrock = session.client(service_name='bedrock-runtime')

    request_body = {}

    if input_text:
        request_body["inputText"] = input_text

    if input_image_base64:
        request_body["inputImage"] = input_image_base64

    body = json.dumps(request_body)

    response = bedrock.invoke_model(
        body=body,
        modelId="amazon.titan-embed-image-v1",
        accept="application/json",
        contentType="application/json"
    )

    response_body = json.loads(response.get('body').read())
    embedding = response_body.get("embedding")

    return embedding
# Creates a vector from a file
def get_vector_from_file(file_path):
    with open(file_path, "rb") as image_file:
        input_image_base64 = base64.b64encode(image_file.read()).decode('utf8')
   
    vector = get_multimodal_vector(input_image_base64=input_image_base64)
   
    return vector


# Creates a list of (path, vector) tuples from a directory
def get_image_vectors_from_directory(path):
    items = []
   
    if not os.path.exists(path):
        st.error(f"Directory '{path}' does not exist.")
        return items
   
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        vector = get_vector_from_file(file_path)
        items.append((file_path, vector))
   
    return items


# Creates and returns an in-memory vector store to be used in the application
def get_index():
    image_vectors = get_image_vectors_from_directory("./images")  # Use absolute path
   
    if not image_vectors:
        st.error("No vectors were created. Please check the 'images' directory.")
        return None
   
    text_embeddings = [("", item[1]) for item in image_vectors]
    metadatas = [{"image_path": item[0]} for item in image_vectors]
   
    index = FAISS.from_embeddings(
        text_embeddings=text_embeddings,
        embedding=None,
        metadatas=metadatas
    )
   
    return index
# Get a base64-encoded string from file bytes
def get_base64_from_bytes(image_bytes):
    image_io = BytesIO(image_bytes)
    image_base64 = base64.b64encode(image_io.getvalue()).decode("utf-8")
    return image_base64


# Get a list of images based on the provided search term and/or search image
def get_similarity_search_results(index, search_term=None, search_image=None):
    search_image_base64 = get_base64_from_bytes(search_image) if search_image else None
    search_vector = get_multimodal_vector(input_text=search_term, input_image_base64=search_image_base64)
   
    results = index.similarity_search_by_vector(embedding=search_vector)
    results_images = []

    for res in results:  # Load images into list
        with open(res.metadata['image_path'], "rb") as f:
            img = BytesIO(f.read())
            results_images.append(img)
   
    return results_images
# Streamlit application
import streamlit as st

st.set_page_config(layout="wide")

st.markdown("""
<style>
.reportview-container {
    margin-top: -2em;
}
#MainMenu {visibility: hidden;}
.stDeployButton {display:none;}
footer {visibility: hidden;}
#stDecoration {display:none;}
</style>
""", unsafe_allow_html=True)

st.title("Image Search Web App")  # page title

if 'vector_index' not in st.session_state:  # see if the vector index hasn't been created yet
    with st.spinner("Indexing images..."):  # show a spinner while the code in this with block runs
        st.session_state.vector_index = get_index()  # retrieve the index through the supporting library and store in the app's session cache

search_images_tab, find_similar_images_tab = st.tabs(["Image search", "Find similar images"])
with search_images_tab:

    search_col_1, search_col_2 = st.columns(2)

    with search_col_1:
        input_text = st.text_input("Search for:")  # display a multiline text box with no label
        search_button = st.button("Search", type="primary")  # display a primary button

    with search_col_2:
        if search_button:  # code in this if block will be run when the button is clicked
            st.subheader("Results")
            with st.spinner("Searching..."):  # show a spinner while the code in this with block runs
                response_content = get_similarity_search_results(index=st.session_state.vector_index, search_term=input_text)
                for res in response_content:
                    st.image(res, width=250)
with find_similar_images_tab:

    find_col_1, find_col_2 = st.columns(2)

    with find_col_1:
        uploaded_file = st.file_uploader("Select an image", type=['png', 'jpg'])

        if uploaded_file:
            uploaded_image_preview = uploaded_file.getvalue()
            st.image(uploaded_image_preview)

        find_button = st.button("Find", type="primary")  # display a primary button

    with find_col_2:
        if find_button:  # code in this if block will be run when the button is clicked
            st.subheader("Results")
            with st.spinner("Finding..."):  # show a spinner while the code in this with block runs
                response_content = get_similarity_search_results(index=st.session_state.vector_index, search_image=uploaded_file.getvalue())
                for res in response_content:
                    st.image(res, width=250)
