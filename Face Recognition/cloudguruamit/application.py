import streamlit as st
import os
import boto3
import json
import base64
import io
from PIL import Image
from pathlib import Path
from langchain_core.prompts import PromptTemplate

# Set AWS credentials and region
os.environ["AWS_ACCESS_KEY_ID"] = "xxx"
os.environ["AWS_SECRET_ACCESS_KEY"] = "xxx"
os.environ["AWS_DEFAULT_REGION"] = "us-east-1"

# Instantiate the Bedrock and Rekognition clients

bedrock_client = boto3.client(
    service_name='bedrock-runtime',  
    region_name="us-east-1"
)

rek_client = boto3.client('rekognition', region_name="us-east-1")

def interactWithLLM(prompt, llm_type):
    try:
        if llm_type == 'titan':
            parameters = {
                "maxTokenCount": 512,
                "stopSequences": [],
                "temperature": 0,
                "topP": 0.9
            }
            body = json.dumps({"inputText": prompt, "textGenerationConfig": parameters})
            modelId = "amazon.titan-text-premier-v1:0"
            accept = "application/json"
            contentType = "application/json"

            response = bedrock_client.invoke_model(
                body=body, modelId=modelId, accept=accept, contentType=contentType
            )
            response_body = json.loads(response.get("body").read())
            response_text_titan = response_body.get("results")[0].get("outputText")
            return response_text_titan
    except Exception as e:
        st.error(f"Error during LLM interaction: {e}")
        return None

def image_base64_encoder(image_path):
    with Image.open(image_path) as open_image:
        image_bytes = io.BytesIO()
        open_image.save(image_bytes, format=open_image.format)
        image_bytes = image_bytes.getvalue()
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        file_type = f"image/{open_image.format.lower()}"
    return file_type, image_base64

def analyze_faces(image_data):
    try:
        response = rek_client.detect_faces(Image={'Bytes': image_data}, Attributes=['ALL'])
        face_details = response['FaceDetails']
        return face_details
    except Exception as e:
        st.error(f"Error during face analysis: {e}")
        return []

def compare_faces(source_data, target_data):
    try:
        response = rek_client.compare_faces(
            SourceImage={'Bytes': source_data},
            TargetImage={'Bytes': target_data},
            SimilarityThreshold=90
        )
        face_matches = response['FaceMatches']
        return face_matches
    except Exception as e:
        st.error(f"Error during face comparison: {e}")
        return []

def analyze_image(image_path, text) -> str:
    file_type, image_base64 = image_base64_encoder(image_path)

    system_prompt = """Identify and describe any faces present in this image. Provide detailed descriptions including apparent age, gender, emotional expressions, and any other notable features."""

    if not text:
        text = "Use the system prompt"

    prompt = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1000,
        "temperature": 0.5,
        "system": system_prompt,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": file_type,
                            "data": image_base64
                        }
                    },
                    {
                        "type": "text",
                        "text": text
                    }
                ]
            }
        ]
    }

    try:
        response = bedrock_client.invoke_model(
            body=json.dumps(prompt),
            modelId="anthropic.claude-3-sonnet-20240229-v1:0",
            accept="application/json",
            contentType="application/json"
        )

        response_body = response.get('body').read().decode('utf-8')
        st.write(f"Raw response: {response_body}")  # Log the raw response for debugging

        response_json = json.loads(response_body)
        llm_output = response_json['content'][0]['text']
        return llm_output

    except Exception as e:
        st.error(f"Error during model invocation: {e}")
        return {}

st.title(":rainbow[Face Analysis and Comparison]")

# Option selection
option = st.radio("Choose an analysis option:", ('Face Comparison using Rekognition', 'Face Description using Claude 3'))

if option == 'Face Comparison using Rekognition':
    st.header("Face Comparison using Rekognition")

    # Upload source and target images
    source_img_file = st.file_uploader("Upload Source Image", type=["jpg", "jpeg", "png"], key="source")
    target_img_file = st.file_uploader("Upload Target Image", type=["jpg", "jpeg", "png"], key="target")

    if source_img_file and target_img_file:
        # Read image data
        source_img_data = source_img_file.read()
        target_img_data = target_img_file.read()

        # Display images
        col1, col2 = st.columns(2)

        with col1:
            st.image(source_img_data, caption='Source Image', use_column_width=True)
        with col2:
            st.image(target_img_data, caption='Target Image', use_column_width=True)

        # Analyze faces in the source image
        face_details = analyze_faces(source_img_data)

        # Compare faces between source and target images
        face_matches = compare_faces(source_img_data, target_img_data)

        # Display face analysis and comparison results
        st.subheader("Face Analysis Results")
        st.write(f"Found {len(face_details)} faces in the source image.")
        for face in face_details:
            st.write(f"Face (Confidence: {face['Confidence']:.2f})")
            st.write(f" - Gender: {face['Gender']['Value']} (Confidence: {face['Gender']['Confidence']:.2f})")
            st.write(f" - Age Range: {face['AgeRange']['Low']} - {face['AgeRange']['High']}")
            st.write(f" - Emotions: {[emotion['Type'] for emotion in face['Emotions']]}")

        st.subheader("Face Comparison Results")
        st.write(f"Found {len(face_matches)} matching faces.")
        for match in face_matches:
            st.write(f"Face (Similarity: {match['Similarity']:.2f})")
            st.write(f" - Source: {match['Face']['BoundingBox']}")
            st.write(f" - Target: {match['Face']['BoundingBox']}")

        # Generate summary with LLM
        prompt_titan = """
        Human: Here are the details of the faces detected and compared between two images:
        Source Image Faces:
        {source_faces}
        Target Image Faces:
        {target_faces}
        Please provide a human-readable and understandable summary based on these details.
        Assistant:
        """

        source_faces_summary = json.dumps(face_details, indent=2)
        target_faces_summary = json.dumps(face_matches, indent=2)
        prompt_template_for_summary_generate = PromptTemplate.from_template(prompt_titan)
        prompt_data_for_summary_generate = prompt_template_for_summary_generate.format(
            source_faces=source_faces_summary,
            target_faces=target_faces_summary
        )

        llm_type = 'titan'  # Set your desired LLM type here
        response_text = interactWithLLM(prompt_data_for_summary_generate, llm_type)

        if response_text:
            st.subheader("Generated Summary")
            st.write(response_text)
        else:
            st.write("Failed to get a response from the LLM.")

elif option == 'Face Description using Claude 3':
    st.header("Face Description using Claude 3")

    with st.container():
        st.subheader('Image File Upload:')
        uploaded_file = st.file_uploader('Upload an Image', type=["png", "jpg", "jpeg"], key="new")
        json_spec = st.text_area("(optional) Insert your custom JSON spec to control image analysis")
        analyze_button = st.button("Analyze Image")

        if analyze_button:
            if uploaded_file is not None:
                st.image(uploaded_file)
                save_folder = "./images"
                save_path = Path(save_folder, uploaded_file.name)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                with open(save_path, mode='wb') as w:
                    w.write(uploaded_file.getvalue())
                if save_path.exists():
                    st.success(f'Image {uploaded_file.name} is successfully saved!')
                    result = analyze_image(save_path, json_spec)
                    st.write(result)
                    os.remove(save_path)
