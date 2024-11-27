
import streamlit as st
from PIL import Image
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.messages import HumanMessage
import pytesseract
from gtts import gTTS
import io
import base64

# Configure Google Gemini API Key
GOOGLE_API_KEY = "Gemini API Key"
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)

# Function to convert an image to Base64 format
def image_to_base64(image):
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode()

# Function to run OCR on an image
def run_ocr(image):
    return pytesseract.image_to_string(image).strip()

# Function to analyze the image using Gemini
def analyze_image(image, prompt):
    try:
        image_base64 = image_to_base64(image)
        message = HumanMessage(
            content=[ 
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": f"data:image/png;base64,{image_base64}"}
            ]
        )
        response = llm.invoke([message])
        return response.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"

# Function to convert text to speech (using a neutral or female voice)
def text_to_speech(text):
    tts = gTTS(text=text, lang='en', slow=False)  # Neutral voice by default
    audio_bytes = io.BytesIO()
    tts.write_to_fp(audio_bytes)
    audio_bytes.seek(0)
    return audio_bytes.getvalue()

# Main app function
def main():
    # Set Streamlit page configuration with futuristic look
    st.set_page_config(page_title="AI Assistive Tool", layout="wide", page_icon="🤖")

    # Title with a futuristic font and icon
    st.title('🔮 AI Assistive Tool for Visually Impaired 👁️')
    st.write(""" 
        This tool is designed to assist visually impaired individuals with image analysis. 
        Upload an image, select an option, and let AI help you with the analysis.
    """)

    # File uploader for images with large button text and futuristic styling
    st.sidebar.header("📂 Upload Image")
    uploaded_file = st.sidebar.file_uploader("Choose an image (jpg, jpeg, png)", type=['jpg', 'jpeg', 'png'])

    # Sidebar instructions with large font and colorful icons
    st.sidebar.header("🔧 Instructions")
    st.sidebar.write("""
    1. Upload an image.
    2. Choose an option below:
       - 🖼️ Describe Scene: Get a description of the image.
       - 📜 Extract Text: Extract text from the image.
       - 🚧 Detect Objects & Obstacles: Identify obstacles.
       - 🛠️ Personalized Assistance: Get task-specific help.
    3. Results will be read aloud for easy understanding.
    """)

    # Reset session state if a new image is uploaded
    if uploaded_file:
        if 'last_uploaded_file' in st.session_state and st.session_state.last_uploaded_file != uploaded_file:
            st.session_state.extracted_text = None
            st.session_state.summarized_text = None

        st.session_state.last_uploaded_file = uploaded_file
        image = Image.open(uploaded_file)

        # Apply CSS style to center the image and set its width to 500px (increase size)
        st.markdown("""
        <style>
            .centered-image {
                display: block;
                margin-left: auto;
                margin-right: auto;
                width: 500px;
            }
        </style>
        """, unsafe_allow_html=True)

        # Display the image using HTML and the centered-image class
        image_base64 = image_to_base64(image)  # Convert the image to base64
        st.markdown(f'<img src="data:image/png;base64,{image_base64}" class="centered-image" />', unsafe_allow_html=True)

        # Function to display buttons with dynamic colors
        def style_button(label, key, active_button_key):
            """Function to display buttons with dynamic color"""
            if 'active_button' not in st.session_state:
                st.session_state.active_button = None  # Initialize the session state for active button if not already present

            # Set color based on the active button
            color = "green" if st.session_state.get('active_button') == active_button_key else "red"

            # Apply button style with dynamic colors
            button_html = f"""
            <style>
                div[data-testid="stButton"] > button {{
                    background-color: {color} !important;
                    color: white !important;
                    border: none !important;
                    padding: 15px 30px !important;
                    cursor: pointer !important;
                    border-radius: 5px;
                    font-size: 20px !important;
                }}
            </style>
            """
            st.markdown(button_html, unsafe_allow_html=True)
            return st.button(label, key=key, help=f"Click to activate {label}", use_container_width=True)

        # Scene Description Button
        if style_button("🖼️ Describe Scene", key="scene_description", active_button_key="scene_description"):
            st.session_state.active_button = "scene_description"  # Set active button state
            with st.spinner("Generating scene description..."):
                scene_prompt = "Describe this image briefly."
                scene_description = analyze_image(image, scene_prompt)
                st.subheader("Scene Description")
                st.success(scene_description)
                st.audio(text_to_speech(scene_description), format='audio/mp3')

        # Text Extraction Button
        if style_button("📜 Extract Text", key="extract_text", active_button_key="extract_text"):
            st.session_state.active_button = "extract_text"  # Set active button state
            with st.spinner("Extracting text..."):
                extracted_text = run_ocr(image)
                if extracted_text:
                    st.session_state.extracted_text = extracted_text
                    st.subheader("Extracted Text")
                    st.info(extracted_text)
                    st.audio(text_to_speech(extracted_text), format='audio/mp3')
                else:
                    no_text_message = "No text detected in the image."
                    st.session_state.extracted_text = no_text_message
                    st.subheader("No Text Detected")
                    st.info(no_text_message)
                    st.audio(text_to_speech(no_text_message), format='audio/mp3')

        # Obstacle Detection Button
        if style_button("🚧 Detect Objects & Obstacles", key="detect_objects", active_button_key="detect_objects"):
            st.session_state.active_button = "detect_objects"  # Set active button state
            with st.spinner("Identifying objects and obstacles..."):
                obstacle_prompt = "Identify objects or obstacles in this image and provide their positions for safe navigation in brief."
                obstacle_description = analyze_image(image, obstacle_prompt)
                st.subheader("Objects & Obstacles Detected")
                st.success(obstacle_description)
                st.audio(text_to_speech(obstacle_description), format='audio/mp3')

        # Personalized Assistance Button
        if style_button("🛠️ Personalized Assistance", key="personalized_assistance", active_button_key="personalized_assistance"):
            st.session_state.active_button = "personalized_assistance"  # Set active button state
            with st.spinner("Providing personalized guidance..."):
                task_prompt = "Provide task-specific guidance based on the content of this image in brief. Include item recognition, label reading, and any relevant context."
                assistance_description = analyze_image(image, task_prompt)
                st.subheader("Personalized Assistance")
                st.success(assistance_description)
                st.audio(text_to_speech(assistance_description), format='audio/mp3')

# Running the Streamlit app
if __name__ == "__main__":
    main()
