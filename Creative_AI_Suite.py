import streamlit as st
import requests
from PIL import Image
import io
from huggingface_hub import InferenceClient
from gtts import gTTS
import os
from transformers import pipeline

# Hugging Face API keys and endpoints
api_key = st.secrets["huggingface"]["api_key"]  

# API URLs
blip_api_url = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-large"
flux_api_url = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-dev"
translation_api_url = "https://api-inference.huggingface.co/models/Helsinki-NLP/opus-mt-en-ar"
headers = {"Authorization": f"Bearer {api_key}"}

# Meta-Llama Chatbot
llama_model = "meta-llama/Meta-Llama-3-8B-Instruct"
client = InferenceClient(api_key=api_key)

# Function to generate an image using FLUX
def flux_generate_image(text_input):
    try:
        response = requests.post(flux_api_url, headers=headers, json={"inputs": text_input})
        response.raise_for_status()  # Raise an error for bad responses
        return response.content
    except Exception as e:
        st.error(f"Image generation error: {e}")
        return None

# Function to caption an image using BLIP
def blip_caption_image(image):
    try:
        response = requests.post(blip_api_url, headers=headers, data=image)
        response.raise_for_status()  # Raise an error for bad responses
        return response.json()
    except Exception as e:
        st.error(f"Captioning error: {e}")
        return None

# Function to handle chatbot conversation
def llama_chatbot(prompt):
    try:
        response = client.chat_completion(
            model=llama_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            stream=False
        )
        return response.choices[0].message['content']
    except Exception as e:
        st.error(f"Chatbot error: {e}")
        return "I'm sorry, I couldn't generate a response."

# Function to convert text to speech using gTTS
def text_to_speech(text):
    try:
        tts = gTTS(text, lang='en')
        tts_file = "story.mp3"
        tts.save(tts_file)
        return tts_file
    except Exception as e:
        st.error(f"Text-to-Speech error: {e}")
        return None

# Function to translate text from English to Arabic
# Function to translate large texts in chunks
def translate_to_arabic(text):
    try:
        translator = pipeline('translation_en_to_ar', model='Helsinki-NLP/opus-mt-en-ar')
        
        # Split the text into manageable chunks (typically by sentence or paragraph)
        max_length = 2000  # Define a reasonable token limit for each chunk
        chunks = [text[i:i + max_length] for i in range(0, len(text), max_length)]
        
        # Translate each chunk
        translated_chunks = [translator(chunk)[0]['translation_text'] for chunk in chunks]
        
        # Join the translated chunks back together
        translated_text = " ".join(translated_chunks)
        return translated_text
    except Exception as e:
        st.error(f"Translation error: {e}")
        return None


# Enhanced Streamlit UI
st.set_page_config(page_title="Creative AI Suite", page_icon="🎨", layout="wide")

st.title("🎨 Creative AI Suite")
st.markdown("Unlock the power of AI for **image generation**, **story creation**, and **speech conversion**.")

# Use session state to preserve the selected option
if 'option' not in st.session_state:
    st.session_state.option = None  # Initialize the option

# Row layout for options
st.header("Choose an option:")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    if st.button("Create an Image and Story"):
        st.session_state.option = "Create an Image and Story from Your Description"

with col2:
    if st.button("Interactive AI Chat"):
        st.session_state.option = "Interactive AI Chat"

with col3:
    if st.button("Convert Text to Speech"):
        st.session_state.option = "Convert Text to Speech"

with col4:
    if st.button("Generate an Image"):
        st.session_state.option = "Generate an Image"

# Add help tooltips to guide users
st.info("Select an option to interact with the AI tools.")

# Option handling
if st.session_state.option == "Create an Image and Story from Your Description":
    st.subheader("🖼️ Create an Image and Story")
    st.markdown("Describe the image you want, and we'll generate it for you along with a creative story.")
    description_input = st.text_input("🔎 Describe the image you want:")
    
    if st.button("Generate Image"):
        if description_input:
            with st.spinner("Generating image..."):
                generated_image = flux_generate_image(description_input)
                if generated_image:
                    image = Image.open(io.BytesIO(generated_image))
                    st.image(image, caption="Generated Image", use_column_width=True)

                    with st.spinner("✍️ Generating caption..."):
                        caption = blip_caption_image(generated_image)
                    
                    if caption and 'generated_text' in caption[0]:
                        title = caption[0]['generated_text']
                        st.write("### 🖋️ Caption (Title):", title)

                        with st.spinner("📖 Generating story..."):
                            story_script = llama_chatbot(f"Write a story based on the title: {title}.")
                            st.markdown("### 📜 Story Script:")
                            st.write(story_script)

                            # Translate the generated story to Arabic
                            with st.spinner("🌍 Translating story to Arabic..."):
                                arabic_translation = translate_to_arabic(story_script)
                            
                            if arabic_translation:
                                st.markdown("### 🌍 Arabic Translation:")
                                st.write(arabic_translation)

                        with st.spinner("🔊 Converting story to speech..."):
                            audio_file = text_to_speech(story_script)

                        if audio_file:
                            st.audio(audio_file, format="audio/mp3")
                            os.remove(audio_file)  # Cleanup after playing
                    else:
                        st.error("⚠️ Unable to generate caption for the image.")
        else:
            st.warning("Please provide a description to generate the image.")
