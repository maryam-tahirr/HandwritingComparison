import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import easyocr
import random

# Constants
image_size = (128, 128)

# Load pre-trained models
similarity_model = load_model("siamese_model.h5")

# EasyOCR reader initialization
reader = easyocr.Reader(['en'])

# Predefined list of words for the game
word_list = ["python", "programming", "developer", "puzzle", "algorithm", "computer", "software", "variable"]

# Current word and its scrambled version
current_word = {"original": None, "scrambled": None}

def generate_scrambled_word():
    """Generate a new scrambled word."""
    original_word = random.choice(word_list)
    scrambled_word = ''.join(random.sample(original_word, len(original_word)))
    current_word["original"] = original_word
    current_word["scrambled"] = scrambled_word
    return scrambled_word

def predict_similarity(file1, file2):
    """Predict handwriting similarity."""
    img1 = img_to_array(load_img(file1, target_size=image_size)) / 255.0
    img2 = img_to_array(load_img(file2, target_size=image_size)) / 255.0

    prediction = similarity_model.predict([np.expand_dims(img1, axis=0), np.expand_dims(img2, axis=0)])[0][0]
    if prediction >= 0.5:
        return f"Similarity Score: {prediction:.2f}\nThese handwriting samples are similar."
    else:
        return f"Similarity Score: {prediction:.2f}\nThese handwriting samples are different."

def extract_text_from_image(file):
    """Extract text from an image."""
    result = reader.readtext(file)
    extracted_text = "\n".join([text[1] for text in result])
    return extracted_text

def check_guess(user_guess):
    """Check the user's guess for the scrambled word."""
    if user_guess.lower() == current_word["original"]:
        feedback = f"🎉 Correct! The word was '{current_word['original']}'. Here's a new word!"
        scrambled = generate_scrambled_word()
    else:
        feedback = f"❌ Incorrect. Try again! The scrambled word is still '{current_word['scrambled']}'."
        scrambled = current_word["scrambled"]
    return scrambled, feedback

# Streamlit App Layout
st.title("Text Analysis Tools")

# Tabs for different functionalities
tab = st.sidebar.selectbox(
    "Choose a tool",
    ["Handwriting Similarity", "Text Extraction", "Scramble Word Game"]
)

if tab == "Handwriting Similarity":
    st.header("Handwriting Similarity Checker")
    st.write("Upload two handwriting images to check their similarity.")

    file1 = st.file_uploader("Upload Handwriting Image 1", type=["png", "jpg", "jpeg"])
    file2 = st.file_uploader("Upload Handwriting Image 2", type=["png", "jpg", "jpeg"])

    if file1 and file2:
        result = predict_similarity(file1, file2)
        st.write(result)

elif tab == "Text Extraction":
    st.header("Text Extraction from Image")
    st.write("Upload an image to extract text using OCR.")

    file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])
    if file:
        extracted_text = extract_text_from_image(file)
        st.text_area("Extracted Text", extracted_text, height=200)

elif tab == "Scramble Word Game":
    st.header("Scramble Word Game")
    st.write("Guess the word from the scrambled letters!")

    if "scrambled_word" not in st.session_state:
        st.session_state["scrambled_word"] = generate_scrambled_word()

    scrambled_word = st.session_state["scrambled_word"]
    st.write(f"Scrambled Word: {scrambled_word}")

    user_guess = st.text_input("Your Guess")
    if st.button("Check"):
        scrambled_word, feedback = check_guess(user_guess)
        st.session_state["scrambled_word"] = scrambled_word
        st.write(feedback)
