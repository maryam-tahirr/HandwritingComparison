import os
import numpy as np
import tf_keras as keras
from tf_keras.models import load_model
from tf_keras.preprocessing.image import load_img, img_to_array
import gradio as gr
import easyocr
import random

# Constants
image_size = (128, 128)

# Load pre-trained models
similarity_model = load_model("siamese_model.h5")

# EasyOCR reader initialization
reader = easyocr.Reader(['en'])

# Function 1: Predict handwriting similarity
def predict_similarity(file1, file2):
    img1 = img_to_array(load_img(file1, target_size=image_size)) / 255.0
    img2 = img_to_array(load_img(file2, target_size=image_size)) / 255.0

    prediction = similarity_model.predict([np.expand_dims(img1, axis=0), np.expand_dims(img2, axis=0)])[0][0]
    if prediction >= 0.5:
        return f"Similarity Score: {prediction:.2f}\nThese handwriting samples are similar."
    else:
        return f"Similarity Score: {prediction:.2f}\nThese handwriting samples are different."

# Function 2: Extract text from an image
def extract_text_from_image(file):
    result = reader.readtext(file)
    extracted_text = "\n".join([text[1] for text in result])
    return extracted_text

# Function 3: Scramble Word Game
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

def check_guess(user_guess):
    """Check the user's guess for the scrambled word."""
    if user_guess.lower() == current_word["original"]:
        feedback = f"🎉 Correct! The word was '{current_word['original']}'. Here's a new word!"
        scrambled = generate_scrambled_word()
    else:
        feedback = f"❌ Incorrect. Try again! The scrambled word is still '{current_word['scrambled']}'."
        scrambled = current_word["scrambled"]
    return scrambled, feedback

# Initialize the first scrambled word
generate_scrambled_word()

# Gradio Interface with Tabs
iface = gr.TabbedInterface(
    interface_list=[
        gr.Interface(
            fn = None,
            title = None,
            description = None,
            inputs = None,
            outputs=gr.Image(value="Word.png", label=" "),
        ),
        gr.Interface(
            fn=predict_similarity,
            inputs=[
                gr.Image(type="filepath", label="Upload Handwriting Image 1"),
                gr.Image(type="filepath", label="Upload Handwriting Image 2")
            ],
            outputs="text",
            title="Handwriting Similarity Checker",
            description="Upload two handwriting images to check their similarity."
        ),
        gr.Interface(
            fn=extract_text_from_image,
            inputs=gr.Image(type="filepath", label="Upload Image for Text Extraction"),
            outputs="text",
            title="Text Extraction from Image",
            description="Upload an image to extract text using OCR."
        ),
        gr.Interface(
            fn=check_guess,
            inputs=gr.Textbox(placeholder="Enter your guess here...", label="Your Guess"),
            outputs=[
                gr.Textbox(value=current_word["scrambled"], label="Scrambled Word", interactive=False),
                gr.Textbox(label="Feedback", interactive=False)
            ],
            title="Scramble Word Game",
            description="Guess the word from the scrambled letters!"
        )
    ],
    tab_names=["Home","Handwriting Similarity", "Text Extraction", "Word Puzzle Game"],
    title="Text Analysis Tools"
)

if __name__ == "__main__":
    iface.launch()
