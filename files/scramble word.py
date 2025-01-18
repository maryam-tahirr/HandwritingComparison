import random
import gradio as gr

# Predefined list of words
word_list = ["python", "programming", "developer", "puzzle", "algorithm", "computer", "software", "variable"]

# Current scrambled word and its original
current_word = {"original": None, "scrambled": None}

def generate_scrambled_word():
    """Generate a scrambled word from the list and store it."""
    original_word = random.choice(word_list)
    scrambled_word = ''.join(random.sample(original_word, len(original_word)))
    current_word["original"] = original_word
    current_word["scrambled"] = scrambled_word
    return scrambled_word

def check_guess(user_guess):
    """Check the user's guess and provide feedback."""
    if user_guess.lower() == current_word["original"]:
        feedback = f"🎉 Correct! The word was '{current_word['original']}'."
        scrambled = generate_scrambled_word()  # Generate a new word
    else:
        feedback = f"❌ Incorrect. The correct word was '{current_word['original']}'. Try this one!"
        scrambled = current_word["scrambled"]  # Keep the same word
    return feedback, scrambled

# Generate the first scrambled word
generate_scrambled_word()

# Gradio Interface
with gr.Blocks() as word_puzzle_game:
    gr.Markdown("## 🎲 Word Puzzle Game")
    gr.Markdown("I’ll give you a scrambled word. Try to guess the original word!")

    # Display scrambled word
    scrambled_word_display = gr.Textbox(value=current_word["scrambled"], label="Scrambled Word", interactive=False)

    # User input for the guess
    user_guess = gr.Textbox(placeholder="Enter your guess here...", label="Your Guess")

    # Output feedback
    feedback_display = gr.Textbox(label="Feedback", interactive=False)

    # Button to submit the guess
    submit_button = gr.Button("Submit Guess")
    
    # Button to generate a new scrambled word
    new_word_button = gr.Button("Give Me a New Word")

    # Define button actions
    submit_button.click(
        fn=check_guess, 
        inputs=[user_guess], 
        outputs=[feedback_display, scrambled_word_display]
    )
    new_word_button.click(
        fn=generate_scrambled_word, 
        inputs=[], 
        outputs=[scrambled_word_display]
    )

# Launch the Gradio app
word_puzzle_game.launch()
