import easyocr
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk



# Initialize the EasyOCR reader (with multiple languages if needed)
reader = easyocr.Reader(['en'])

# Function to handle image selection and text extraction
def extract_text():
    file_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
    
    if file_path:
        # Open the image
        image = Image.open(file_path)

        # Use EasyOCR to extract text from the image
        result = reader.readtext(file_path)
        extracted_text = "\n".join([text[1] for text in result])

        # Display extracted text in the GUI
        text_output.delete(1.0, tk.END)
        text_output.insert(tk.END, extracted_text)

        # Convert image to Tkinter format for displaying
        image_tk = ImageTk.PhotoImage(image)
        image_label.config(image=image_tk)
        image_label.image = image_tk

# Set up the main window
root = tk.Tk()
root.title("Handwriting Text Extractor")

# Create and pack widgets
upload_button = tk.Button(root, text="Upload Image", command=extract_text)
upload_button.pack(pady=20)

image_label = tk.Label(root)
image_label.pack()

text_output = tk.Text(root, wrap=tk.WORD, height=10, width=50)
text_output.pack(pady=10)

# Start the GUI loop
root.mainloop()