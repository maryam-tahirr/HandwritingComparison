import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tf_keras 
from tensorflow import keras
from tf_keras.models import load_model
from tf_keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# Constants
data_path = "combined_output_pairs.csv"
image_folder = "2combined_output_images"
image_size = (128, 128)
batch_size = 32

# Load the dataset
data = pd.read_csv(data_path)

# Split into train and test sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Function to preprocess images
def preprocess_image(image_path):
    image = load_img(image_path, target_size=image_size)
    return img_to_array(image) / 255.0

# Function to prepare data for evaluation
def prepare_data(dataframe, image_folder):
    images1 = []
    images2 = []
    labels = []
    
    for _, row in dataframe.iterrows():
        img1_path = os.path.join(image_folder, row['Image1'])
        img2_path = os.path.join(image_folder, row['Image2'])
        
        if os.path.exists(img1_path) and os.path.exists(img2_path):
            images1.append(preprocess_image(img1_path))
            images2.append(preprocess_image(img2_path))
            labels.append(row['Label'])
    
    return [np.array(images1), np.array(images2)], np.array(labels)

# Prepare test data
test_images, test_labels = prepare_data(test_data, image_folder)

# Load the trained Siamese model
model = load_model("siamese_model.h5")

# Evaluate the model
predictions = (model.predict(test_images) > 0.5).astype("int32")
accuracy = accuracy_score(test_labels, predictions)

print(f"Accuracy Score: {accuracy:.2f}")

# Confusion Matrix and Classification Report
cm = confusion_matrix(test_labels, predictions)
report = classification_report(test_labels, predictions, target_names=["Not Similar", "Similar"])

print("\nConfusion Matrix:")
print(cm)
print("\nClassification Report:")
print(report)

# Visualizations
# 1. Confusion Matrix Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Not Similar", "Similar"], yticklabels=["Not Similar", "Similar"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("confusion_matrix.png")
plt.show()

# 2. Sample predictions visualization
fig, axes = plt.subplots(5, 2, figsize=(12, 20))
axes = axes.ravel()

for i in range(10):
    index = np.random.randint(0, len(test_data))
    img1_path = os.path.join(image_folder, test_data.iloc[index]['Image1'])
    img2_path = os.path.join(image_folder, test_data.iloc[index]['Image2'])
    img1 = preprocess_image(img1_path)
    img2 = preprocess_image(img2_path)
    label = test_data.iloc[index]['Label']
    prediction = "Similar" if predictions[index] == 1 else "Not Similar"

    axes[i].imshow(np.hstack((img1, img2)))
    axes[i].axis('off')
    axes[i].set_title(f"Actual: {'Similar' if label == 1 else 'Not Similar'} | Predicted: {prediction}")

plt.tight_layout()
plt.savefig("sample_predictions.png")
plt.show()

# 3. Distribution of Predictions
plt.figure(figsize=(8, 6))
sns.countplot(predictions.flatten(), palette="Set2")
plt.title("Distribution of Predictions")
plt.xlabel("Prediction (0=Not Similar, 1=Similar)")
plt.ylabel("Count")
plt.savefig("predictions_distribution.png")
plt.show()

# 4. Distribution of Test Labels
plt.figure(figsize=(8, 6))
sns.countplot(test_labels.flatten(), palette="Set1")
plt.title("Distribution of Actual Labels")
plt.xlabel("Label (0=Not Similar, 1=Similar)")
plt.ylabel("Count")
plt.savefig("labels_distribution.png")
plt.show()
