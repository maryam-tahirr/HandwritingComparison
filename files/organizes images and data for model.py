import pandas as pd
import random
import os
import shutil

# Function to organize and merge session data
def organize_combined_handwriting_data(csv_file, session_folders, output_csv, output_image_folder):
    # Read the metadata CSV file
    df = pd.read_csv(csv_file)
    
    # Initialize dictionary to store image names by writer
    writer_images = {}

    # Process each session folder
    for session_id, session_folder in enumerate(session_folders, start=1):
        session_name = f"s{session_id:02d}"  # Generate session name (e.g., s01, s02)
        
        for index, row in df.iterrows():
            writer_id = f"w{int(row['wid']):04d}"  # Ensure writer ID is zero-padded to 4 digits
            prompts = ["pLND", "pPHR", "pWOZ"]
            replications = [1, 2, 3]  # Assume 3 replications per prompt
            
            # Collect images for this writer in the current session
            for prompt in prompts:
                for replication in replications:
                    img_name = f"{writer_id}_{session_name}_{prompt}_r{replication:02d}.png"
                    img_path = os.path.join(session_folder, img_name)
                    
                    # Check if the image exists
                    if os.path.exists(img_path):
                        # Add image to the writer's data
                        if writer_id in writer_images:
                            writer_images[writer_id].append(img_name)
                        else:
                            writer_images[writer_id] = [img_name]

    # Create output folder if it doesn't exist
    os.makedirs(output_image_folder, exist_ok=True)
    
    # Generate pairs
    pairs = []

    # Positive pairs (same writer)
    for writer_id, images in writer_images.items():
        for i in range(len(images)):
            for j in range(i + 1, len(images)):
                pairs.append((images[i], images[j], 1))  # Label 1 for same writer

    # Negative pairs (different writers)
    writer_ids = list(writer_images.keys())
    for _ in range(len(pairs)):  # Adjust number of negative pairs as needed
        writer1, writer2 = random.sample(writer_ids, 2)
        img1 = random.choice(writer_images[writer1])
        img2 = random.choice(writer_images[writer2])
        pairs.append((img1, img2, 0))  # Label 0 for different writers

    # Shuffle pairs
    random.shuffle(pairs)

    # Save pairs to CSV
    pair_data = pd.DataFrame(pairs, columns=['Image1', 'Image2', 'Label'])
    pair_data.to_csv(output_csv, index=False)

    # Copy images to the output folder
    for pair in pairs:
        for img in [pair[0], pair[1]]:
            session_folder = session_folders[0] if "s01" in img else session_folders[1]
            src_path = os.path.join(session_folder, img)
            dest_path = os.path.join(output_image_folder, img)
            if not os.path.exists(dest_path):  # Avoid duplicate copies
                shutil.copy(src_path, dest_path)

    print(f"Data organized. Pairs saved to {output_csv}. Images copied to {output_image_folder}.")

# Define paths
csv_file = "D:\Machine Learning\session1\session1\surveydata.csv"  # Update with your CSV path
session_folders = [
    "session1",  # Path to session 1 images
    "session2"   # Path to session 2 images
]
output_csv = "combined_output_pairs.csv"  # Output CSV for image pairs
output_image_folder = "2combined_output_images"  # Output folder for selected images

# Call the function
organize_combined_handwriting_data(csv_file, session_folders, output_csv, output_image_folder)
