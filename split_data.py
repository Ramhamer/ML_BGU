import os
import shutil
import random
from math import floor

def split_dataset(src_dir, dest_dirs, percentages):
    # Validate input percentages
    if sum(percentages) != 100:
        raise ValueError("Percentages must sum to 100")
    
    # Get all image filenames in the source directory
    images = [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))]
    random.shuffle(images)  # Shuffle to ensure randomness
    
    # Calculate the number of images for each split
    total_images = len(images)
    split_counts = [floor(total_images * (p / 100)) for p in percentages]
    
    # Adjust the count for the last directory to include remaining images due to flooring
    split_counts[-1] = total_images - sum(split_counts[:-1])
    
    # Create destination directories if they don't exist
    for dest_dir in dest_dirs:
        os.makedirs(dest_dir, exist_ok=True)
    
    # Split and move images
    start = 0
    for count, dest_dir in zip(split_counts, dest_dirs):
        end = start + count
        for img in images[start:end]:
            shutil.move(os.path.join(src_dir, img), os.path.join(dest_dir, img))
        start = end

    print(f"Dataset split complete: {split_counts}")

# Example usage
src_directory = "/home/oury/Documents/Ram/ct_project/data/Non-COVID-19/photos"
destination_directories = ["/home/oury/Documents/Ram/ct_project/data/train/NONCOVID19", "/home/oury/Documents/Ram/ct_project/data/validation/NONCOVID19", "/home/oury/Documents/Ram/ct_project/data/test/NONCOVID19"]
split_percentages = [90, 5, 5]  # Example: 90%, 5%, 5%

split_dataset(src_directory, destination_directories, split_percentages)
