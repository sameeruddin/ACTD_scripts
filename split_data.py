import os
import shutil
from sklearn.model_selection import train_test_split

# Define the paths
dataset_dir = r"D:\PhD\Year - IV\Datasets\private\group_species_dataset"  # Replace with your dataset directory
output_dir = r"D:\PhD\Year - IV\Datasets\private\species_dataset"  # Replace with your desired output directory


# Create the output directories for each category
categories = [name for name in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, name))]

# Create train and test directories
train_dir = os.path.join(output_dir, 'train_data')
test_dir = os.path.join(output_dir, 'test_data')

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Create category directories in train and test directories
for category in categories:
    os.makedirs(os.path.join(train_dir, category), exist_ok=True)
    os.makedirs(os.path.join(test_dir, category), exist_ok=True)

# Function to copy files
def copy_files(file_list, source_dir, dest_dir):
    for file in file_list:
        # shutil.copy(os.path.join(source_dir, file), os.path.join(dest_dir, file))
        shutil.move(os.path.join(source_dir, file), os.path.join(dest_dir, file))

# Load images and labels
images = []
labels = []

for category in categories:
    category_dir = os.path.join(dataset_dir, category)
    for img_file in os.listdir(category_dir):
        if img_file.endswith(('.png', '.jpg', '.jpeg', '.JPG')):
            images.append(img_file)
            labels.append(category)

# Split the dataset
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, stratify=labels, test_size=0.25, random_state=42)

# Copy files to respective directories
for img, label in zip(train_images, train_labels):
    source_dir = os.path.join(dataset_dir, label)
    dest_dir = os.path.join(train_dir, label)
    copy_files([img], source_dir, dest_dir)

for img, label in zip(test_images, test_labels):
    source_dir = os.path.join(dataset_dir, label)
    dest_dir = os.path.join(test_dir, label)
    copy_files([img], source_dir, dest_dir)

print("Data successfully split and moved to train_data and test_data directories.")
