import zipfile
import os
import shutil
import random
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

# Get the current directory (where the script is being run)
current_dir = os.getcwd()

# Path to the zip file (the file is expected to be in the same directory as the script)
zip_file_path = os.path.join(current_dir, 'kagglecatsanddogs_5340.zip')
extract_path = current_dir  # Extract to the current directory

# Step 1: Unzip the file
if not os.path.exists(os.path.join(current_dir, 'PetImages')):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print(f"Files extracted to {extract_path}")
else:
    print("Files already extracted.")

# Path to the PetImages folder
pet_images_dir = os.path.join(extract_path, 'PetImages')

# Step 2: Validate the images
# Temporary folder for validated images before splitting
validated_cats_dir = 'cats-v-dogs/validated/cats'
validated_dogs_dir = 'cats-v-dogs/validated/dogs'

# Invalid images lists
invalid_cats = []
invalid_dogs = []

# Create necessary directories
os.makedirs(validated_cats_dir, exist_ok=True)
os.makedirs(validated_dogs_dir, exist_ok=True)

def validate_images_and_copy(src_folder, valid_folder, invalid_images_list):
    """Validate and copy valid images to new folder, skipping corrupted or empty files."""
    valid_images = 0
    invalid_images = 0

    for img_name in os.listdir(src_folder):
        img_path = os.path.join(src_folder, img_name)

        try:
            # 1. Skip empty files (size = 0)
            if os.path.getsize(img_path) == 0:
                raise ValueError("File size is 0 bytes")

            # 2. Try loading and processing the image
            img = load_img(img_path, target_size=(150, 150))
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0  # Normalize

            # 3. If successful, copy to valid folder
            shutil.copy(img_path, valid_folder)
            valid_images += 1

        except Exception as e:
            # Catch corrupted/invalid files
            invalid_images_list.append(img_path)
            invalid_images += 1
            print(f"Skipping invalid image {img_path}: {e}")

    return valid_images, invalid_images

# Validate Cat and Dog images
cats_dir = os.path.join(pet_images_dir, 'Cat')
dogs_dir = os.path.join(pet_images_dir, 'Dog')

valid_cats_count, invalid_cats_count = validate_images_and_copy(cats_dir, validated_cats_dir, invalid_cats)
valid_dogs_count, invalid_dogs_count = validate_images_and_copy(dogs_dir, validated_dogs_dir, invalid_dogs)

# Print validation summary
print("\n===== Validation Summary =====")
print(f"Valid cat images:   {valid_cats_count}")
print(f"Invalid cat images: {invalid_cats_count}")
print(f"Valid dog images:   {valid_dogs_count}")
print(f"Invalid dog images: {invalid_dogs_count}")
print("================================")
print(f"TOTAL valid images:   {valid_cats_count + valid_dogs_count}")
print(f"TOTAL invalid images: {invalid_cats_count + invalid_dogs_count}\n")

# Step 3: Create directories for training and validation data
train_cats_dir = 'cats-v-dogs/train/cats'
train_dogs_dir = 'cats-v-dogs/train/dogs'
val_cats_dir = 'cats-v-dogs/validation/cats'
val_dogs_dir = 'cats-v-dogs/validation/dogs'

# Create necessary directories for train/validation
os.makedirs(train_cats_dir, exist_ok=True)
os.makedirs(train_dogs_dir, exist_ok=True)
os.makedirs(val_cats_dir, exist_ok=True)
os.makedirs(val_dogs_dir, exist_ok=True)

# Step 4: Split the data into training and validation sets
def split_data(source_folder, train_folder, valid_folder, split_ratio=0.8):
    """Split images into training and validation sets."""
    all_images = os.listdir(source_folder)
    random.shuffle(all_images)
    split_point = int(len(all_images) * split_ratio)
    
    # Copy to training folder
    for img in all_images[:split_point]:
        shutil.copy(os.path.join(source_folder, img), train_folder)
    
    # Copy to validation folder
    for img in all_images[split_point:]:
        shutil.copy(os.path.join(source_folder, img), valid_folder)

# Split cat images into training and validation
split_data(validated_cats_dir, train_cats_dir, val_cats_dir)

# Split dog images into training and validation
split_data(validated_dogs_dir, train_dogs_dir, val_dogs_dir)

print("Data split into training and validation sets successfully.")

# Step 5: Set up ImageDataGenerators for training and validation datasets
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,      # Normalize the image
    rotation_range=40,      # Randomly rotate images
    width_shift_range=0.2,  # Randomly shift width
    height_shift_range=0.2, # Randomly shift height
    shear_range=0.2,        # Randomly shear images
    zoom_range=0.2,         # Randomly zoom images
    horizontal_flip=True,   # Randomly flip images
    fill_mode='nearest'     # Fill missing pixels after transformations
)

validation_datagen = ImageDataGenerator(rescale=1.0/255.0)  # Just normalize the validation images

# Load training and validation data
train_generator = train_datagen.flow_from_directory(
    'cats-v-dogs/train', 
    target_size=(150, 150), 
    batch_size=32, 
    class_mode='binary'
)

validation_generator = validation_datagen.flow_from_directory(
    'cats-v-dogs/validation', 
    target_size=(150, 150), 
    batch_size=32, 
    class_mode='binary'
)

print("Generators created successfully and ready for training!")

# Optional: Start training (example of how to use model.fit)
# model.fit(train_generator, epochs=10, validation_data=validation_generator)
