# import os
# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from PIL import Image
# import numpy as np

# # Directory where your original images are stored, organized by font types
# input_dir = 'C:/Users/csman/Desktop/Desktop/7th sem/niqo/FONT_Recognition/images/'  
# output_dir = 'C:/Users/csman/Desktop/Desktop/7th sem/niqo/FONT_Recognition/augmented_images/'

# # Parameters for data augmentation
# data_gen = ImageDataGenerator(
#     rotation_range=10,
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     shear_range=0.1,
#     zoom_range=0.05,
#     fill_mode='nearest'
# )

# # Creating output directories if they don't exist
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)

# # Loop through each font folder and perform augmentation
# font_types = os.listdir(input_dir)  # Assumes each folder inside 'original_images' is a font type

# for font in font_types:
#     font_input_path = os.path.join(input_dir, font)
#     font_output_path = os.path.join(output_dir, font)

#     if not os.path.exists(font_output_path):
#         os.makedirs(font_output_path)

#     # Loop over all images for the current font
#     for img_name in os.listdir(font_input_path):
#         img_path = os.path.join(font_input_path, img_name)
        
#         # Load image
#         img = Image.open(img_path)
#         img = img.resize((64, 64))  # Resize to the desired shape for the model
#         img_array = np.array(img)
#         img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        
#         # Generate augmented images
#         i = 0
#         for batch in data_gen.flow(img_array, batch_size=1, save_to_dir=font_output_path, save_prefix=font, save_format='jpeg'):
#             i += 1
#             if i > 9:  # Augment each image 10 times
#                 break

# print("Data augmentation complete and images saved.")
import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFilter
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Function to apply image sharpening, contrast enhancement, and denoising
def preprocess_image(image_path):
    # 1. Open image
    image = Image.open(image_path)

    # 2. Sharpen the image
    sharp_image = image.filter(ImageFilter.SHARPEN)

    # 3. Enhance the contrast
    enhancer = ImageEnhance.Contrast(sharp_image)
    contrast_image = enhancer.enhance(2)  # Adjust contrast level

    # 4. Convert to numpy array for denoising
    contrast_image_np = np.array(contrast_image)

    # 5. Apply denoising
    denoised_image = cv2.fastNlMeansDenoisingColored(contrast_image_np, None, 10, 10, 7, 21)

    # 6. Convert back to PIL Image
    preprocessed_image = Image.fromarray(denoised_image)
    
    return preprocessed_image

# Directory paths
input_dir = 'C:/Users/csman/Desktop/Desktop/7th sem/niqo/FONT_Recognition/images/'
output_dir = 'C:/Users/csman/Desktop/Desktop/7th sem/niqo/FONT_Recognition/augmented_images2/'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Apply augmentation using ImageDataGenerator
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    brightness_range=[0.8, 1.2],
    horizontal_flip=False,
    fill_mode='nearest'
)

# Loop through the dataset, preprocess and augment images
for font_class in os.listdir(input_dir):
    font_class_path = os.path.join(input_dir, font_class)
    
    # Create output directory for each font class
    output_class_dir = os.path.join(output_dir, font_class)
    if not os.path.exists(output_class_dir):
        os.makedirs(output_class_dir)

    for img_file in os.listdir(font_class_path):
        img_path = os.path.join(font_class_path, img_file)
        
        # Preprocess the image
        preprocessed_image = preprocess_image(img_path)
        
        # Convert the preprocessed image to numpy array for augmentation
        img_array = np.array(preprocessed_image)
        img_array = img_array.reshape((1,) + img_array.shape)  # Reshape to add batch dimension
        
        # Generate and save augmented images
        i = 0
        for batch in datagen.flow(img_array, batch_size=1, save_to_dir=output_class_dir, save_prefix='aug', save_format='png'):
            i += 1
            if i > 5:  # Generate 5 augmented images per original image
                break

print("Augmentation and preprocessing completed!")
