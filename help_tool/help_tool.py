"""Helper module for EDA notebook to perform 
data cleaning and preprocessing"""


from scipy.stats import chi2_contingency
import os
from typing import Optional, Any


from collections import defaultdict
from PIL import Image, ImageOps, ImageFile
from tensorflow.keras.preprocessing.image import img_to_array, load_img, save_img

# Allow loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from sklearn.metrics import (accuracy_score, auc, confusion_matrix, roc_curve)
from sklearn.model_selection import KFold
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
pd.plotting.register_matplotlib_converters()



def distribution_check(df: pd.DataFrame, hue: str) -> None:
    """Box plot graph for identifying numeric column outliers, normality of distribution."""
    df = df.reset_index(drop=True)

    for feature in df.columns:

        if df[feature].dtype.name in ['object', 'bool']:
            pass

        else:

            fig, axes = plt.subplots(1, 2, figsize=(12, 3))

            print(f'{feature}')

            # Outlier check (Box plot)
            sns.boxplot(data=df, y=feature, hue=hue, fill=False, legend=False, ax=axes[0])
            axes[0].set_title(
                f'{feature} ranges from {df[feature].min()} to {df[feature].max()}')
            axes[0].axhline(224, color='r', linestyle='--')

            # Distribution check (Histogram).
            sns.kdeplot(data=df, x=feature, hue=hue,  #kde=True, 
                         #bins=20, 
                         ax=axes[1])
            axes[1].set_title(f'Distribution of {feature}')

            plt.tight_layout()
            plt.show()



def count_images_in_folders(base_dir):
    # Create a dictionary to store the count of images for each class
    class_counts = defaultdict(int)
    
    # Iterate over each subdirectory (class folder)
    for class_folder in os.listdir(base_dir):
        class_folder_path = os.path.join(base_dir, class_folder)
        
        if os.path.isdir(class_folder_path):  # Ensure it's a directory
            # Count the number of files in the class directory
            num_images = len([f for f in os.listdir(class_folder_path) if os.path.isfile(os.path.join(class_folder_path, f))])
            class_counts[class_folder] = num_images
    
    return class_counts


def plot_image_distribution(base_dir):
    """ Count of class distribution for images in different folders """
    class_counts = defaultdict(int)
    
    for class_folder in os.scandir(base_dir):
        if class_folder.is_dir():  
            class_folder_path = class_folder.path
            num_images = sum(1 for f in os.scandir(class_folder_path) if f.is_file())
            class_counts[class_folder.name] = num_images
    
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    
    plt.figure(figsize=(10, 6))
    plt.bar(classes, counts, color='skyblue')
    plt.xlabel('Mushroom Classes')
    plt.ylabel('Number of Images')
    plt.title('Class Distribution of Mushroom Images')
    plt.xticks(rotation=45)
    plt.show()


def get_class_folders(base_dir):
    """Return a list of class folders in the base directory."""
    return [entry.path for entry in os.scandir(base_dir) if entry.is_dir()]

def load_sample_image(folder_path):
    """Load a sample image from the folder."""
    for entry in os.scandir(folder_path):
        if entry.is_file():
            return Image.open(entry.path)
    return None

def visualize_sample_images(base_dir):
    class_folders = get_class_folders(base_dir)
    
    plt.figure(figsize=(12, 8))
    
    for i, folder in enumerate(class_folders):
        sample_image = load_sample_image(folder)
        if sample_image:
            plt.subplot(3, 3, i + 1)
            plt.imshow(sample_image)
            plt.title(os.path.basename(folder))
            plt.axis('off')
        else:
            print(f'No images found in {folder}')
    
    plt.tight_layout()
    plt.show()


def extract_and_combine_image_attributes(base_dir):
    """Extract image attributes from all folders and return a DataFrame."""
    all_image_data = []

    # Helper function to extract image attributes
    def extract_image_attributes(image_path, class_name):
        with Image.open(image_path) as img:
            width, height = img.size
            file_size = os.path.getsize(image_path)
            file_format = img.format
            mode = img.mode

            return {
                'ID': os.path.basename(image_path),
                'Class': class_name,
                'Width': width,
                'Height': height,
                'File Size': file_size,
                'File Format': file_format,
                'Mode': mode
            }

    # Process each class folder
    class_folders = get_class_folders(base_dir)
    for folder in class_folders:
        class_name = os.path.basename(folder)
        for entry in os.scandir(folder):
            if entry.is_file():
                image_data = extract_image_attributes(entry.path, class_name)
                all_image_data.append(image_data)

    # Create a DataFrame from the collected data
    df = pd.DataFrame(all_image_data)
    df['ID_number'] = df['ID'].str.split('_', expand=True)[0].astype(int)
    
    return df


# def extract_and_combine_image_attributes(base_dir):
#     """Extract image attributes from all folders and return a DataFrame."""
#     all_image_data = []

#     # Helper function to extract image attributes
#     def extract_image_attributes(image_path, class_name):
#         with Image.open(image_path) as img:
                
#             width, height = img.size
#             file_size = os.path.getsize(image_path)
#             file_format = img.format
#             mode = img.mode

#             # Convert image to NumPy array
#             img_array = np.array(img)
            
#             # Calculate average color
#             avg_color = img_array.mean(axis=(0, 1))
#             std_color = img_array.std(axis=(0, 1))
            
#             return {
#                 'ID': os.path.basename(image_path),
#                 'Class': class_name,
#                 'Width': width,
#                 'Height': height,
#                 'File Size': file_size,
#                 'File Format': file_format,
#                 'Mode': mode,
#                 'Avg R': avg_color[0],
#                 'Avg G': avg_color[1],
#                 'Avg B': avg_color[2],
#                 'Std R': std_color[0],
#                 'Std G': std_color[1],
#                 'Std B': std_color[2]
#             }

#     # Process each class folder
#     class_folders = get_class_folders(base_dir)
#     for folder in class_folders:
#         class_name = os.path.basename(folder)
#         for entry in os.scandir(folder):
#             if entry.is_file():
#                 image_data = extract_image_attributes(entry.path, class_name)
#                 all_image_data.append(image_data)

#     # Create a DataFrame from the collected data
#     df = pd.DataFrame(all_image_data)
    
#     return df

