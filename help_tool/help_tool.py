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


import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torchvision import models
from torch.utils.data import DataLoader
from torchsummary import summary
from sklearn.metrics import confusion_matrix
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torchvision.models import ResNet18_Weights
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


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
    return df



def get_model_metrics(model, dataset, batch_size):

    classes = dataset.classes
    num_classes = len(classes)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    metrics = {
        'accuracy': torchmetrics.Accuracy(task='multiclass', num_classes=num_classes).to(device),
        'precision': torchmetrics.Precision(task='multiclass', num_classes=num_classes, average='macro').to(device),
        'recall': torchmetrics.Recall(task='multiclass', num_classes=num_classes, average='macro').to(device),
        'f1': torchmetrics.F1Score(task='multiclass', num_classes=num_classes, average='macro').to(device),
        'roc': torchmetrics.ROC(task='multiclass', num_classes=num_classes, average='macro').to(device)
    }


    model.to(device)
    model.eval()


    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for batch in test_loader:
            x, y = batch
            x, y = x.to(device), y.to(device)  # Move data to the same device the model
            
            y_hat = model(x)

            # Update metrics
            metrics['accuracy'].update(y_hat, y)
            metrics['precision'].update(y_hat, y)
            metrics['recall'].update(y_hat, y)
            metrics['f1'].update(y_hat, y)




    accuracy = metrics['accuracy'].compute()
    precision = metrics['precision'].compute()
    recall = metrics['recall'].compute()
    f1 = metrics['f1'].compute()

    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')




def confusion_matrix_plot(model, dataset, loader):
    # Function to get model predictions and true labels
    def get_predictions_and_labels(model, dataloader, device):
        model.eval()  # Set model to evaluation mode
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        
        return np.array(predictions), np.array(true_labels)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Assuming val_loader is your DataLoader for the validation set
    predictions, true_labels = get_predictions_and_labels(model, loader, device)

    # Get class names (assuming class_names is a list of your class labels)
    class_names = dataset.classes

    # Compute confusion matrix
    cm = confusion_matrix(true_labels, predictions)

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()



def mean_inference_time(model):

   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   model.to(device)
   dummy_input = torch.randn(1, 3,224,224,dtype=torch.float).to(device)
   starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
   repetitions = 300
   timings=np.zeros((repetitions,1))

   #GPU-WARM-UP
   for _ in range(10):
      _ = model(dummy_input)

   # MEASURE PERFORMANCE
   with torch.no_grad():
      for rep in range(repetitions):
         starter.record()
         _ = model(dummy_input)
         ender.record()

         # WAIT FOR GPU SYNC
         torch.cuda.synchronize()
         curr_time = starter.elapsed_time(ender)
         timings[rep] = curr_time

   mean_syn = np.sum(timings) / repetitions
   std_syn = np.std(timings)

   print(f"Mean Inference = {mean_syn:.2f} [ms], Standard deviation = {std_syn:.2f} [ms]")



def loss_accuracy_plots(event_log_path, batch_size, train_loader):
    # Function to extract scalar data
    def extract_scalars(event_accumulator, tag):
        events = event_accumulator.Scalars(tag)
        steps = [event.step for event in events]
        values = [event.value for event in events]
        return steps, values

    # Function to aggregate scalar data by epoch
    def aggregate_by_epoch(steps, values):

        num_samples = len(train_loader.dataset)
        steps_per_epoch = num_samples // batch_size

        epoch_values = {}
        for step, value in zip(steps, values):
            epoch = step // steps_per_epoch
            if epoch not in epoch_values:
                epoch_values[epoch] = []
            epoch_values[epoch].append(value)
        epoch_means = {epoch: sum(values)/len(values) for epoch, values in epoch_values.items()}
        return list(epoch_means.keys()), list(epoch_means.values())
    
    event_accumulator = EventAccumulator(event_log_path)
    event_accumulator.Reload()

    # Define tags for metrics
    loss_tags = ['train_loss', 'val_loss']
    accuracy_tags = ['train_accuracy', 'val_accuracy']

    # Extract data from TensorBoard logs
    loss_data = {tag: extract_scalars(event_accumulator, tag) for tag in loss_tags}
    accuracy_data = {tag: extract_scalars(event_accumulator, tag) for tag in accuracy_tags}

    # Aggregate data by epoch
    aggregated_loss_data = {tag: aggregate_by_epoch(steps, values) for tag, (steps, values) in loss_data.items()}
    aggregated_accuracy_data = {tag: aggregate_by_epoch(steps, values) for tag, (steps, values) in accuracy_data.items()}

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot Loss
    for tag, (epochs, values) in aggregated_loss_data.items():
        ax1.plot(epochs, values, marker='o', label=tag)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()

    # Plot Accuracy
    for tag, (epochs, values) in aggregated_accuracy_data.items():
        ax2.plot(epochs, values, marker='o', label=tag)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()


    fig.suptitle('Deep Neural Net Performance', fontsize=14)
    plt.show()


