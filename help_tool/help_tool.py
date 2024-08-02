"""Helper module for EDA notebook to perform 
data cleaning and preprocessing"""


from torchvision.transforms.functional import resize
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.data import DataLoader
import torchmetrics
import torch.nn.functional as F
import torch
import cv2
from pathlib import Path
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from itertools import combinations
from collections import defaultdict
from PIL import Image, ImageFile

# Allow loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True


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

            sns.boxplot(data=df, y=feature, hue=hue,
                        fill=False, legend=False, ax=axes[0])
            axes[0].set_title(
                f'{feature} ranges from {df[feature].min()} to {df[feature].max()}')
            axes[0].axhline(224, color='r', linestyle='--')

            sns.kdeplot(data=df, x=feature, hue=hue, ax=axes[1])
            axes[1].set_title(f'Distribution of {feature}')

            plt.tight_layout()
            plt.show()


def count_images_in_folders(base_dir):
    """Count number of images in each folder"""
    class_counts = defaultdict(int)

    for class_folder in os.listdir(base_dir):
        class_folder_path = os.path.join(base_dir, class_folder)

        if os.path.isdir(class_folder_path):
            num_images = len([f for f in os.listdir(class_folder_path) if os.path.isfile(
                os.path.join(class_folder_path, f))])
            class_counts[class_folder] = num_images

    return class_counts


def plot_image_distribution(base_dir):
    """ Count of class distribution for images in different folders """
    class_counts = defaultdict(int)

    for class_folder in os.scandir(base_dir):
        if class_folder.is_dir():
            class_folder_path = class_folder.path
            num_images = sum(1 for f in os.scandir(
                class_folder_path) if f.is_file())
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


def visualize_handfull_images(base_dir):
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

    class_folders = get_class_folders(base_dir)
    for folder in class_folders:
        class_name = os.path.basename(folder)
        for entry in os.scandir(folder):
            if entry.is_file():
                image_data = extract_image_attributes(entry.path, class_name)
                all_image_data.append(image_data)

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
            # Move data to the same device the model
            x, y = x.to(device), y.to(device)

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
    def get_predictions_and_labels(model, dataloader, device):
        model.eval()
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    predictions, true_labels = get_predictions_and_labels(
        model, loader, device)

    class_names = dataset.classes

    cm = confusion_matrix(true_labels, predictions)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()


def mean_inference_time(model):
    """Mean time for processing an image"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float).to(device)
    starter, ender = torch.cuda.Event(
        enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 300
    timings = np.zeros((repetitions, 1))

    # GPU-WARM-UP
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

    print(
        f"Mean Inference = {mean_syn:.2f} [ms], Standard deviation = {std_syn:.2f} [ms]")


def loss_accuracy_plots(event_log_path, batch_size, train_loader):

    def extract_scalars(event_accumulator, tag):
        events = event_accumulator.Scalars(tag)
        steps = [event.step for event in events]
        values = [event.value for event in events]
        return steps, values

    def aggregate_by_epoch(steps, values):

        num_samples = len(train_loader.dataset)
        steps_per_epoch = num_samples // batch_size

        epoch_values = {}
        for step, value in zip(steps, values):
            epoch = step // steps_per_epoch
            if epoch not in epoch_values:
                epoch_values[epoch] = []
            epoch_values[epoch].append(value)
        epoch_means = {epoch: sum(values)/len(values)
                       for epoch, values in epoch_values.items()}
        return list(epoch_means.keys()), list(epoch_means.values())

    event_accumulator = EventAccumulator(event_log_path)
    event_accumulator.Reload()

    loss_tags = ['train_loss', 'val_loss']
    accuracy_tags = ['train_accuracy', 'val_accuracy']

    loss_data = {tag: extract_scalars(
        event_accumulator, tag) for tag in loss_tags}
    accuracy_data = {tag: extract_scalars(
        event_accumulator, tag) for tag in accuracy_tags}

    aggregated_loss_data = {tag: aggregate_by_epoch(
        steps, values) for tag, (steps, values) in loss_data.items()}
    aggregated_accuracy_data = {tag: aggregate_by_epoch(
        steps, values) for tag, (steps, values) in accuracy_data.items()}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    for tag, (epochs, values) in aggregated_loss_data.items():
        ax1.plot(epochs, values, marker='o', label=tag)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()

    for tag, (epochs, values) in aggregated_accuracy_data.items():
        ax2.plot(epochs, values, marker='o', label=tag)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()

    fig.suptitle('Deep Neural Net Performance', fontsize=14)
    plt.show()


def calculate_histogram(image_path):
    """Calculate and return the normalized color histogram for an image."""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    hist_r = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([image], [1], None, [256], [0, 256])
    hist_b = cv2.calcHist([image], [2], None, [256], [0, 256])

    hist_r = cv2.normalize(hist_r, hist_r).flatten()
    hist_g = cv2.normalize(hist_g, hist_g).flatten()
    hist_b = cv2.normalize(hist_b, hist_b).flatten()

    return hist_r, hist_g, hist_b


def compare_histograms(hist1, hist2):
    score_r = cv2.compareHist(hist1[0], hist2[0], cv2.HISTCMP_CORREL)
    score_g = cv2.compareHist(hist1[1], hist2[1], cv2.HISTCMP_CORREL)
    score_b = cv2.compareHist(hist1[2], hist2[2], cv2.HISTCMP_CORREL)
    return (score_r + score_g + score_b) / 3  # Average score of RGB channels


def compare_images_in_directory(directory_path):
    image_files = [os.path.join(directory_path, f) for f in os.listdir(
        directory_path) if f.endswith(('jpg', 'jpeg', 'png'))]

    if len(image_files) < 2:
        print("Need at least two images to compare.")
        return image_files

    histograms = {image_file: calculate_histogram(
        image_file) for image_file in image_files}

    comparison_results = []

    for img1, img2 in combinations(image_files, 2):
        score = compare_histograms(histograms[img1], histograms[img2])
        comparison_results.append(
            (directory_path, os.path.basename(img1), os.path.basename(img2), score))

    comparison_results.sort(key=lambda x: x[3], reverse=True)
    df_results = pd.DataFrame(comparison_results, columns=[
                              'Location', 'Image1', 'Image2', 'SimilarityScore'])

    return df_results


def build_image_comparition_df(base_dir):
    df = pd.DataFrame()

    for class_folder in os.scandir(base_dir):
        if class_folder.is_dir():
            class_folder_path = class_folder.path
            comparison_results_df = compare_images_in_directory(
                class_folder_path)

            df = pd.concat([comparison_results_df, df], ignore_index=True)
    return df


def average_color(image_path):
    """Calculate the average color of an image."""
    with Image.open(image_path) as img:
        img = img.convert('RGB')
        np_img = np.array(img)
        average_color = np_img.mean(axis=(0, 1))
    return average_color


def average_color_distribution(base_dir):
    """Calculate average color distribution per class in a base directory."""
    color_distributions = {}

    for class_folder in os.scandir(base_dir):
        if class_folder.is_dir():
            class_name = class_folder.name
            colors = []

            for image_file in os.scandir(class_folder.path):
                if image_file.is_file() and image_file.name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    try:
                        avg_color = average_color(image_file.path)
                        colors.append(avg_color)
                    except Exception as e:
                        print(f"Error processing {image_file.path}: {e}")

            if colors:
                avg_class_color = np.mean(colors, axis=0)
                color_distributions[class_name] = avg_class_color

    return color_distributions


def display_image_pairs(df):

    for index, row in df.iterrows():
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        img1_path = os.path.join(row['Location'], row['Image1'])
        img2_path = os.path.join(row['Location'], row['Image2'])

        img1 = Image.open(img1_path)
        img2 = Image.open(img2_path)

        axes[0].imshow(img1)
        axes[0].axis('off')
        axes[0].set_title(f"{row['Image1']}, {row['SimilarityScore']:.3f}")

        axes[1].imshow(img2)
        axes[1].axis('off')
        axes[1].set_title(f"{row['Image2']}, {row['SimilarityScore']:.3f}")

        plt.show()


def visualize_sample_images(df):
    """Visualizes sample images from DataFrame in a grid of 6 columns."""

    num_images = len(df)

    num_rows = (num_images + 5) // 6

    plt.figure(figsize=(18, num_rows * 3))

    for i, (index, row) in enumerate(df.iterrows()):
        img_path = os.path.join(row['Location'], row['Image1'])

        if os.path.exists(img_path):
            img = Image.open(img_path)
            plt.subplot(num_rows, 6, i + 1)
            plt.imshow(img)
            plt.title(f"{row['Image1']}\n{row['SimilarityScore']:.3f}")
            plt.axis('off')
        else:
            print(f'Image not found: {img_path}')

    num_plots = num_rows * 6
    if num_images < num_plots:
        for j in range(num_images + 1, num_plots + 1):
            plt.subplot(num_rows, 6, j)
            plt.axis('off')

    plt.tight_layout()
    plt.show()


def display_images_grid(results_df, base_dir, grid_size=(4, 4), resize_dim=(150, 150)):
    """Display images in a grid and resize them to reduce memory usage."""

    num_images = len(results_df)
    num_rows, num_cols = grid_size
    total_images_to_show = num_rows * num_cols

    for start in range(0, num_images, total_images_to_show):
        end = min(start + total_images_to_show, num_images)
        subset_df = results_df.iloc[start:end]

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 15))
        axes = axes.flatten()

        for ax, (_, row) in zip(axes, subset_df.iterrows()):
            image_path = os.path.join(base_dir, row['Class'], row['ID'])

            if os.path.exists(image_path):
                with Image.open(image_path) as img:
                    # Resize to smaller dimensions
                    img = img.resize(resize_dim, Image.LANCZOS)
                    ax.imshow(img)
                    ax.axis('off')
                    ax.set_title(f"{row['Class']}/{row['ID']}", fontsize=8)
            else:
                ax.axis('off')
                ax.set_title(f"Image not found", fontsize=8)

        for ax in axes[len(subset_df):]:
            ax.axis('off')

        plt.tight_layout()
        plt.show()



def get_misclassified_and_display(model, data_loader, predicted_class, num_images=8):
    """Print out images with wrong classification."""

    def denormalize(img, mean, std):
        img = img.clone()
        for t, m, s in zip(img, mean, std):
            t.mul_(s).add_(m)
        return img

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    # Desired size for the displayed images (smaller resolution)
    target_size = (100, 100)

    misclassified_images = []
    true_labels = []
    preds = []

    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(
                device)  # Move data to the appropriate device
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            incorrect = (predicted != labels) & (predicted == predicted_class)

            for i in range(len(labels)):
                if incorrect[i]:
                    # Move image back to CPU for visualization
                    misclassified_images.append(inputs[i].cpu())
                    true_labels.append(labels[i].cpu().item())
                    preds.append(predicted[i].cpu().item())

                    if len(misclassified_images) >= num_images:
                        break
            if len(misclassified_images) >= num_images:
                break

    # Display images in a 2x4 grid
    num_rows = 2
    num_cols = 4
    # Adjust figsize to make the images smaller
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 4))
    for i in range(min(num_images, len(misclassified_images))):
        row = i // num_cols
        col = i % num_cols
        ax = axes[row, col]
        img = denormalize(misclassified_images[i], mean, std)

        img = resize(img, target_size)
        img = img.permute(1, 2, 0)  # CHW to HWC format
        img = torch.clamp(img, 0, 1)  # Ensure the values are in [0, 1]
        ax.imshow(img)
        ax.set_title(f'True: {true_labels[i]}\nPred: {preds[i]}')
        ax.axis('off')

    for i in range(num_images, num_rows * num_cols):
        fig.delaxes(axes.flatten()[i])
    plt.tight_layout()
    plt.show()
