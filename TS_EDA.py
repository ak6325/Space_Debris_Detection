# Importing modules 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os

# File paths
train_csv_path = 'debris-detection/train.csv'  # Replace with the actual path
val_csv_path = 'debris-detection/val.csv'      # Replace with the actual path
train_image_folder = 'debris-detection/train'  # Replace with the actual path to the train images folder
val_image_folder = 'debris-detection/val'      # Replace with the actual path to the val images folder

# Dataset loading 
data1 = pd.read_csv(train_csv_path)
data2 = pd.read_csv(val_csv_path)

# Step 1: General EDA for datasets
# Displaying basic information from datasets
print("Training Data Info:")
print(data1.info())
print("\nValidation Data Info:")
print(data2.info())

# Displaying statistics for both datasets
print("\nTraining Data Summary Statistics:")
print(data1.describe())
print("\nValidation Data Summary Statistics:")
print(data2.describe())

# Step 2: Analyzing the missing values
print("\nTraining Data Missing Values:")
print(data1.isnull().sum())
print("\nValidation Data Missing Values:")
print(data2.isnull().sum())

# Step 3: Visualizing distributions for numerical columns in both datasets
# Selecting the numerical columns for visualization
numerical_cols_data1 = data1.select_dtypes(include=['float64', 'int64']).columns
numerical_cols_data2 = data2.select_dtypes(include=['float64', 'int64']).columns

# Plotting histograms for numerical columns in training data
plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_cols_data1[:9]):  # Limiting to the first 9 columns
    plt.subplot(3, 3, i + 1)
    sns.histplot(data1[col], kde=True, bins=20)
    plt.title(f"Distribution of {col}")
plt.tight_layout()
plt.show()

# Plotting histograms for numerical columns in validation data
plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_cols_data2[:9]):  # Limiting to the first 9 columns
    plt.subplot(3, 3, i + 1)
    sns.histplot(data2[col], kde=True, bins=20)
    plt.title(f"Distribution of {col}")
plt.tight_layout()
plt.show()

# Step 4: Correlation analysis
# Correlation matrix for training data
correlation_matrix_data1 = data1[numerical_cols_data1].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix_data1, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix for Training Data')
plt.show()

# Correlation matrix for validation data
correlation_matrix_data2 = data2[numerical_cols_data2].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix_data2, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix for Validation Data')
plt.show()

# Step 5: Exploring categorical data
# Showing value counts for categorical columns in training data
categorical_cols_data1 = data1.select_dtypes(include=['object']).columns
for col in categorical_cols_data1:
    print(f"\nValue counts for {col} in Training Data:")
    print(data1[col].value_counts())

# Showing value counts for categorical columns in validation data
categorical_cols_data2 = data2.select_dtypes(include=['object']).columns
for col in categorical_cols_data2:
    print(f"\nValue counts for {col} in Validation Data:")
    print(data2[col].value_counts())

# Step 6: Visualizing sample images from the training set
# Function to plot a batch of images
def plot_images(dataframe, base_path, n=9):
    """
    Plots a batch of images.
    :param dataframe: DataFrame with 'image_path' and 'label' columns.
    :param base_path: Path to the folder containing images.
    :param n: Number of images to display (default is 9).
    """
    plt.figure(figsize=(10, 10))
    
    # Select a sample of images to display
    sample = dataframe.sample(n=n)
    
    for i, (index, row) in enumerate(sample.iterrows()):
        img_path = os.path.join(base_path, row['image_path'])  # Assuming 'image_path' column exists
        try:
            img = Image.open(img_path)
            plt.subplot(3, 3, i + 1)
            plt.imshow(img)
            plt.title(f"Label: {row['label']}")  # Adjust if 'label' column exists
            plt.axis('off')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
    
    plt.tight_layout()
    plt.show()

# Call the function to display images
plot_images(data1, train_image_folder, n=9)
