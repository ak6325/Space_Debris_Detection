# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming you already have loaded data1 and data2
# Example data loading (you will replace this with actual file loading)
data1 = pd.read_csv('preprocessed_space_data.csv')  # Replace with actual file path
data2 = pd.read_csv('space_decay.csv')  # Replace with actual file path

# Step 1: General EDA for data1 and data2

# Display basic information about data1 and data2
print("Data1 Info:")
print(data1.info())
print("\nData2 Info:")
print(data2.info())

# Display summary statistics for both datasets
print("\nData1 Summary Statistics:")
print(data1.describe())
print("\nData2 Summary Statistics:")
print(data2.describe())

# Step 2: Missing values analysis
print("\nData1 Missing Values:")
print(data1.isnull().sum())
print("\nData2 Missing Values:")
print(data2.isnull().sum())

# Step 3: Visualizing distributions for numerical columns in both datasets
# For simplicity, only select a few numerical columns here

# Select numerical columns for visualization
numerical_cols_data1 = data1.select_dtypes(include=['float64', 'int64']).columns
numerical_cols_data2 = data2.select_dtypes(include=['float64', 'int64']).columns

# Plot histograms for numerical columns in data1
plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_cols_data1[:9]):  # Limiting to the first 9 columns
    plt.subplot(3, 3, i + 1)
    sns.histplot(data1[col], kde=True, bins=20)
    plt.title(f"Distribution of {col}")
plt.tight_layout()
plt.show()

# Plot histograms for numerical columns in data2
plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_cols_data2[:9]):  # Limiting to the first 9 columns
    plt.subplot(3, 3, i + 1)
    sns.histplot(data2[col], kde=True, bins=20)
    plt.title(f"Distribution of {col}")
plt.tight_layout()
plt.show()

# Step 4: Correlation analysis
# Correlation matrix for data1
correlation_matrix_data1 = data1[numerical_cols_data1].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix_data1, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix for Data1')
plt.show()

# Correlation matrix for data2
correlation_matrix_data2 = data2[numerical_cols_data2].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix_data2, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix for Data2')
plt.show()

# Step 5: Exploring categorical data
# Show value counts for categorical columns in data1
categorical_cols_data1 = data1.select_dtypes(include=['object']).columns
for col in categorical_cols_data1:
    print(f"\nValue counts for {col} in Data1:")
    print(data1[col].value_counts())

# Show value counts for categorical columns in data2
categorical_cols_data2 = data2.select_dtypes(include=['object']).columns
for col in categorical_cols_data2:
    print(f"\nValue counts for {col} in Data2:")
    print(data2[col].value_counts())

# Step 6: Visualizing sample images (if applicable)

# If you have images associated with the data, you can plot a sample of images as follows
# (Here assuming you have a dataset that has image data)

# Function to plot a batch of images
def plot_images(data_batch):
    images, labels = next(data_batch)
    plt.figure(figsize=(10, 10))
    
    # Loop through only the first 9 images
    for i in range(9):  # Limiting to 9 images for a 3x3 grid
        plt.subplot(3, 3, i + 1)  # Corrected: using i+1 to avoid starting at 0
        plt.imshow(images[i])
        plt.title(f"Class: {labels[i].argmax()}")
        plt.axis('off')
    
    plt.show()

# Example usage: Plot sample images from training data
# Assuming you have a data generator or dataloader train_data
# plot_images(train_data)

# Optional: You can also add analysis for outliers, skewness, etc., if needed