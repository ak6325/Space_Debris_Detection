# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, LSTM, Dense, TimeDistributed
from tensorflow.keras.layers import Dropout, BatchNormalization, Reshape

# Assuming you already have loaded data1 and data2
# Example data loading (you will replace this with actual file loading)
data1 = pd.read_csv('preprocessed_space_data.csv')  # Replace with actual file path
data2 = pd.read_csv('space_decay.csv')  # Replace with actual file path

# Assuming image data is loaded separately as `image_sequences`
# image_sequences would be a 4D array: (num_sequences, time_steps, height, width, channels)
# For example, image_sequences.shape could be (1000, 10, 64, 64, 3)

# Step 1: Prepare data (if necessary, perform your previous EDA steps)

# Here, we assume image_sequences contains spatial and temporal data for each sequence
# and labels contains the labels for each sequence (1 for debris, 0 for no debris)

# Load your image sequence and label data
# For this example, create random dummy data (Replace this with actual data loading)
num_sequences = 100  # Number of sequences (e.g., different objects over time)
time_steps = 10  # Number of time steps in each sequence
height, width, channels = 64, 64, 3  # Dimensions of each image frame

# Create dummy data for demonstration (use real data in practice)
image_sequences = np.random.rand(num_sequences, time_steps, height, width, channels).astype(np.float32)
labels = np.random.randint(2, size=num_sequences)  # Binary labels (1 = debris, 0 = no debris)

# Step 2: Build CNN-LSTM Model

model = Sequential()

# Step 2.1: CNN for spatial feature extraction (apply Conv2D on each frame in the sequence)
model.add(TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'), input_shape=(time_steps, height, width, channels)))
model.add(TimeDistributed(MaxPooling2D((2, 2))))
model.add(TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same')))
model.add(TimeDistributed(MaxPooling2D((2, 2))))
model.add(TimeDistributed(Conv2D(128, (3, 3), activation='relu', padding='same')))
model.add(TimeDistributed(MaxPooling2D((2, 2))))
model.add(TimeDistributed(Flatten()))  # Flatten output for each frame

# Step 2.2: LSTM for temporal dependencies
model.add(LSTM(128, activation='relu', return_sequences=False))
model.add(Dropout(0.5))

# Step 2.3: Output layer for classification
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))  # Sigmoid for binary classification

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Model summary
model.summary()

# Step 3: Train the Model
# Split data into train and test sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(image_sequences, labels, test_size=0.2, random_state=42)

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=16, validation_data=(X_test, y_test))

# Step 4: Evaluate the Model
# Evaluate on test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# Step 5: Visualize Training Performance
# Plot accuracy and loss curves
plt.figure(figsize=(12, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
