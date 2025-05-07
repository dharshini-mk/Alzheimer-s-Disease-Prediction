# =====================================================================
# STEP 1: IMPORTING LIBRARIES
# =====================================================================

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import json
from PIL import Image

# =====================================================================
# STEP 2: DATA LOADING AND EXPLORATION
# =====================================================================

# Define paths for training and testing directories
train_dir = 'C:/Users/Admin/Downloads/Combined Dataset/train'
test_dir = 'C:/Users/Admin/Downloads/Combined Dataset/test'

# Count the number of images in each class for training and testing
def get_image_counts(directory):
    counts = {}
    for class_name in sorted(os.listdir(directory)):
        class_dir = os.path.join(directory, class_name)
        if os.path.isdir(class_dir):
            counts[class_name] = len(os.listdir(class_dir))
    return counts

# Recount to avoid stale variables
train_counts = get_image_counts(train_dir)
test_counts = get_image_counts(test_dir)

# Print counts
print("Training data distribution:")
for class_name, count in train_counts.items():
    print(f"{class_name}: {count} images")

print("\nTesting data distribution:")
for class_name, count in test_counts.items():
    print(f"{class_name}: {count} images")

# =====================================================================
# STEP 3: DATA PREPROCESSING AND AUGMENTATION
# =====================================================================

# Define image dimensions
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32

# Data augmentation for training data
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Only rescaling for validation and test data
test_datagen = ImageDataGenerator(rescale=1./255)

# Flow training images in batches using train_datagen
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

# Flow test images in batches using test_datagen
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Get class indices
class_indices = train_generator.class_indices
print("Class indices:", class_indices)

# Invert the dictionary to map indices to class names
class_names = {v: k for k, v in class_indices.items()}
print("Class names:", class_names)

# =====================================================================
# STEP 4: MODEL BUILDING USING RESNET-101 TRANSFER LEARNING
# =====================================================================

# Load the ResNet-101 model without the top classification layer
base_model = ResNet101(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))

# Add custom layers on top of ResNet-101
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(len(class_names), activation='softmax')(x)

# Create the final model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Show the model architecture
model.summary()

# =====================================================================
# STEP 5: MODEL TRAINING AND VALIDATION WITH PLOTS
# =====================================================================

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Callbacks
checkpoint = ModelCheckpoint(
    'alzheimer_resnet_model_best.keras',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    verbose=1,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=2,
    min_lr=1e-5,
    verbose=1
)

callbacks = [checkpoint, early_stopping, reduce_lr]

# Safer step calculation (for Colab quirks)
steps_per_epoch = max(train_generator.samples // BATCH_SIZE, 1)
validation_steps = max(test_generator.samples // BATCH_SIZE, 1)

# Model training
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=20,
    validation_data=test_generator,
    validation_steps=validation_steps,
    callbacks=callbacks,
    verbose=1
)

# Plotting accuracy & loss
plt.figure(figsize=(12, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.grid(True)

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.grid(True)

plt.tight_layout()
plt.show()

# =====================================================================
# STEP 6: FINE-TUNING THE MODEL (UNFREEZE SOME LAYERS)
# =====================================================================

# Unfreeze some layers of the base model for fine-tuning
for layer in model.layers:
    layer.trainable = True

for layer in model.layers[:-30]:
    layer.trainable = False

# Recompile the model with a lower learning rate
model.compile(
    optimizer=Adam(learning_rate=0.0001),  # Lower learning rate for fine-tuning
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Fine-tune the model
history_fine = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=20,  # Additional epochs for fine-tuning
    validation_data=test_generator,
    validation_steps=validation_steps,
    callbacks=callbacks
)

# Plot training & validation accuracy and loss for fine-tuning
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history_fine.history['accuracy'])
plt.plot(history_fine.history['val_accuracy'])
plt.title('Model Accuracy (Fine-tuning)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(history_fine.history['loss'])
plt.plot(history_fine.history['val_loss'])
plt.title('Model Loss (Fine-tuning)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.tight_layout()
plt.show()

# =====================================================================
# STEP 7: MODEL EVALUATION
# =====================================================================

# Load the best model (saved by ModelCheckpoint)
best_model = load_model('alzheimer_resnet_model_best.keras')

# Compile the model (this step is necessary to resolve the warning)
best_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Evaluate the model on the test data
test_loss, test_acc = best_model.evaluate(test_generator, steps=validation_steps)
print(f'Test accuracy: {test_acc:.4f}')
print(f'Test loss: {test_loss:.4f}')

# Get predictions
test_generator.reset()
predictions = best_model.predict(test_generator, steps=validation_steps)
predicted_classes = np.argmax(predictions, axis=1)

# Get true labels
true_classes = test_generator.classes[:len(predicted_classes)]

# =====================================================================
# STEP 8: CLASSIFICATION REPORT AND CONFUSION MATRIX
# =====================================================================

# Generate classification report
class_labels = list(class_names.values())
report = classification_report(true_classes, predicted_classes, target_names=class_labels)
print("Classification Report:")
print(report)

# Generate confusion matrix
cm = confusion_matrix(true_classes, predicted_classes)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.show()

# =====================================================================
# STEP 9: SAVING THE MODEL AND CLASS INDICES
# =====================================================================

# Save the model
best_model.save('alzheimer_resnet_model_final.keras')

# Save the class indices for later use
with open('class_indices.json', 'w') as f:
    json.dump(class_indices, f)

print("Model and class indices saved successfully!")

# =====================================================================
# STEP 10: 
# =====================================================================

# Function to preprocess the image
def preprocess_image(img):
    img = img.resize((IMG_HEIGHT, IMG_WIDTH))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Function to make predictions
def predict_alzheimer(img):
    # Preprocess the image
    img = preprocess_image(img)

    # Make prediction
    prediction = best_model.predict(img)[0]

    # Get the predicted class and confidence
    predicted_class_idx = np.argmax(prediction)
    confidence = prediction[predicted_class_idx]

    # Map the index to class name
    predicted_class = class_names[predicted_class_idx]

    # Create a dictionary of all class probabilities
    results = {class_names[i]: float(prediction[i]) for i in range(len(prediction))}

    return results

