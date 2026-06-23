import kagglehub

# Download latest version
path = kagglehub.dataset_download("tolgadincer/labeled-chest-xray-images")

print("Path to dataset files:", path)

import os
import glob
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

contents = os.listdir(path)
print("Dataset contents:",contents)

chest_xray_path = os.path.join(path, 'chest_xray')
print(os.listdir(chest_xray_path))
train_dir = os.path.join(chest_xray_path, "train")
test_dir = os.path.join(chest_xray_path, "test")
print("Training categories:", os.listdir(train_dir))
print("Testing categories:", os.listdir(test_dir))

train_normal = glob.glob(train_dir + "/NORMAL/*.jpeg")
train_pneumonia = glob.glob(train_dir + "/PNEUMONIA/*.jpeg")

test_normal = glob.glob(test_dir + "/NORMAL/*.jpeg")
test_pneumonia = glob.glob(test_dir + "/PNEUMONIA/*.jpeg")

print(f"\nTraining set:")
print(f"Normal: {len(train_normal)} images")
print(f"Pneumonia: {len(train_pneumonia)} images")

print(f"\nTest set:")
print(f" Normal: {len(test_normal)} images")
print(f" Pneumonia: {len(test_pneumonia)}")

train_list = train_normal + train_pneumonia
train_labels = ["Normal"] * len(train_normal) + ["Pneumonia"] * len(train_pneumonia)

df_train = pd.DataFrame({
        'image': train_list,
        'class': train_labels
    })

test_list = test_normal + test_pneumonia
test_labels = ["Normal"] * len(test_normal) + ["Pneumonia"] * len(test_pneumonia)

df_test = pd.DataFrame({
	'image': test_list,
	'class': test_labels
})

train_df, val_df = train_test_split(
	df_train,
        test_size=0.2,
        random_state=7,
        stratify=df_train["class"]
)

print(f"Training set: {len(train_df)} images")
print(f"Validation set: {len(val_df)} images")

import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

plt.figure(figsize=(15,4))
plt.subplot(1,3,1)
sns.countplot(x="class", data=train_df, palette="mako")
plt.title("Training set", fontsize=12, fontweight="bold")

plt.subplot(1,3,2)
sns.countplot(x="class", data=val_df, palette="mako")
plt.title("Validation set", fontsize=12, fontweight="bold")

plt.subplot(1,3,3)
sns.countplot(x="class", data=val_df, palette="mako")
plt.title("Tests set", fontsize=12, fontweight="bold")

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(2, 6, figsize=(15,6))
normal_samples = train_df[train_df["class"] == "Normal"].sample(6, random_state=7)
pneumonia_samples = train_df[train_df["class"] == "Pneumonia"].sample(6, random_state=7)

for idx, (i, row) in enumerate(normal_samples.iterrows()):
       img = Image.open(row["image"])
       axes[0, idx].imshow(img, cmap="gray")
       axes[0, idx].set_title("Normal", fontsize=10)
       axes[0, idx].axis("off")

plt.suptitle("Sample Chest X-Rays: Normal(top) vs Pneumonia(bottom)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()

sample_imgs = train_df.sample(50, random_state = 42)
dimensions = []

for img_path in sample_imgs["image"]:
   img = Image.open(img_path)
   dimensions.append(img.size)

dimensions = np.array(dimensions)

print("\nImage Properties:")
print(f"Dimensions vary from {dimensions.min(axis=0)} to {dimensions.max(axis=0)}")
print(f"Image mode: {Image.open(train_df.iloc[0]["image"]).mode} (Grayscale)")
print(f"Total unique dimensions : {len(set(map(tuple, dimensions)))}")

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
IMG_SIZE = 224
BATCH_SIZE = 32
train_datagen = ImageDataGenerator(
    rescale=1./255,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1
)
val_datagen = ImageDataGenerator(
	rescale=1./255
)
train_generator = train_datagen.flow_from_dataframe(
  train_df,
  x_col="image",
  y_col="class",
  target_size=(IMG_SIZE, IMG_SIZE),
  batch_size=BATCH_SIZE,
  class_mode="binary",
  seed=7
)
val_generator = val_datagen.flow_from_dataframe(
  val_df,
  x_col="image",
  y_col="class",
  target_size=(IMG_SIZE, IMG_SIZE),
  batch_size=BATCH_SIZE,
  class_mode="binary",
  seed = 7
)

print("Validation generator created")

test_generator = val_datagen.flow_from_dataframe(
   df_test,
   x_col="image",
   y_col="class",
   target_size=(IMG_SIZE,IMG_SIZE),
   batch_size=1,
   class_mode="binary",
   shuffle=False,
   seed = 7
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

model = Sequential()
model.add(Input(shape=(IMG_SIZE, IMG_SIZE, 3)))
model.add(Conv2D(8,(3,3), activation="relu"))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.2))

model.add(Conv2D(16, (3,3), activation="relu"))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(64, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

print("Model created successfully")

model.summary()

model.compile(
   loss="binary_crossentropy",
   optimizer=Adam(learning_rate=3e-5),
   metrics=["binary_accuracy"]
)
print("Model compiled")

early_stopping = EarlyStopping(
   monitor="val_loss",
   patience=5,
   restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
   monitor="val_loss",
   factor=0.2,
   patience = 2,
   min_lr = 1e-7,
   verbose = 1
)

print("Starting training...")

history = model.fit(
   train_generator,
   epochs=50,
   validation_data = val_generator,
   callbacks = [early_stopping, reduce_lr],
   verbose = 1
)

print("\nTraining complete!")
baseline_model = model
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history.history["loss"], label="Training Loss")
axes[0].plot(history.history["val_loss"], label="Validation Loss")
axes[0].set_title("Model Loss Over Epochs", fontsize=14, fontweight="bold")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].grid(True)

axes[1].plot(history.history["binary_accuracy"], label="Training Accuracy")
axes[1].plot(history.history["val_binary_accuracy"], label="Validation Accuracy")
axes[1].set_title("Model Accuracy Over Epochs", fontsize=14, fontweight="bold")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()

val_loss,  val_accuracy = model.evaluate(val_generator, verbose=0)

print(f"\nValidation Results:")
print(f" Loss: {val_loss:.4f}")
print(f" Accuracy:{val_accuracy:.4f} ({val_accuracy*100:.2f})")

from tensorflow.keras.applications import ResNet152V2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input

base_model = ResNet152V2(
   weights="imagenet",
   include_top=False,
   input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

print(f"Loaded ResNet152V2 with {len(base_model.layers)} layers")

base_model.trainable = False

inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.3)(x)

outputs = Dense(1, activation="sigmoid")(x)
model = Model(inputs=inputs, outputs=outputs)

print("\nTransfer learning model created")

model.summary()
model.compile(
   loss="binary_crossentropy",
   optimizer=Adam(learning_rate=1e-4),
   metrics=["binary_accuracy"]
)

print("Model compiled and ready to train")

early_stopping = EarlyStopping(
   monitor='val_loss',
   patience=5,
   restore_best_weights=True
)

reduce_lor = ReduceLROnPlateau(
   monitor='val_loss',
   factor=0.2,
   patience=2,
   min_lor=1e-7,
   verbose=1
)

print("Starting transfer learning training...")

history = model.fit(
   train_generator,
   epochs=50,
   validation_data = val_generator,
   callbacks = [early_stopping, reduce_lr],
   verbose = 1
)

print("\nTraining complete!")

transfer_model = model

axes[0].plot(history.history["loss"], label="Training Loss")
axes[0].plot(history.history["val_loss"], label="Validation Loss")
axes[0].set_title("Model Loss Over Epochs", fontsize=14, fontweight="bold")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].grid(True)

axes[1].plot(history.history["binary_accuracy"], label="Training Accuracy")
axes[1].plot(history.history["val_binary_accuracy"], label="Validation Accuracy")
axes[1].set_title("Model Accuracy Over Epochs", fontsize=14, fontweight="bold")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].legend()
axes[1].grid(True)

val_loss,  val_accuracy = model.evaluate(val_generator, verbose=0)

print(f"\nTransfer Learning Results:")
print(f" Loss: {val_loss:.4f}")
print(f" Accuracy:{val_accuracy:.4f} ({val_accuracy*100:.2f})")

from sklearn.metrics import confusion_matrix, classification_report

baseline_loss, baseline_acc = baseline_model.evaluate(test_generator, verbose=0)
transfer_loss, transfer_acc = transfer_model.evaluate(test_generator, verbose=0)

print(f"Baseline CNN Test Accuracy: {baseline_acc*100:.2f}")
print(f"Transfer Learning Test Accuracy: {transfer_acc*100:.2f}")

test_generator.reset()
baseline_preds = (baseline_model.predict(test_generator, verbose=0) > 0.5)

test_generator.reset()
transfer_preds = (transfer_model.predict(test_generator, verbose=0) > 0.5)

test_generator.reset()
true_labels = test_generator.classes

baseline_cm = confusion_matrix(true_labels, baseline_preds)
transfer_cm = confusion_matrix(true_labels, transfer_preds)

fig, axes = plt.subplots(1, 2, figsize=(14,5))
sns.heatmap(
    baseline_cm, 
    annot=True, 
    fmt="d", 
    cmap="Blues", 
    xticklabels=["Normal", "Pneumonia"], 
    yticklabels=["Normal", "Pneumonia"],  
    ax=axes[0]
)
axes[0].set_title("Baseline CNN", fontsize=14, fontweight="bold")
axes[0].set_title("True Label")
axes[0].set_xlabel("Predicted Label")

sns.heatmap(transfer_cm, annot=True, fmt='d', cmap='Greens', xticklabels=['Normal','Pneumonia'], yticklabels=['Normal','Pneumonia'], ax=axes[1])
axes[1].set_title('Transfer Learning', fontsize=14, fontweight='bold')
axes[1].set_ylabel('True Label')
axes[1].set_xlabel('Predicted Label')

plt.tight_layout()
plt.show()
   
