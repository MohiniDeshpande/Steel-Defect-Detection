import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
import random
from matplotlib.patches import Rectangle
from lxml import etree
import os

# Enable Matplotlib backend for displaying static images
%matplotlib inline

# Load image and annotation directories
image_path = glob.glob("images/*/*.jpg")
xmls_path = glob.glob("label/label/*.xml")

# Sorting images and annotations
xmls_path.sort(key=lambda x: x.split("/")[-1].split(".xml")[0])
image_path.sort(key=lambda x: x.split("/")[-1].split(".jpg")[0])

# Train image list and annotations
xmls_train = [path.split("/")[-1].split(".")[0] for path in xmls_path]
imgs_train = [img for img in image_path if img.split("/")[-1].split(".jpg")[0] in xmls_train]

# Extract label names as DataFrame column
labels = [label.split("/")[-2] for label in imgs_train]
labels = pd.DataFrame(labels, columns=["Defect Type"])

# One-hot encoding for multiple classes
from sklearn.preprocessing import LabelBinarizer

Class = labels["Defect Type"].unique()
Class_dict = dict(zip(Class, range(1, len(Class) + 1)))
labels["Class"] = labels["Defect Type"].apply(lambda x: Class_dict[x])

lb = LabelBinarizer()
lb.fit(list(Class_dict.values()))
transformed_labels = lb.transform(labels["Class"])

for i in range(transformed_labels.shape[1]):
    labels["Class" + str(i + 1)] = transformed_labels[:, i]

# Drop redundant columns
labels.drop(["Class", "Defect Type"], axis=1, inplace=True)

# Function to parse and extract information from annotation files
def to_labels(path):
    xml = open("{}".format(path)).read()
    sel = etree.HTML(xml)
    width = int(sel.xpath("//size/width/text()")[0])
    height = int(sel.xpath("//size/height/text()")[0])
    xmin = int(sel.xpath("//bndbox/xmin/text()")[0])
    xmax = int(sel.xpath("//bndbox/xmax/text()")[0])
    ymin = int(sel.xpath("//bndbox/ymin/text()")[0])
    ymax = int(sel.xpath("//bndbox/ymax/text()")[0])
    return [xmin/width, ymin/height, xmax/width, ymax/height]

# Extract bounding box coordinates
coors = [to_labels(path) for path in xmls_path]
xmin, ymin, xmax, ymax = list(zip(*coors))
xmin, ymin, xmax, ymax = map(np.array, [xmin, ymin, xmax, ymax])

# Labels dataset
label = np.array(labels.values)
labels_dataset = tf.data.Dataset.from_tensor_slices((xmin, ymin, xmax, ymax, label))

# Load image from image path
def load_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, 3)
    image = tf.image.resize(image, [224, 224])
    image = tf.cast(image, tf.float32) / 255
    return image

# Build the dataset
dataset = tf.data.Dataset.from_tensor_slices(imgs_train)
dataset = dataset.map(load_image)
dataset_label = tf.data.Dataset.zip((dataset, labels_dataset))

# Shuffle and batch the dataset
batch_size = 16  # Smaller batch size
dataset_label = dataset_label.repeat().shuffle(500).batch(batch_size)
dataset_label = dataset_label.prefetch(tf.data.experimental.AUTOTUNE)

# Train/test split
train_count = int(len(imgs_train) * 0.8)
test_count = int(len(imgs_train) * 0.2)
train_dataset = dataset_label.skip(test_count)
test_dataset = dataset_label.take(test_count)

class_dict = {v: k for k, v in Class_dict.items()}

# Load base model with pre-trained weights
base_resnet152v2 = tf.keras.applications.ResNet152V2(
    weights="imagenet", include_top=False, input_shape=(224, 224, 3), pooling='avg')

# Set layers as trainable
base_resnet152v2.trainable = False  # Start with frozen base

# Define the model
inputs = keras.Input(shape=(224, 224, 3))
x = base_resnet152v2(inputs)

x1 = keras.layers.Dense(1024, activation="relu")(x)
x1 = keras.layers.Dense(512, activation="relu")(x1)
out1 = keras.layers.Dense(1, name="xmin")(x1)
out2 = keras.layers.Dense(1, name="ymin")(x1)
out3 = keras.layers.Dense(1, name="xmax")(x1)
out4 = keras.layers.Dense(1, name="ymax")(x1)

x2 = keras.layers.Dense(1024, activation="relu")(x)
x2 = keras.layers.Dropout(0.4)(x2)  # Reduced dropout rate
x2 = keras.layers.Dense(512, activation="relu")(x2)
out_class = keras.layers.Dense(10, activation="softmax", name="class")(x2)

out = [out1, out2, out3, out4, out_class]

# Create the model
resnet152v2 = keras.models.Model(inputs=inputs, outputs=out)
resnet152v2.summary()

# Optimizer with learning rate schedule
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.0005, decay_steps=10000, decay_rate=0.9)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

# Compile the model with metrics for each output
resnet152v2.compile(
    optimizer=optimizer,
    loss={
        "xmin": "mse", 
        "ymin": "mse", 
        "xmax": "mse", 
        "ymax": "mse", 
        "class": "categorical_crossentropy"
    },
    metrics={
        "xmin": ["mae"], 
        "ymin": ["mae"], 
        "xmax": ["mae"], 
        "ymax": ["mae"], 
        "class": ["accuracy"]  # Ensure accuracy is included here
    }
)



# Train the model
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

history = resnet152v2.fit(
    train_dataset,
    steps_per_epoch=train_count // batch_size,
    epochs=20,  # Fewer epochs with early stopping
    validation_data=test_dataset,
    validation_steps=test_count // batch_size,
    callbacks=[early_stopping]
)

# After training the model
print(history.history.keys())
def plot_history(history):
    plt.figure(figsize=(12, 6))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    if 'class_accuracy' in history.history:  # Check if 'accuracy' exists
        plt.plot(history.history['class_accuracy'], label='Accuracy')
    if 'val_class_accuracy' in history.history:  # Check if 'val_accuracy' exists
        plt.plot(history.history['val_class_accuracy'], label='Val Accuracy')
    plt.legend()
    plt.title('Accuracy over epochs')

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.legend()
    plt.title('Loss over epochs')
    plt.show()

plot_history(history)
# Save the model
resnet152v2.save("resnet152v2.h5")
