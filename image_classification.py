# Step 1: Set Up Google Colab - (Run this code in a Google Colab notebook)

# Step 2: Install TensorFlow (usually pre-installed in Colab)
# !pip install tensorflow

# Step 3: Import Required Libraries
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt





# Step 4: Load and Preprocess Data
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()





train_images

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0



train_images

# Step 5: Define Model Architecture
model = models.Sequential()
model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(256, (3, 3), activation='relu'))



model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))



# Step 6: Compile the Model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])



# Step 7: Train the Model
history = model.fit(train_images, train_labels, epochs=20,
                    validation_data=(test_images, test_labels))



# Step 8: Evaluate Model Performance
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')



test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print(test_acc)

import seaborn as sns
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing import image




# Predict the values from the test dataset
test_predictions = model.predict(test_images)
test_predictions_classes = np.argmax(test_predictions, axis=1)



# Convert test labels to class indices if they are in one-hot encoded format
test_labels_classes = np.argmax(test_labels, axis=1) if test_labels.ndim > 1 else test_labels



# Compute the confusion matrix
cm = confusion_matrix(test_labels_classes, test_predictions_classes)



# Plot the confusion matrix using Seaborn
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Step 9: Model Improvement and Fine-Tuning
# Based on the performance, you may decide to adjust the model architecture or hyperparameters

# Step 10: Save and Deploy the Model
model.save('my_cifar10_model.h5')

from tensorflow.keras.preprocessing import image
import numpy as np

from google.colab import files
uploaded = files.upload()



# Assuming you uploaded a single file named 'my_image.jpg'
img_path = 'cat (2).jpg'

img = image.load_img(img_path, target_size=(32, 32))  # Make sure to resize it to the size your model expects

# Convert the image to a numpy array and normalize it
img_array = image.img_to_array(img) / 255.0

# Add an extra dimension to the array (for batch size)
img_array = np.expand_dims(img_array, axis=0)


img

img_array

result = ["Airplane", "Automobile", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]


predictions = model.predict(img_array)

predicted_class = np.argmax(predictions[0])
print("Predicted class:", predicted_class)
print("Image is ", result[predicted_class])

