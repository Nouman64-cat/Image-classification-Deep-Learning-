
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt





# Step 4: Load and Preprocess Data
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()



# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0



# Step 5: Define Model Architecture
model = models.Sequential()

#Layer 1

#convolutional layer for extracting certain features
model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 3))) #64 filters , filter size 3x3

#Layer 2

#max pooling layer reduces spatial dimensions
model.add(layers.MaxPooling2D((2, 2)))


#another convolutional layer
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
#another 
model.add(layers.MaxPooling2D((2, 2)))

#another
model.add(layers.Conv2D(512, (3, 3), activation='relu'))



#Layer 3

#flattening layer transforming 2D data into 1D array
model.add(layers.Flatten())

#Layer 4

#fully connected layer
model.add(layers.Dense(256, activation='relu'))

#another fully connected layer
model.add(layers.Dense(128, activation='relu'))

#fully connected layer
model.add(layers.Dense(64, activation='relu'))


#Output layer


model.add(layers.Dense(10, activation='softmax'))#10 represents number of classes



# Step 6: Compile the Model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])



early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
history = model.fit(
    train_images, 
    train_labels, 
    epochs=60,
    validation_data=(test_images, test_labels),
    callbacks=[early_stopping]
)



# Step 8: Evaluate Model Performance
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.grid()
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')



test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print(test_acc)

import seaborn as sns
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing import image



import numpy as np

# Predict the values from the test dataset
test_predictions = model.predict(test_images)
test_predictions_classes = np.argmax(test_predictions, axis=1)



test_labels_classes = test_labels.squeeze()




# Compute the confusion matrix
cm = confusion_matrix(test_labels_classes, test_predictions_classes)



# Plot the confusion matrix using Seaborn
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d')
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
img_path = 'plane1.jpeg'

img = image.load_img(img_path)  # Load the original image
plt.imshow(img)  # Display the original image
plt.show()

# Continue with preprocessing after displaying
img = img.resize((32, 32))  # Resize for the model


# Convert the image to a numpy array and normalize it
img_array = image.img_to_array(img) / 255.0

# Add an extra dimension to the array (for batch size)
img_array = np.expand_dims(img_array, axis=0)


result = ["Airplane", "Automobile", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]


predictions = model.predict_on_batch(img_array)


predicted_class = np.argmax(predictions[0])
print("Predicted class:", predicted_class)
print("Image is ", result[predicted_class])

