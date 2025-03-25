import cv2
import numpy as np
import os
from skimage.feature import hog, local_binary_pattern
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
import keras_tuner as kt
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Paths to your datasets
MASKED_IMAGES_DIR = 'part1/with_mask'
UNMASKED_IMAGES_DIR = 'part1/without_mask'

def extract_features(image):
    # Extract HOG features
    hog_features = hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), 
                     block_norm='L2-Hys', transform_sqrt=True, feature_vector=True)
    
    return hog_features


def extract_sift_descriptors(image):
    # Extract SIFT features
    sift = cv2.SIFT_create()
    _, descriptors = sift.detectAndCompute(image, None)
    
    if descriptors is not None:
        # Flatten descriptors to fixed size using mean
        features = descriptors.mean(axis=0)
    else:
        # Return zeros if no descriptors are found (128-dimensional for SIFT)
        features = np.zeros(128)
    
    return features


# Load dataset
def load_image_data(directory, label):
    feature_list = []
    label_list = []
    images = []
    
    for filename in os.listdir(directory):
        if filename.endswith(('jpg', 'png', 'jpeg')):
            image = cv2.imread(os.path.join(directory, filename))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, (128, 128))
            images.append(image)
            feature_list.append(extract_features(image))
            label_list.append(label)
    
    return feature_list, label_list, images


def build_model(hp):
    activation = hp.Choice("activation", ["relu", "leaky_relu"])  # Try different activation functions
    dropout_rate = hp.Float("dropout_rate", 0.2, 0.5, step=0.1)  # Vary dropout between 0.2 and 0.5
    learning_rate = hp.Choice("learning_rate", [1e-2, 1e-3, 1e-4])  # Try different learning rates

    model = keras.Sequential([
        layers.Input(shape=(128, 128, 1)),

        # First Conv block
        layers.Conv2D(hp.Int("filters_1", 32, 64, step=16), (3, 3), activation=activation),
        layers.BatchNormalization(),
        layers.Conv2D(hp.Int("filters_2", 32, 64, step=16), (3, 3), activation=activation, padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(dropout_rate),

        # Second Conv block
        layers.Conv2D(hp.Int("filters_3", 64, 128, step=32), (3, 3), activation=activation, padding='same'),
        layers.Conv2D(hp.Int("filters_4", 64, 128, step=32), (3, 3), activation=activation, padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(dropout_rate),

        layers.Flatten(),
        layers.Dense(hp.Int("dense_units", 128, 512, step=128), activation=activation),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),
        layers.Dense(2, activation='softmax')
    ])

    # Compile model with tunable learning rate
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model


# Load data
masked_features, masked_labels, masked_images = load_image_data(MASKED_IMAGES_DIR, 1)
unmasked_features, unmasked_labels, unmasked_images = load_image_data(UNMASKED_IMAGES_DIR, 0)


# Combine and split data
X = np.array(masked_features + unmasked_features)
y = np.array(masked_labels + unmasked_labels)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Normalize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Train SVM classifier
svm_classifier = SVC(kernel='rbf', C=2, gamma="scale")
svm_classifier.fit(X_train, y_train)
svm_predictions = svm_classifier.predict(X_test)

# Evaluate SVM
print("SVM Accuracy:", accuracy_score(y_test, svm_predictions))
print("SVM F1-Score:", f1_score(y_test, svm_predictions))


# Train Logistic Regression classifier
logistic_classifier = LogisticRegression(max_iter=500, solver="liblinear", C=0.5, random_state=42)
logistic_classifier.fit(X_train, y_train)
logistic_predictions = logistic_classifier.predict(X_test)

# Evaluate Logistic Regression
print("Logistic Regression Accuracy:", accuracy_score(y_test, logistic_predictions))
print("Logistic Regression F1-Score:", f1_score(y_test, logistic_predictions))



# Train Neural Network classifier
neural_network = MLPClassifier(hidden_layer_sizes=(256, 128, 64), max_iter=500, activation='relu', solver='adam', random_state=42)
neural_network.fit(X_train, y_train)
nn_predictions = neural_network.predict(X_test)

# Evaluate Neural Network
print("Neural Network Accuracy:", accuracy_score(y_test, nn_predictions))
print("Neural Network F1-Score:", f1_score(y_test, nn_predictions))


X_cnn = np.array(masked_images + unmasked_images)
y_cnn = np.array(masked_labels + unmasked_labels)
y_cnn = to_categorical(y_cnn, num_classes=2)
# X_cnn = X_cnn / 255.0  # Normalize pixel values
X_cnn = np.expand_dims(X_cnn, axis=-1)  # Add channel dimension
X_train_cnn, X_test_cnn, y_train_cnn, y_test_cnn = train_test_split(X_cnn, y_cnn, test_size=0.2, random_state=42)

tuner = kt.RandomSearch(build_model,
                        objective='val_accuracy',
                        max_trials=20,  # Set max trials
                        executions_per_trial=2,  # Run each config twice
                        directory='hyperparam_tuning',
                        project_name='cnn_tuning')


# Run the search for best hyperparameters
tuner.search(X_train_cnn, y_train_cnn, epochs=10, validation_data=(X_test_cnn, y_test_cnn))

# Get the best model and evaluate it
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
best_model = tuner.hypermodel.build(best_hps)


best_model.fit(X_train_cnn, y_train_cnn, epochs=10, validation_data=(X_test_cnn, y_test_cnn))
cnn_loss, cnn_accuracy = best_model.evaluate(X_test_cnn, y_test_cnn)
print(f"Best Model Accuracy: {cnn_accuracy}")


history = best_model.fit(X_train_cnn, y_train_cnn, epochs=10, validation_data=(X_test_cnn, y_test_cnn))


for param in best_hps.values:
    print(f"{param}: {best_hps.get(param)}")


y_pred = best_model.predict(X_test_cnn)
y_pred_classes = np.argmax(y_pred, axis=1)  # Convert one-hot to class labels
y_true = np.argmax(y_test_cnn, axis=1)  # Convert one-hot to class labels

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)

# Plot confusion matrix
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()