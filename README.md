
# PROJECT-1

**Part-1**

The goal is to classify facial images as "with mask" or "without mask" using two approaches:

- **ML with Handcrafted Features** – Extract features, train ML classifiers (e.g., SVM, Neural Network), and compare their accuracy.
- **CNN-based Classification** – Design, train, and optimize a CNN for the same task, testing different hyperparameters.

**Part-2**




## Documentation

**Part-1**

*Image Preprocessing*
- Converting to grayscale to reduce computational complexity and focus on structural features
- Resizing to (128, 128) pixels for consistency.
- Features are normalized using StandardScaler because it prevents features from dominating. 

*Feature Extraction*
- Histogram of Gradients (HoG): Extracts shape and texture information from grayscale images.
- Scale-Invariant Feature Transform (SIFT) : It detects and describes key points,computes 128-dimensional descriptors and averages them for a fixed feature vector.

*Machine Learning Models*
- Support Vector Machine (SVM) : It is a supervised learning algorithm that finds the optimal hyperplane to separate data points into different classes by maximizing the margin between them. We use Radial Basis Function (RBF) kernel with C=2 for optimal decision boundary.

- Logistic Regression : Logistic Regression is a statistical model that predicts the probability of a binary outcome using a linear combination of input features and the sigmoid function to map outputs between 0 and 1.It uses L1 & L2 regularization with C=0.5 to prevent overfitting.

- Multi-layer Perceptron (MLP) : There are 3 hidden layers (256, 128, 64), trained with ReLU activation, adam optimizer is used and max_iter=500 to ensure learning.

- Convolutional Neural Network (CNN) : The grayscale images to are reshaped to (128, 128, 1) shape. Pixel values are normalized for better convergence. The output is one-hot encoded. Test-train split of 20-80 is performed. 


## Dataset

**Part-1**

- *Face Mask Detection Dataset:* [GitHub Link](https://github.com/chandrikadeb7/Face-Mask-Detection/tree/master/dataset)

```bash
├── with_mask/ 
├── without_mask/
```

- with_mask/: Contains images of people with mask.
- without_mask/: Contains images of people without mask.


## Results

**Part-1**

Best Model Accuracy: 0.9523227214813232

![Confusion Matrix](https://github.com/Sreyas-J/VR_Project1_Sreyas_IMT2022554_Pradyumna_IMT2022555_Chinmay_IMT2022561/blob/main/ConfusionMatrix.png) 


## Run Locally

Cloning the Github repository using SSH

```bash
  git clone git@github.com:Sreyas-J/VR_Assignment1_SreyasJanamanchi_IMT2022554.git
```
or

Cloning the Github repository using HTTPS

```bash
    git clone https://github.com/Sreyas-J/VR_Assignment1_SreyasJanamanchi_IMT2022554.git
```

Installing the required packages in a conda environment

```bash
    cd VR_Assignment1_SreyasJanamanchi_IMT2022554
    conda create --name <env_name> --file requirements.txt
```

Running part1

The input can be changed by updating the following variable: input_image

```bash
    cd part1
    python3 part1.py
```

Running part2

The inputs can be changed by updating the following variables:-
- input_folder
- NumberOfInputs

```bash
    cd part2
    python3 part2.py
```

