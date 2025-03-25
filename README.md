
# PROJECT-1

**Part-1**

The goal is to classify facial images as "with mask" or "without mask" using two approaches:

- **ML with Handcrafted Features** – Extract features, train ML classifiers (e.g., SVM, Neural Network), and compare their accuracy.
- **CNN-based Classification** – Design, train, and optimize a CNN for the same task, testing different hyperparameters.

**Part-2**

This project explores image segmentation using a pre-trained U-Net model, comparing its performance with traditional methods like Otsu’s Thresholding and Canny Edge Detection. In addition, we experiment with various hyperparameter configurations to optimize the U-Net model and achieve the best possible segmentation performance. The goal is to assess the effectiveness of deep learning-based segmentation in contrast to classical techniques.


## Dataset

**Part-1**

- *Face Mask Detection Dataset:* [GitHub Link](https://github.com/chandrikadeb7/Face-Mask-Detection/tree/master/dataset)

```bash
├── with_mask/ 
├── without_mask/
```

- with_mask/: Contains images of people with mask.
- without_mask/: Contains images of people without mask.

**Part-2**

- **Source**: The dataset consists of images of people wearing masks, along with their corresponding ground truth segmentation masks : [MFSD](https://github.com/sadjadrz/MFSD)
- **Images_Path**: `./MSFD/1/face_crop`
- **Ground_Truth_Path**: `./MSFD/1/face_crop_segmentation`
- **Structure**: 
```bash
    ├── MFSD
        ├── 1
            ├── face_crop
            ├── face_crop_segmentation
            ├── img
```
## Run Locally

Cloning the Github repository using SSH

```bash
  git clone git@github.com:Sreyas-J/VR_Project1_Sreyas_IMT2022554_Pradyumna_IMT2022555_Chinmay_IMT2022561.git
```
or

Cloning the Github repository using HTTPS

```bash
    git clone https://github.com/Sreyas-J/VR_Project1_Sreyas_IMT2022554_Pradyumna_IMT2022555_Chinmay_IMT2022561.git
    cd VR_Project1_Sreyas_IMT2022554_Pradyumna_IMT2022555_Chinmay_IMT2022561
```

Installing the required packages for part-1

```bash
    pip install opencv-python numpy scikit-image scikit-learn tensorflow keras-tuner matplotlib seaborn
```

Installing the required packages for part-2

```bash
    pip install pip install opencv-python numpy scikit-learn matplotlib tensorflow segmentation-models
```

Running part1

```bash
    python3 part1.py
```

Running part2

```bash
    python3 part2.py
```


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

**Part-2**

### 1.Data Preprocessing
#### **i) Loading and Preprocessing**
- Images and masks are read from the dataset.
- Images are resized to 128x128 and normalized(**values scaled to [0,1]**).
- Masks are binarized to ensure clear segmentation boundaries(**values to 0 or 1**).
#### **ii) Splitting the Dataset**
- The dataset is split into training and testing sets (**80% test, 20% train**).

### 2.Classical Segmentation
- **Grayscale Conversion & Smoothing**: Convert RGB images to grayscale and apply Gaussian blur.

- **Thresholding & Edge Detection**:
    
    - Apply Otsu’s thresholding for segmentation
    - Use Canny edge detection for boundary enhancement.

Intersection over Union (IoU) and Dice coefficient are used to compare predicted masks with ground truth. (**Given in the latter section**)

**Classical Segmentation Visualization**

![Classical Segmentation Visualization](https://github.com/Sreyas-J/VR_Project1_Sreyas_IMT2022554_Pradyumna_IMT2022555_Chinmay_IMT2022561/blob/main/Traditional.jpeg)

### 3.Deep Learning Based Approach
#### i) **Model Selection**:
- Backbone: We use ResNet50, VGG16 and VGG19.
- A U-Net architecture with a a backbone from above is used.
- Pretrained weights (ImageNet) are utilized for improved performance.

#### ii) **Loss Function**:

- Both Binary cross-entropy and Dice loss have been checked as a loss function.

#### iii) **Training**:

- Training is performed over:
    - Epochs: 10, 15
    - Batch size: 8, 16
- **Optimizer**: 
    - Adam
    - SGD: The model is trained using SGD with a learning rate of 0.01 and momentum of 0.9.  

#### iv) **Evaluation**:

- IoU and Dice coefficient are computed to measure segmentation accuracy.
![UNET](https://github.com/Sreyas-J/VR_Project1_Sreyas_IMT2022554_Pradyumna_IMT2022555_Chinmay_IMT2022561/blob/main/UNET.jpeg)
## Hyperparameter Tuning

**Part-1**

| Activation  | Dropout | LR    | Filters 1 | Filters 2 | Filters 3 | Filters 4 | FC Neurons | Accuracy |
|------------|---------|-------|-----------|-----------|-----------|-----------|------------|----------|
| relu       | 0.3     | 0.001 | 32        | 48        | 128       | 96        | 256        | 0.9413   |
| relu       | 0.2     | 0.0001| 64        | 32        | 96        | 64        | 256        | 0.9401   |
| leaky_relu | 0.3     | 0.0001| 48        | 32        | 96        | 96        | 384        | 0.9083   |
| leaky_relu | 0.3     | 0.0001| 32        | 48        | 128       | 64        | 384        | 0.8899   |
| leaky_relu | 0.3     | 0.0001| 32        | 32        | 128       | 64        | 256        | 0.9108   |
| leaky_relu | 0.3     | 0.01  | 64        | 48        | 96        | 96        | 128        | 0.8704   |
| relu       | 0.4     | 0.01  | 48        | 64        | 96        | 64        | 512        | 0.8716   |
| relu       | 0.3     | 0.01  | 64        | 32        | 64        | 64        | 128        | 0.9156   |
| leaky_relu | 0.2     | 0.01  | 64        | 32        | 128       | 64        | 128        | 0.8068   |
| relu       | 0.4     | 0.01  | 48        | 48        | 96        | 64        | 512        | 0.7469   |
| relu       | 0.3     | 0.001  | 32        | 48        | 128        | 96        | 256        | 0.9511  |

**Part-2**

## Hyperparameter Results(U-Net)
### i) Adam Optimizer
|Backbone  | Loss Function        | Epoch| Batch Size | IoU Score | Dice Score |
|-----------|----------------------|-- |------------|-----------|------------|
| ResNet-50 | Dice Loss             |10| 8          | 0.8863    | 0.9334     |
| VGG16     | Binary Cross Entropy  |10| 8          | 0.8512    | 0.9092     |
| VGG19     | Binary Cross Entropy  |10 | 8          | 0.8972    | 0.9402    |
| ResNet-50 | Binary Cross Entropy  |10 | 8          | 0.9036    | 0.9443    |
| ResNet-50 | Binary Cross Entropy  |15 | 8         | 0.9059    | 0.9453     |
| ResNet-50 | Binary Cross Entropy  |10 | 16         | 0.8908    | 0.9366    | 

---
### ii) SGD with Batch size 8 and Epochs 10
|Backbone  | Loss Function        |Learning Rate| IoU Score | Dice Score |
|-----------|----------------------|------------|-----------|------------|
| ResNet-50 | Binary Cross Entropy  |0.01 | 0.9092    | 0.9481    | 
| ResNet-50 | Binary Cross Entropy  |0.1 | 0.9136    | 0.9503    |

---

## Observations

**Part-1**

### CNN
**1. Highest Accuracy (0.9511)**
- **Configuration**: ReLU, Dropout 0.3, LR 0.001, Filters: [32, 48, 128, 96], FC Neurons: 256  
- **Reason**:
  - Balanced filter sizes ensure optimal feature extraction.
  - ReLU prevents vanishing gradients and speeds up training.
  - Dropout (0.3) prevents overfitting while maintaining learning capacity.
  - LR (0.001) provides stable convergence.
  - 256 FC neurons balance model complexity and generalization.

**2. High Accuracy (0.9401 - 0.9413)**
- **Configuration**: ReLU, LR 0.001 / 0.0001, Dropout 0.2 - 0.3  
- **Reason**:
  - Lower dropout retains more neurons for better learning.
  - LR (0.001) enables smooth convergence; 0.0001 allows fine-tuning.
  - Filters [32, 48, 128, 96] and [64, 32, 96, 64] capture diverse features.

**3. Moderate Accuracy (0.9083 - 0.9108)**
- **Configuration**: Leaky ReLU, LR 0.0001, Dropout 0.3  
- **Reason**:
  - Leaky ReLU prevents dead neurons but offers minimal advantage in deep CNNs.
  - LR (0.0001) provides precise weight adjustments but slows convergence.
  - Filter choices allow diverse feature extraction.

**4. Drop in Accuracy (~0.87 - 0.89)**
- **Configuration**: Leaky ReLU, LR 0.01, Dropout 0.3  
- **Reason**:
  - High LR (0.01) may have caused unstable weight updates.
  - Dropout 0.3 prevents overfitting but might remove key neurons.
  - Filters [64, 48, 96, 96] still extract important features, but the high LR hinders stability.

**5. Significant Drop in Accuracy (0.8068 - 0.7469)**
- **Configuration**: Leaky ReLU / ReLU, LR 0.01, Dropout 0.4  
- **Reason**:
  - Dropout (0.4) removes too many neurons, reducing model capacity.
  - High LR (0.01) likely causes weight overshooting and unstable training.
  - Filters are well-designed, but aggressive regularization limits learning

**Key Takeaways**

- **Best Performer**: ReLU, Dropout 0.3, LR 0.001, well-balanced filters.  
- **Leaky ReLU vs. ReLU**: Leaky ReLU performed slightly worse, likely due to diminishing benefits in deep CNNs.  
- **Learning Rate Impact**: 0.001 is optimal, while 0.01 causes instability.  
- **Dropout Effect**: 0.3 is ideal; 0.4 leads to performance degradation.  
- **Over-regularization Hurts**: Too high dropout and LR lead to accuracy drops. 

**Part-2**

- We can clearly observe Deep learning based aproach(**U-net**) outperforms traditional based segmentation tasks.
- VGG 16 achieves the lowest IOU and Dice score. 
- VGG19 performs better compared to VGG16 indicating that deeper architectures provide improved feature extraction for segmentation.
- ResNet-50 with Binary Cross Entropy achieves the highest IoU and Dice Score.
- Changing the number of epochs between 10 and 15 didn't give a huge difference in the scores, so we continued with 8 epochs for rest of the tunings.
- Dice Loss underperforms compared to Binary Cross Entropy Loss function.
- Increasing the batch size from 8 to 16 slightly reduces performance.
- SGD outperforms Adam optimizer in all cases. 
- SGD with a learning rate of 0.1 has the best performance.
- Hence, the best model: ResNet-50 + BCE + SGD(0.1 learning rate) with 0.9136 IoU score and 0.9503 Dice score.
## Results

**Part-1**

- SVM Accuracy: 0.9388
- Logistic Regression: 0.9022
- MLP: 0.9437
- Best CNN Model Accuracy: 0.9511

**CNN Confusion Matrix**
![Confusion Matrix](https://github.com/Sreyas-J/VR_Project1_Sreyas_IMT2022554_Pradyumna_IMT2022555_Chinmay_IMT2022561/blob/main/ConfusionMatrix.png) 

**Part-2**

| Approach                         | IoU Score | Dice Score |
|-----------------------------------|-----------|------------|
| **Traditional Segmentation (Otsu + Canny)** | 0.2779    | 0.0015     |
| **Pretrained U-Net (Best)** | 0.9136    | 0.9503     |

**Best**:
- Epochs: 10
- Loss Function: Binary Cross Entropy
- Batch size: 8
- Optimizer: SGD with learning rate 0.1
## Screenshots

![App Screenshot](https://github.com/Sreyas-J/VR_Project1_Sreyas_IMT2022554_Pradyumna_IMT2022555_Chinmay_IMT2022561/blob/main/ConfusionMatrix.png)


