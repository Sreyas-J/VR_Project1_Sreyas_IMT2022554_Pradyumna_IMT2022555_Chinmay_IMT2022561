import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
import tensorflow.keras.backend as K
# import segmentation_models as sm

# Configure environment
# os.environ["SM_FRAMEWORK"] = "tf.keras"

import segmentation_models as sm

def load_image_dataset(img_directory, annotation_directory):
    """Load and preprocess image-mask pairs"""
    image_collection = []
    annotation_collection = []
    
    sorted_img_files = sorted(os.listdir(img_directory))

    for img_file in sorted_img_files:
        full_img_path = os.path.join(img_directory, img_file)
        full_mask_path = os.path.join(annotation_directory, img_file)

        if os.path.exists(full_mask_path):
            # Read and resize image
            original_img = cv2.imread(full_img_path)
            processed_img = cv2.resize(original_img, (128, 128)) / 255.0
            
            # Read and process mask
            original_mask = cv2.imread(full_mask_path, cv2.IMREAD_GRAYSCALE)
            resized_mask = cv2.resize(original_mask, (128, 128), 
                                     interpolation=cv2.INTER_NEAREST)
            
            # Binarize mask
            _, binary_mask = cv2.threshold(resized_mask, 127, 255, cv2.THRESH_BINARY)
            normalized_mask = binary_mask / 255.0

            image_collection.append(processed_img)
            annotation_collection.append(normalized_mask)

    return np.array(image_collection), np.array(annotation_collection)

# Path configuration
input_img_dir = "MSFD/1/face_crop"
input_mask_dir = "MSFD/1/face_crop_segmentation"

# Load and split dataset
input_images, target_masks = load_image_dataset(input_img_dir, input_mask_dir)
train_images, test_images, train_masks, test_masks = train_test_split(
    input_images, target_masks, test_size=0.8, random_state=42)

# Classical Computer Vision Approach
def classical_segmentation_approach(rgb_image):
    """Apply traditional segmentation methods"""
    grayscale = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
    smoothed = cv2.GaussianBlur(grayscale, (5, 5), 0)

    # Thresholding
    _, thresholded = cv2.threshold(smoothed, 0, 255, 
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Edge detection
    detected_edges = cv2.Canny(smoothed, 100, 200)

    return thresholded, detected_edges

def assess_classical_method(image_set, mask_set):
    """Evaluate traditional segmentation performance"""
    intersection_over_union = []
    dice_coefficients = []

    for img, gt_mask in zip(image_set, mask_set):
        predicted_mask, _ = classical_segmentation_approach((img * 255).astype(np.uint8))
        
        # Prepare ground truth
        gt_mask_uint8 = (gt_mask * 255).astype(np.uint8)
        _, gt_mask_binary = cv2.threshold(gt_mask_uint8, 127, 255, cv2.THRESH_BINARY)

        # Calculate metrics
        overlap = np.logical_and(predicted_mask, gt_mask_binary).sum()
        combined = np.logical_or(predicted_mask, gt_mask_binary).sum()

        iou = overlap / (combined + 1e-6)
        dice = (2. * overlap) / (predicted_mask.sum() + gt_mask_binary.sum() + 1e-6)

        intersection_over_union.append(iou)
        dice_coefficients.append(dice)

    print(f"Classical Method - IoU: {np.mean(intersection_over_union):.4f}, "
          f"Dice: {np.mean(dice_coefficients):.4f}")

def display_comparison(image_set, mask_set, sample_count=5):
    """Visualize comparison between methods"""
    plt.figure(figsize=(12, sample_count * 4))

    for idx in range(sample_count):
        current_img = (image_set[idx] * 255).astype(np.uint8)
        current_gt = (mask_set[idx] * 255).astype(np.uint8)
        classical_result, edge_map = classical_segmentation_approach(current_img)

        # Original image
        plt.subplot(sample_count, 4, idx * 4 + 1)
        plt.imshow(cv2.cvtColor(current_img, cv2.COLOR_BGR2RGB))
        plt.title("Input Image")
        plt.axis("off")

        # Ground truth
        plt.subplot(sample_count, 4, idx * 4 + 2)
        plt.imshow(current_gt, cmap="gray")
        plt.title("Reference Segmentation")
        plt.axis("off")

        # Classical result
        plt.subplot(sample_count, 4, idx * 4 + 3)
        plt.imshow(classical_result, cmap="gray")
        plt.title("Classical Segmentation")
        plt.axis("off")

        # Edge detection
        plt.subplot(sample_count, 4, idx * 4 + 4)
        plt.imshow(edge_map, cmap="gray")
        plt.title("Edge Detection")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

# Evaluate classical approach
assess_classical_method(test_images, test_masks)
display_comparison(test_images, test_masks)

# Deep Learning Approach

def dice_loss(y_true, y_pred):
    smooth = 1e-6  # To avoid division by zero
    y_true_f = K.flatten(y_true)  # Flatten ground truth mask
    y_pred_f = K.flatten(y_pred)  # Flatten predicted mask
    
    intersection = K.sum(y_true_f * y_pred_f)  # Compute intersection
    union = K.sum(y_true_f) + K.sum(y_pred_f)  # Compute union
    
    dice = (2. * intersection + smooth) / (union + smooth)  # Dice coefficient
    return 1 - dice  # Dice loss = 1 - Dice coefficient


def build_segmentation_model():
    """Configure and compile segmentation model"""
    base_architecture = "resnet50"
    # base_architecture = "vgg19"
    
    segmentation_model = sm.Unet(
        base_architecture,
        encoder_weights="imagenet",
        input_shape=(128, 128, 3),
        classes=1,
        activation="sigmoid"
    )
    
    segmentation_model.compile(
        # optimizer="adam",
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    
    return segmentation_model

# Train the model
dl_model = build_segmentation_model()
training_history = dl_model.fit(
    train_images,
    train_masks,
    validation_split=0.2,
    epochs=10,
    batch_size=8
)

def evaluate_dl_model(model, image_set, mask_set):
    """Evaluate deep learning model performance"""
    predictions = model.predict(image_set)
    iou_results = []
    dice_results = []

    for pred, gt in zip(predictions, mask_set):
        binary_pred = (pred.squeeze() > 0.5).astype(np.uint8)
        binary_gt = (gt > 0.5).astype(np.uint8)

        overlap_area = np.logical_and(binary_pred, binary_gt).sum()
        union_area = np.logical_or(binary_pred, binary_gt).sum()

        iou = overlap_area / (union_area + 1e-6)
        dice = (2. * overlap_area) / (binary_pred.sum() + binary_gt.sum() + 1e-6)

        iou_results.append(iou)
        dice_results.append(dice)

    print(f"Deep Learning Model - IoU: {np.mean(iou_results):.4f}, "
          f"Dice: {np.mean(dice_results):.4f}")

def visualize_dl_results(model, image_set, mask_set, num_examples=5):
    """Display deep learning segmentation results"""
    model_predictions = model.predict(image_set[:num_examples]) > 0.5

    fig, ax = plt.subplots(num_examples, 3, figsize=(12, num_examples * 4))

    for i in range(num_examples):
        # Original image
        ax[i, 0].imshow(cv2.cvtColor((image_set[i] * 255).astype(np.uint8), 
                                    cv2.COLOR_BGR2RGB))
        ax[i, 0].set_title("Input Image")
        ax[i, 0].axis("off")

        # Ground truth
        ax[i, 1].imshow((mask_set[i] * 255).astype(np.uint8), cmap="gray")
        ax[i, 1].set_title("Reference Mask")
        ax[i, 1].axis("off")

        # Prediction
        ax[i, 2].imshow((model_predictions[i].squeeze() * 255).astype(np.uint8), 
                       cmap="gray")
        ax[i, 2].set_title("Model Prediction")
        ax[i, 2].axis("off")

    plt.tight_layout()
    plt.show()

# Evaluate deep learning model
visualize_dl_results(dl_model, test_images, test_masks)
evaluate_dl_model(dl_model, test_images, test_masks)