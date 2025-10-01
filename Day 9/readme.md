[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1irvXd86APCWU3UedGPys730jAGMWaXxw?usp=sharing)

# Transfer Learning with Oxford Flowers 102 Dataset

This project demonstrates the application of transfer learning techniques using pre-trained convolutional neural networks (ResNet50, VGG16, and MobileNetV2) to classify images from the Oxford Flowers 102 dataset. The performance of the different models on this dataset is compared.

## Dataset

The dataset used is the **Oxford Flowers 102** dataset, consisting of 102 categories of flowers. It is loaded using TensorFlow Datasets (`tfds.load('oxford_flowers102:2.1.1')`) and split into training, validation, and testing sets.

## Models Used

The project utilizes three popular pre-trained CNN models:

*   **ResNet50:** A deep residual network known for its skip connections, which help in training very deep networks.
*   **VGG16:** A simpler architecture with a uniform structure of repeated convolutional and max-pooling layers.
*   **MobileNetV2:** A lightweight architecture designed for mobile and embedded vision applications, using inverted residual blocks and depthwise separable convolutions.

These models were pre-trained on the ImageNet dataset.

## Implementation

The implementation involves the following steps:

1.  **Data Loading and Exploration:** Loading the dataset using TensorFlow Datasets and exploring its structure and characteristics.
2.  **Data Preprocessing:**
    *   Resizing images to a fixed size (224x224 pixels) suitable for the pre-trained models.
    *   Applying model-specific preprocessing functions to normalize pixel values.
    *   Applying one-hot encoding to the labels.
    *   Batching and prefetching the datasets for efficient training.
3.  **Model Adaptation:**
    *   Loading each pre-trained model (ResNet50, VGG16, MobileNetV2) without its top classification layer (`include_top=False`).
    *   Adding new custom classification layers (GlobalAveragePooling2D and a Dense layer with 102 units and softmax activation) on top of the pre-trained base.
    *   Initially freezing the layers of the pre-trained base model.
4.  **Model Compilation:** Compiling each adapted model with the 'adam' optimizer and 'categorical\_crossentropy' loss (since using one-hot encoded labels), and 'accuracy' as a metric.
5.  **Training:**
    *   **Initial Training:** Training the models for a few epochs with the base layers frozen, allowing the new classification layers to learn to map the pre-extracted features to the flower classes.
    *   **Fine-tuning:** Unfreezing a portion of the pre-trained base model's layers and continuing training with a significantly lower learning rate to adapt the pre-trained features more specifically to the Oxford Flowers 102 dataset.
6.  **Model Evaluation:** Evaluating the performance of each trained model on the test dataset using loss and accuracy metrics.
7.  **Analysis and Visualization:** Analyzing the training history (accuracy and loss plots) and the final evaluation results to compare the performance of the different models.

## How to Implement

To run this project on your system:

1.  **Environment Setup:** Ensure you have Python and TensorFlow installed. Using a virtual environment is recommended. You will also need `tensorflow-datasets` and `matplotlib`.