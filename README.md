COMPANY: CODTECH IT SOLUTIONS

NAME: SAHANA

INTERN ID: CT04DZ1200

DOMAIN: MACHINE LEARNING

DURATION: 4 WEEKS

MENTOR: NEELA SANTOSH

##DESCRIPTION OF THE TASK 3: IMAGE-CLASSIFICATION-MODEL

The provided Python script is a complete implementation of a Convolutional Neural Network (CNN) using PyTorch, designed to classify images from the CIFAR-10 dataset. The CIFAR-10 dataset consists of 60,000 color images across ten different categories: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. Each image is 32x32 pixels in size and contains three color channels (RGB). This model is trained to distinguish among these classes using a relatively simple CNN architecture. The program is divided into several logical sections: importing necessary modules, configuring the GPU device, loading and normalizing the dataset, defining the CNN model architecture, specifying the loss function and optimizer, training the model, evaluating its accuracy on the test set, and finally, visualizing a few predictions. Let us explore each component in detail, along with its methods and practical significance.This task is done by using chatgpt.com. 

To begin with, the script imports several core libraries. The torch module is the backbone of PyTorch and provides core functionalities for tensor computation and automatic differentiation. The torchvision module is specifically designed for computer vision tasks, offering popular datasets, model architectures, and image transformation utilities. The transforms module within torchvision is used for preprocessing the images, such as converting them into tensors and normalizing their pixel values. nn from torch refers to the neural network package, containing layers like convolutional layers (Conv2d), pooling layers (MaxPool2d), fully connected layers (Linear), and activation functions like ReLU. The optim module provides various optimization algorithms like Adam, which is used here for training the network by adjusting 
the weights based on computed gradients. matplotlib.pyplot is imported as plt for visualizing the images and their predictions, and numpy is used for array manipulation and display adjustments in image plots.

The next section of the script sets up the computing device. The line torch.device('cuda' if torch.cuda.is_available() else 'cpu') detects if a GPU is available and sets it as the primary device. If no GPU is detected, it falls back to the CPU. This ensures that the script runs efficiently on machines with CUDA-enabled GPUs while remaining compatible with standard CPU-only systems.

The CIFAR-10 dataset is then loaded and normalized. The transformation applied to the images involves converting them to PyTorch tensors and normalizing their RGB pixel values to lie between -1 and 1, which 
is achieved by subtracting 0.5 and dividing by 0.5. This normalization ensures that all features contribute equally to the loss function, helping the network converge faster. The training and test datasets 
are downloaded using torchvision.datasets.CIFAR10 with the appropriate train flag. These datasets are wrapped in DataLoader objects, which efficiently handle batching, shuffling (for training), and parallel 
data loading. The batch size is set to 32, meaning the model processes 32 images at a time during training and evaluation.

The heart of the script is the SimpleCNN class, which defines the architecture of the convolutional neural network. The network begins with a convolutional layer (conv1) that takes 3 input channels (RGB) and produces 32 output channels using 3x3 filters with padding of 1 to maintain spatial dimensions. The output is then passed through a ReLU activation function to introduce non-linearity and is followed by a 2x2 
max-pooling layer that reduces the spatial resolution by half. This pattern is repeated with a second convolutional layer (conv2) that increases the channel depth to 64, followed by another ReLU and max-pooling layer. After the convolutional and pooling operations, the feature map is flattened into a one-dimensional vector using view(), preparing it for the fully connected layers. The first fully connected layer (fc1) has 128 units and is followed by another ReLU activation. The final layer (fc2) outputs a 10-dimensional vector representing the scores for each class.

The model is then transferred to the selected computing device using model.to(device). The loss function used is Cross Entropy Loss, which is ideal for multi-class classification problems as it measures the difference between the predicted class distribution and the actual label distribution. The Adam optimizer is chosen for updating the model weights, known for its adaptive learning rate and efficiency in practice.

The training loop runs for 10 epochs. For each batch of images, the model computes predictions, calculates the loss, performs backpropagation to compute gradients, and updates the weights using the optimizer.
The loss is accumulated to monitor training performance. After training, the model is evaluated on the test dataset in a no-gradient context (torch.no_grad()) for efficiency. Predictions are compared with actual labels to compute accuracy.

Finally, the model's predictions are visualized. A batch of test images is displayed using matplotlib, and the predicted and actual class labels for the first four images are printed. The imshow() function reverses the earlier normalization for correct image display.

This CNN model has a wide range of practical applications in image classification, particularly in areas like autonomous vehicles (identifying objects), security systems (face or object recognition), medical diagnostics (classifying MRI or X-ray images), and even mobile apps for real-time image tagging or visual search. It demonstrates how CNNs can be trained effectively using PyTorch with basic components and can
be further improved with techniques like data augmentation, dropout, and batch normalization. This script provides a foundational template for anyone entering the field of deep learning and computer vision.

##OUTPUT

img width="1770" height="401" alt="Image" src="https://github.com/user-attachments/assets/36b76df5-b024-46b6-b660-0b02e3fefe7c" />
<img width="1741" height="463" alt="Image" src="https://github.com/user-attachments/assets/548dbfd7-b935-4a91-a245-4e61dbb3bbba" />
