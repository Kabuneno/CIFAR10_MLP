# CIFAR-10 Neural Network Classifier

This project implements a simple neural network from scratch using NumPy to classify images from the CIFAR-10 dataset. The neural network uses a 3-layer architecture with ReLU activations and softmax output.

## Project Structure

- `activations.py`: Contains activation functions (ReLU, Softmax) and one-hot encoding
- `loss.py`: Implements cross-entropy loss function
- `network.py`: Defines forward and backward propagation functions
- `main.py`: Entry point that loads data, trains the model, and evaluates performance

## Architecture

The neural network has the following architecture:
- Input layer: 3072 neurons (32×32×3 flattened RGB images)
- First hidden layer: 512 neurons with ReLU activation
- Second hidden layer: 256 neurons with ReLU activation
- Output layer: 10 neurons with Softmax activation (for 10 CIFAR-10 classes)

## CIFAR-10 Classes

The model classifies images into these 10 categories:
1. Airplane
2. Automobile
3. Bird
4. Cat
5. Deer
6. Dog
7. Frog
8. Horse
9. Ship
10. Truck

## Requirements

- NumPy
- Matplotlib
- TensorFlow (for loading the CIFAR-10 dataset)
- scikit-learn (for confusion matrix visualization)
- tqdm (for progress bars)

## Installation

```bash
pip install numpy matplotlib tensorflow scikit-learn tqdm
```

## Usage

Run the training and evaluation:

```bash
python main.py
```

This will:
1. Load and preprocess the CIFAR-10 dataset
2. Train the neural network on the training data
3. Evaluate the model on test data
4. Display a confusion matrix of the results
5. Visualize predictions on sample images

## Model Details

- **Initialization**: Small random weights (scaled by 0.01)
- **Optimization**: Mini-batch gradient descent
- **Batch Size**: 32
- **Learning Rate**: 0.1
- **Epochs**: 20

## Performance

The model achieves reasonable accuracy for a simple neural network implementation. The confusion matrix visualization helps identify which classes are most challenging for the model.

## Implementation Details

This implementation follows the standard feedforward and backpropagation algorithms:

1. **Forward Pass**:
   - Linear transformation (Z = X @ W + b)
   - ReLU activation (A = max(0, Z))
   - Final layer uses Softmax for classification probabilities

2. **Backward Pass**:
   - Compute gradients using chain rule
   - Update weights with gradient descent

