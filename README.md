# Neural Network Playground

An interactive web-based visualization tool for understanding how neural networks learn. Watch decision boundaries form in real-time as you train networks on various datasets.

![Neural Network Playground](https://img.shields.io/badge/ML-Visualization-blue)
![JavaScript](https://img.shields.io/badge/JavaScript-ES6+-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

## Features

- **Interactive Training**: Start/stop training and watch the network learn in real-time
- **Multiple Datasets**: Circle, XOR, Spiral, and Gaussian classification problems
- **Customizable Architecture**: Adjust hidden layers (1-6) and neurons per layer (1-8)
- **Activation Functions**: Choose between ReLU, Tanh, and Sigmoid
- **Learning Rate Control**: Fine-tune training speed from 0.001 to 0.3
- **Live Visualizations**:
  - Decision boundary heatmap
  - Network architecture with weight visualization
  - Training loss chart

## Demo

Open `index.html` in any modern browser to run the playground.

## How It Works

The playground implements a feedforward neural network from scratch using vanilla JavaScript:

- **Forward Propagation**: Computes activations layer by layer
- **Backpropagation**: Calculates gradients using the chain rule
- **Gradient Descent**: Updates weights to minimize binary cross-entropy loss
- **Xavier Initialization**: Smart weight initialization for stable training

## Usage

1. Select a dataset (Circle is easiest, Spiral is hardest)
2. Configure the network architecture
3. Click "Train" to start training
4. Watch the decision boundary evolve
5. Experiment with different settings!

## Technologies

- Pure JavaScript (no ML libraries)
- HTML5 Canvas for visualizations
- CSS3 with modern design

## Author

Built by [Parsa Faraji Alamouti](https://github.com/parsa-faraji)

## License

MIT License
