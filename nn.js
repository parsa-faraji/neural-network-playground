/**
 * Neural Network Implementation
 * A simple feedforward neural network with backpropagation
 */

class NeuralNetwork {
    constructor(layerSizes, activation = 'relu', learningRate = 0.01) {
        this.layerSizes = layerSizes;
        this.learningRate = learningRate;
        this.activation = activation;
        this.weights = [];
        this.biases = [];
        this.initializeWeights();
    }

    initializeWeights() {
        this.weights = [];
        this.biases = [];

        for (let i = 0; i < this.layerSizes.length - 1; i++) {
            const inputSize = this.layerSizes[i];
            const outputSize = this.layerSizes[i + 1];

            // Xavier initialization
            const scale = Math.sqrt(2.0 / (inputSize + outputSize));
            const weight = [];
            for (let j = 0; j < inputSize; j++) {
                const row = [];
                for (let k = 0; k < outputSize; k++) {
                    row.push((Math.random() * 2 - 1) * scale);
                }
                weight.push(row);
            }
            this.weights.push(weight);

            const bias = [];
            for (let j = 0; j < outputSize; j++) {
                bias.push(0);
            }
            this.biases.push(bias);
        }
    }

    activate(x) {
        switch (this.activation) {
            case 'relu':
                return Math.max(0, x);
            case 'tanh':
                return Math.tanh(x);
            case 'sigmoid':
                return 1 / (1 + Math.exp(-Math.max(-500, Math.min(500, x))));
            default:
                return Math.max(0, x);
        }
    }

    activateDerivative(x) {
        switch (this.activation) {
            case 'relu':
                return x > 0 ? 1 : 0;
            case 'tanh':
                const t = Math.tanh(x);
                return 1 - t * t;
            case 'sigmoid':
                const s = 1 / (1 + Math.exp(-Math.max(-500, Math.min(500, x))));
                return s * (1 - s);
            default:
                return x > 0 ? 1 : 0;
        }
    }

    forward(input) {
        let activations = [input];
        let zValues = [input];
        let current = input;

        for (let layer = 0; layer < this.weights.length; layer++) {
            const newValues = [];
            const newZ = [];

            for (let j = 0; j < this.weights[layer][0].length; j++) {
                let sum = this.biases[layer][j];
                for (let i = 0; i < current.length; i++) {
                    sum += current[i] * this.weights[layer][i][j];
                }
                newZ.push(sum);

                // Use sigmoid for output layer, activation function for hidden
                if (layer === this.weights.length - 1) {
                    newValues.push(1 / (1 + Math.exp(-Math.max(-500, Math.min(500, sum)))));
                } else {
                    newValues.push(this.activate(sum));
                }
            }

            zValues.push(newZ);
            activations.push(newValues);
            current = newValues;
        }

        return { output: current, activations, zValues };
    }

    predict(input) {
        return this.forward(input).output[0];
    }

    train(inputs, targets) {
        let totalLoss = 0;
        let correct = 0;

        for (let sample = 0; sample < inputs.length; sample++) {
            const input = inputs[sample];
            const target = targets[sample];

            // Forward pass
            const { output, activations, zValues } = this.forward(input);

            // Calculate loss (binary cross-entropy)
            const prediction = output[0];
            const clampedPred = Math.max(1e-7, Math.min(1 - 1e-7, prediction));
            totalLoss += -(target * Math.log(clampedPred) + (1 - target) * Math.log(1 - clampedPred));

            if ((prediction >= 0.5 && target === 1) || (prediction < 0.5 && target === 0)) {
                correct++;
            }

            // Backward pass
            const deltas = [];

            // Output layer delta
            const outputDelta = [prediction - target];
            deltas.unshift(outputDelta);

            // Hidden layer deltas
            for (let layer = this.weights.length - 2; layer >= 0; layer--) {
                const layerDelta = [];
                for (let i = 0; i < this.weights[layer][0].length; i++) {
                    let sum = 0;
                    for (let j = 0; j < deltas[0].length; j++) {
                        sum += deltas[0][j] * this.weights[layer + 1][i][j];
                    }
                    layerDelta.push(sum * this.activateDerivative(zValues[layer + 1][i]));
                }
                deltas.unshift(layerDelta);
            }

            // Update weights and biases
            for (let layer = 0; layer < this.weights.length; layer++) {
                for (let i = 0; i < this.weights[layer].length; i++) {
                    for (let j = 0; j < this.weights[layer][i].length; j++) {
                        this.weights[layer][i][j] -= this.learningRate * deltas[layer][j] * activations[layer][i];
                    }
                }
                for (let j = 0; j < this.biases[layer].length; j++) {
                    this.biases[layer][j] -= this.learningRate * deltas[layer][j];
                }
            }
        }

        return {
            loss: totalLoss / inputs.length,
            accuracy: correct / inputs.length
        };
    }

    getWeights() {
        return this.weights;
    }
}

// Dataset generators
const Datasets = {
    circle: (n = 200) => {
        const inputs = [];
        const targets = [];
        for (let i = 0; i < n; i++) {
            const x = Math.random() * 2 - 1;
            const y = Math.random() * 2 - 1;
            const distance = Math.sqrt(x * x + y * y);
            inputs.push([x, y]);
            targets.push(distance < 0.5 ? 1 : 0);
        }
        return { inputs, targets };
    },

    xor: (n = 200) => {
        const inputs = [];
        const targets = [];
        for (let i = 0; i < n; i++) {
            const x = Math.random() * 2 - 1;
            const y = Math.random() * 2 - 1;
            inputs.push([x, y]);
            targets.push((x > 0) !== (y > 0) ? 1 : 0);
        }
        return { inputs, targets };
    },

    spiral: (n = 200) => {
        const inputs = [];
        const targets = [];
        const pointsPerClass = n / 2;

        for (let i = 0; i < pointsPerClass; i++) {
            const r = i / pointsPerClass;
            const t = 1.75 * i / pointsPerClass * 2 * Math.PI + Math.random() * 0.2;
            inputs.push([r * Math.cos(t), r * Math.sin(t)]);
            targets.push(0);
        }

        for (let i = 0; i < pointsPerClass; i++) {
            const r = i / pointsPerClass;
            const t = 1.75 * i / pointsPerClass * 2 * Math.PI + Math.PI + Math.random() * 0.2;
            inputs.push([r * Math.cos(t), r * Math.sin(t)]);
            targets.push(1);
        }

        return { inputs, targets };
    },

    gaussian: (n = 200) => {
        const inputs = [];
        const targets = [];
        const pointsPerClass = n / 2;

        const gaussian = () => {
            let u = 0, v = 0;
            while (u === 0) u = Math.random();
            while (v === 0) v = Math.random();
            return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
        };

        for (let i = 0; i < pointsPerClass; i++) {
            inputs.push([gaussian() * 0.3 - 0.4, gaussian() * 0.3 - 0.4]);
            targets.push(0);
        }

        for (let i = 0; i < pointsPerClass; i++) {
            inputs.push([gaussian() * 0.3 + 0.4, gaussian() * 0.3 + 0.4]);
            targets.push(1);
        }

        return { inputs, targets };
    }
};

// Export for use
window.NeuralNetwork = NeuralNetwork;
window.Datasets = Datasets;
