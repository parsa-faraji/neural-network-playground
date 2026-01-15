/**
 * Neural Network Playground Application
 */

(function() {
    'use strict';

    // State
    let nn = null;
    let dataset = null;
    let isTraining = false;
    let epoch = 0;
    let lossHistory = [];
    let animationId = null;

    // Settings
    let settings = {
        dataset: 'circle',
        learningRate: 0.01,
        activation: 'relu',
        hiddenLayers: 2,
        neuronsPerLayer: 4
    };

    // DOM Elements
    const mainCanvas = document.getElementById('mainCanvas');
    const networkCanvas = document.getElementById('networkCanvas');
    const lossChart = document.getElementById('lossChart');
    const mainCtx = mainCanvas.getContext('2d');
    const networkCtx = networkCanvas.getContext('2d');
    const lossCtx = lossChart.getContext('2d');

    // Initialize
    function init() {
        setupCanvases();
        setupEventListeners();
        resetNetwork();
    }

    function setupCanvases() {
        // Main canvas
        const mainRect = mainCanvas.parentElement.getBoundingClientRect();
        mainCanvas.width = mainRect.width;
        mainCanvas.height = 400;

        // Network canvas
        networkCanvas.width = 300;
        networkCanvas.height = 400;

        // Loss chart
        const lossRect = lossChart.parentElement.getBoundingClientRect();
        lossChart.width = lossRect.width - 50;
        lossChart.height = 150;
    }

    function setupEventListeners() {
        document.getElementById('dataset').addEventListener('change', (e) => {
            settings.dataset = e.target.value;
            resetNetwork();
        });

        document.getElementById('learningRate').addEventListener('change', (e) => {
            settings.learningRate = parseFloat(e.target.value);
            if (nn) nn.learningRate = settings.learningRate;
        });

        document.getElementById('activation').addEventListener('change', (e) => {
            settings.activation = e.target.value;
            resetNetwork();
        });

        document.getElementById('addLayer').addEventListener('click', () => {
            if (settings.hiddenLayers < 6) {
                settings.hiddenLayers++;
                document.getElementById('layerCount').textContent = settings.hiddenLayers;
                resetNetwork();
            }
        });

        document.getElementById('removeLayer').addEventListener('click', () => {
            if (settings.hiddenLayers > 1) {
                settings.hiddenLayers--;
                document.getElementById('layerCount').textContent = settings.hiddenLayers;
                resetNetwork();
            }
        });

        document.getElementById('addNeuron').addEventListener('click', () => {
            if (settings.neuronsPerLayer < 8) {
                settings.neuronsPerLayer++;
                document.getElementById('neuronCount').textContent = settings.neuronsPerLayer;
                resetNetwork();
            }
        });

        document.getElementById('removeNeuron').addEventListener('click', () => {
            if (settings.neuronsPerLayer > 1) {
                settings.neuronsPerLayer--;
                document.getElementById('neuronCount').textContent = settings.neuronsPerLayer;
                resetNetwork();
            }
        });

        document.getElementById('playPause').addEventListener('click', toggleTraining);
        document.getElementById('reset').addEventListener('click', resetNetwork);

        window.addEventListener('resize', () => {
            setupCanvases();
            draw();
        });
    }

    function resetNetwork() {
        // Stop training
        if (isTraining) {
            toggleTraining();
        }

        // Generate dataset
        dataset = Datasets[settings.dataset](300);

        // Create network architecture
        const layers = [2]; // Input layer (x, y)
        for (let i = 0; i < settings.hiddenLayers; i++) {
            layers.push(settings.neuronsPerLayer);
        }
        layers.push(1); // Output layer

        // Initialize network
        nn = new NeuralNetwork(layers, settings.activation, settings.learningRate);

        // Reset state
        epoch = 0;
        lossHistory = [];

        // Update display
        document.getElementById('epochCount').textContent = '0';
        document.getElementById('lossValue').textContent = '-';
        document.getElementById('accuracyValue').textContent = '-';

        draw();
    }

    function toggleTraining() {
        isTraining = !isTraining;
        const btn = document.getElementById('playPause');

        if (isTraining) {
            btn.innerHTML = '<span class="play-icon">⏸</span> Pause';
            btn.classList.add('running');
            train();
        } else {
            btn.innerHTML = '<span class="play-icon">▶</span> Train';
            btn.classList.remove('running');
            if (animationId) {
                cancelAnimationFrame(animationId);
            }
        }
    }

    function train() {
        if (!isTraining) return;

        // Train for multiple iterations per frame
        for (let i = 0; i < 10; i++) {
            const result = nn.train(dataset.inputs, dataset.targets);
            epoch++;

            if (epoch % 10 === 0) {
                lossHistory.push(result.loss);
                if (lossHistory.length > 100) {
                    lossHistory.shift();
                }

                document.getElementById('epochCount').textContent = epoch;
                document.getElementById('lossValue').textContent = result.loss.toFixed(4);
                document.getElementById('accuracyValue').textContent = (result.accuracy * 100).toFixed(1) + '%';
            }
        }

        draw();
        animationId = requestAnimationFrame(train);
    }

    function draw() {
        drawDecisionBoundary();
        drawNetwork();
        drawLossChart();
    }

    function drawDecisionBoundary() {
        const width = mainCanvas.width;
        const height = mainCanvas.height;
        const resolution = 50;
        const cellWidth = width / resolution;
        const cellHeight = height / resolution;

        mainCtx.clearRect(0, 0, width, height);

        // Draw decision boundary
        for (let i = 0; i < resolution; i++) {
            for (let j = 0; j < resolution; j++) {
                const x = (i / resolution) * 2 - 1;
                const y = (j / resolution) * 2 - 1;
                const prediction = nn.predict([x, y]);

                // Color based on prediction
                const r = Math.floor((1 - prediction) * 245 + prediction * 0);
                const g = Math.floor((1 - prediction) * 87 + prediction * 242);
                const b = Math.floor((1 - prediction) * 108 + prediction * 254);

                mainCtx.fillStyle = `rgba(${r}, ${g}, ${b}, 0.6)`;
                mainCtx.fillRect(i * cellWidth, j * cellHeight, cellWidth + 1, cellHeight + 1);
            }
        }

        // Draw data points
        for (let i = 0; i < dataset.inputs.length; i++) {
            const x = (dataset.inputs[i][0] + 1) / 2 * width;
            const y = (dataset.inputs[i][1] + 1) / 2 * height;
            const label = dataset.targets[i];

            mainCtx.beginPath();
            mainCtx.arc(x, y, 5, 0, Math.PI * 2);
            mainCtx.fillStyle = label === 1 ? '#00f2fe' : '#f5576c';
            mainCtx.fill();
            mainCtx.strokeStyle = 'rgba(255, 255, 255, 0.8)';
            mainCtx.lineWidth = 1.5;
            mainCtx.stroke();
        }
    }

    function drawNetwork() {
        const width = networkCanvas.width;
        const height = networkCanvas.height;
        const layers = nn.layerSizes;
        const weights = nn.getWeights();

        networkCtx.clearRect(0, 0, width, height);

        const layerSpacing = width / (layers.length + 1);
        const nodePositions = [];

        // Calculate node positions
        for (let l = 0; l < layers.length; l++) {
            const layerPositions = [];
            const nodeSpacing = height / (layers[l] + 1);

            for (let n = 0; n < layers[l]; n++) {
                layerPositions.push({
                    x: layerSpacing * (l + 1),
                    y: nodeSpacing * (n + 1)
                });
            }
            nodePositions.push(layerPositions);
        }

        // Draw connections
        for (let l = 0; l < weights.length; l++) {
            for (let i = 0; i < weights[l].length; i++) {
                for (let j = 0; j < weights[l][i].length; j++) {
                    const weight = weights[l][i][j];
                    const absWeight = Math.min(Math.abs(weight), 2);

                    networkCtx.beginPath();
                    networkCtx.moveTo(nodePositions[l][i].x, nodePositions[l][i].y);
                    networkCtx.lineTo(nodePositions[l + 1][j].x, nodePositions[l + 1][j].y);

                    if (weight > 0) {
                        networkCtx.strokeStyle = `rgba(0, 242, 254, ${absWeight / 2})`;
                    } else {
                        networkCtx.strokeStyle = `rgba(245, 87, 108, ${absWeight / 2})`;
                    }
                    networkCtx.lineWidth = absWeight * 1.5;
                    networkCtx.stroke();
                }
            }
        }

        // Draw nodes
        for (let l = 0; l < layers.length; l++) {
            for (let n = 0; n < layers[l]; n++) {
                const pos = nodePositions[l][n];

                // Gradient fill
                const gradient = networkCtx.createRadialGradient(
                    pos.x, pos.y, 0,
                    pos.x, pos.y, 12
                );

                if (l === 0) {
                    gradient.addColorStop(0, '#4facfe');
                    gradient.addColorStop(1, '#667eea');
                } else if (l === layers.length - 1) {
                    gradient.addColorStop(0, '#f093fb');
                    gradient.addColorStop(1, '#f5576c');
                } else {
                    gradient.addColorStop(0, '#667eea');
                    gradient.addColorStop(1, '#764ba2');
                }

                networkCtx.beginPath();
                networkCtx.arc(pos.x, pos.y, 12, 0, Math.PI * 2);
                networkCtx.fillStyle = gradient;
                networkCtx.fill();
                networkCtx.strokeStyle = 'rgba(255, 255, 255, 0.5)';
                networkCtx.lineWidth = 2;
                networkCtx.stroke();
            }
        }

        // Draw layer labels
        networkCtx.fillStyle = 'rgba(255, 255, 255, 0.5)';
        networkCtx.font = '11px Inter';
        networkCtx.textAlign = 'center';

        networkCtx.fillText('Input', layerSpacing, height - 10);
        for (let l = 1; l < layers.length - 1; l++) {
            networkCtx.fillText(`Hidden ${l}`, layerSpacing * (l + 1), height - 10);
        }
        networkCtx.fillText('Output', layerSpacing * layers.length, height - 10);
    }

    function drawLossChart() {
        const width = lossChart.width;
        const height = lossChart.height;

        lossCtx.clearRect(0, 0, width, height);

        if (lossHistory.length < 2) return;

        // Find min/max
        const maxLoss = Math.max(...lossHistory, 0.1);

        // Draw grid
        lossCtx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
        lossCtx.lineWidth = 1;
        for (let i = 0; i <= 4; i++) {
            const y = (i / 4) * height;
            lossCtx.beginPath();
            lossCtx.moveTo(0, y);
            lossCtx.lineTo(width, y);
            lossCtx.stroke();
        }

        // Draw loss line
        const gradient = lossCtx.createLinearGradient(0, 0, width, 0);
        gradient.addColorStop(0, '#667eea');
        gradient.addColorStop(0.5, '#f093fb');
        gradient.addColorStop(1, '#f5576c');

        lossCtx.beginPath();
        lossCtx.strokeStyle = gradient;
        lossCtx.lineWidth = 2;

        for (let i = 0; i < lossHistory.length; i++) {
            const x = (i / (lossHistory.length - 1)) * width;
            const y = height - (lossHistory[i] / maxLoss) * height * 0.9;

            if (i === 0) {
                lossCtx.moveTo(x, y);
            } else {
                lossCtx.lineTo(x, y);
            }
        }
        lossCtx.stroke();

        // Draw area under curve
        lossCtx.lineTo(width, height);
        lossCtx.lineTo(0, height);
        lossCtx.closePath();

        const areaGradient = lossCtx.createLinearGradient(0, 0, 0, height);
        areaGradient.addColorStop(0, 'rgba(102, 126, 234, 0.3)');
        areaGradient.addColorStop(1, 'rgba(102, 126, 234, 0)');
        lossCtx.fillStyle = areaGradient;
        lossCtx.fill();
    }

    // Start
    init();
})();
