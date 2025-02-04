#include <iostream>
#include <bitset>
#include <vector>
#include <algorithm>
#include <numeric>

int main() {
    const int size = 8; // Bitset size
    std::bitset<size> bits; // Initialize bitset (all bits are 0)

    // Set the leading bits to 1
    for (int i = size - 1; i >= size / 2; --i) {
        bits.set(i);
    }

    // Print bitset
    std::cout << "Bitset: " << bits << std::endl;

    // Use a vector to set bits to 1
    std::vector<int> vec(size, 0);
    std::fill(vec.begin() + size / 2, vec.end(), 1);
    
    // Print vector
    std::cout << "Vector: ";
    for (const auto& bit : vec) {
        std::cout << bit;
    }
    std::cout << std::endl;

    return 0;
}

#include <vector>

// Linear Regression function
std::pair<double, double> linearRegression(const std::vector<double>& x, const std::vector<double>& y) {
    int n = x.size();
    double sumX = std::accumulate(x.begin(), x.end(), 0.0);
    double sumY = std::accumulate(y.begin(), y.end(), 0.0);
    double sumXY = std::inner_product(x.begin(), x.end(), y.begin(), 0.0);
    double sumX2 = std::inner_product(x.begin(), x.end(), x.begin(), 0.0);

    double slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
    double intercept = (sumY - slope * sumX) / n;

    return {slope, intercept};
}

#include <vector>

int main() {
    // Existing code

    // Example data for linear regression
    std::vector<double> x = {1, 2, 3, 4, 5};
    std::vector<double> y = {2, 3, 5, 7, 11};

    // Compute linear regression
    auto [slope, intercept] = linearRegression(x, y);

    // Print results
    std::cout << "Slope: " << slope << std::endl;
    std::cout << "Intercept: " << intercept << std::endl;

    return 0;
}

#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>

// Activation function (e.g., sigmoid function)
double sigmoid(double x) {
    return 1 / (1 + std::exp(-x));
}

// Neural network class for linear regression
class LinearRegressionNN {
public:
    LinearRegressionNN(int inputSize) : weights(inputSize, 0.0), bias(0.0) {}

    // Prediction function
    double predict(const std::vector<double>& x) {
        double linearSum = std::inner_product(x.begin(), x.end(), weights.begin(), bias);
        return sigmoid(linearSum);
    }

    // Training function
    void train(const std::vector<std::vector<double>>& x, const std::vector<double>& y, double learningRate, int epochs) {
        int n = x.size();
        for (int epoch = 0; epoch < epochs; ++epoch) {
            for (int i = 0; i < n; ++i) {
                double prediction = predict(x[i]);
                double error = y[i] - prediction;

                // Update weights and bias
                for (int j = 0; j < weights.size(); ++j) {
                    weights[j] += learningRate * error * x[i][j];
                }
                bias += learningRate * error;
            }
        }
    }

private:
    std::vector<double> weights;
    double bias;
};

int main() {
    // Example training data
    std::vector<std::vector<double>> x = {{1}, {2}, {3}, {4}, {5}};
    std::vector<double> y = {2, 3, 5, 7, 11};

    // Initialize neural network
    LinearRegressionNN nn(1);

    // Train
    nn.train(x, y, 0.01, 1000);

    // Predict
    std::vector<double> testInput = {6};
    double prediction = nn.predict(testInput);

    // Print result
    std::cout << "Prediction for input 6: " << prediction << std::endl;

    return 0;
}

#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <random>

// Activation functions
inline double sigmoid(double x) {
    return 1 / (1 + std::exp(-x));
}

inline double relu(double x) {
    return std::max(0.0, x);
}

// Deep Neural Network class
class DeepNeuralNetwork {
public:
    DeepNeuralNetwork(int inputSize, int hiddenSize, int outputSize)
        : weights1(inputSize, std::vector<double>(hiddenSize)),
          weights2(hiddenSize, std::vector<double>(outputSize)),
          bias1(hiddenSize, 0.0),
          bias2(outputSize, 0.0) {
        initializeWeights();
    }

    // Initialize weights randomly
    void initializeWeights() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(-1.0, 1.0);
        
        for (auto& row : weights1)
            for (auto& w : row)
                w = dis(gen);
        
        for (auto& row : weights2)
            for (auto& w : row)
                w = dis(gen);
    }

    // Forward propagation
    std::vector<double> forward(const std::vector<double>& input) {
        hiddenLayer.resize(weights1[0].size());
        outputLayer.resize(weights2[0].size());

        for (size_t j = 0; j < hiddenLayer.size(); ++j) {
            hiddenLayer[j] = bias1[j];
            for (size_t i = 0; i < input.size(); ++i) {
                hiddenLayer[j] += input[i] * weights1[i][j];
            }
            hiddenLayer[j] = relu(hiddenLayer[j]);
        }

        for (size_t k = 0; k < outputLayer.size(); ++k) {
            outputLayer[k] = bias2[k];
            for (size_t j = 0; j < hiddenLayer.size(); ++j) {
                outputLayer[k] += hiddenLayer[j] * weights2[j][k];
            }
            outputLayer[k] = sigmoid(outputLayer[k]);
        }
        
        return outputLayer;
    }

private:
    std::vector<std::vector<double>> weights1, weights2;
    std::vector<double> bias1, bias2;
    std::vector<double> hiddenLayer, outputLayer;
};

int main() {
    DeepNeuralNetwork dnn(2, 5, 1);
    std::vector<double> input = {1.0, 2.0};
    std::vector<double> output = dnn.forward(input);
    
    std::cout << "Prediction: " << output[0] << std::endl;
    return 0;
}
