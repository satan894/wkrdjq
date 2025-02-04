#include <iostream>
#include <bitset>
#include <vector>
#include <algorithm>
#include <numeric>

int main() {
    const int size = 8; // 비트셋 크기
    std::bitset<size> bits; // 비트셋 초기화 (모든 비트가 0)

    // 앞자리 비트를 모두 1로 설정
    for (int i = size - 1; i >= size / 2; --i) {
        bits.set(i);
    }

    // 비트셋 출력
    std::cout << "Bitset: " << bits << std::endl;

    // 벡터를 이용하여 비트셋을 1로 설정
    std::vector<int> vec(size, 0);
    std::fill(vec.begin() + size / 2, vec.end(), 1);
    
    // 벡터 출력
    std::cout << "Vector: ";
    for (const auto& bit : vec) {
        std::cout << bit;
    }
    std::cout << std::endl;

    return 0;
}

#include <vector>

// 선형 회귀 함수
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
    // 기존 코드

    // 선형 회귀 예제 데이터
    std::vector<double> x = {1, 2, 3, 4, 5};
    std::vector<double> y = {2, 3, 5, 7, 11};

    // 선형 회귀 계산
    auto [slope, intercept] = linearRegression(x, y);

    // 결과 출력
    std::cout << "Slope: " << slope << std::endl;
    std::cout << "Intercept: " << intercept << std::endl;

    return 0;
}

#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>

// 활성화 함수 (예: 시그모이드 함수)
double sigmoid(double x) {
    return 1 / (1 + std::exp(-x));
}

// 선형 회귀를 위한 인공 신경망 클래스
class LinearRegressionNN {
public:
    LinearRegressionNN(int inputSize) : weights(inputSize, 0.0), bias(0.0) {}

    // 예측 함수
    double predict(const std::vector<double>& x) {
        double linearSum = std::inner_product(x.begin(), x.end(), weights.begin(), bias);
        return sigmoid(linearSum);
    }

    // 학습 함수
    void train(const std::vector<std::vector<double>>& x, const std::vector<double>& y, double learningRate, int epochs) {
        int n = x.size();
        for (int epoch = 0; epoch < epochs; ++epoch) {
            for (int i = 0; i < n; ++i) {
                double prediction = predict(x[i]);
                double error = y[i] - prediction;

                // 가중치와 바이어스 업데이트
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
    // 학습 데이터 (예제)
    std::vector<std::vector<double>> x = {{1}, {2}, {3}, {4}, {5}};
    std::vector<double> y = {2, 3, 5, 7, 11};

    // 인공 신경망 초기화
    LinearRegressionNN nn(1);

    // 학습
    nn.train(x, y, 0.01, 1000);

    // 예측
    std::vector<double> testInput = {6};
    double prediction = nn.predict(testInput);

    // 결과 출력
    std::cout << "Prediction for input 6: " << prediction << std::endl;

    return 0;
}

#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <random>

// 활성화 함수
inline double sigmoid(double x) {
    return 1 / (1 + std::exp(-x));
}

inline double relu(double x) {
    return std::max(0.0, x);
}

// 다층 신경망 클래스
class DeepNeuralNetwork {
public:
    DeepNeuralNetwork(int inputSize, int hiddenSize, int outputSize)
        : weights1(inputSize, std::vector<double>(hiddenSize)),
          weights2(hiddenSize, std::vector<double>(outputSize)),
          bias1(hiddenSize, 0.0),
          bias2(outputSize, 0.0) {
        initializeWeights();
    }

    // 가중치 초기화 (랜덤 값)
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

    // 순전파 (Forward Propagation)
    std::vector<double> forward(const std::vector<double>& input) {
        hiddenLayer.resize(weights1[0].size());
        outputLayer.resize(weights2[0].size());

        // 첫 번째 은닉층 계산
        for (size_t j = 0; j < hiddenLayer.size(); ++j) {
            hiddenLayer[j] = bias1[j];
            for (size_t i = 0; i < input.size(); ++i) {
                hiddenLayer[j] += input[i] * weights1[i][j];
            }
            hiddenLayer[j] = relu(hiddenLayer[j]);
        }

        // 출력층 계산
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
    DeepNeuralNetwork dnn(2, 5, 1); // 입력 2개, 은닉층 뉴런 5개, 출력 1개
    
    // 테스트 입력 데이터
    std::vector<double> input = {1.0, 2.0};
    std::vector<double> output = dnn.forward(input);
    
    std::cout << "Prediction: " << output[0] << std::endl;
    return 0;
}

    #include <iostream>
    #include <vector>
    #include <cmath>
    #include <numeric>

    // 8진법 변환 함수
    std::vector<int> toOctal(int num) {
        std::vector<int> octal;
        while (num > 0) {
            octal.push_back(num % 8);
            num /= 8;
        }
        std::reverse(octal.begin(), octal.end());
        return octal;
    }

    // 원주율을 이용한 데이터 생성
    std::vector<double> generatePiData(int size) {
        std::vector<double> piData(size);
        double pi = 3.14159265358979323846;
        for (int i = 0; i < size; ++i) {
            piData[i] = std::pow(pi, i);
        }
        return piData;
    }

    int main() {
        // 8진법 변환 예제
        int num = 123;
        std::vector<int> octal = toOctal(num);
        std::cout << "Octal representation of " << num << ": ";
        for (int digit : octal) {
            std::cout << digit;
        }
        std::cout << std::endl;

        // 원주율 데이터 생성
        int dataSize = 5;
        std::vector<double> piData = generatePiData(dataSize);

        // 선형 회귀 예제 데이터
        std::vector<double> x = {1, 2, 3, 4, 5};
        std::vector<double> y = piData;

        // 선형 회귀 계산
        auto [slope, intercept] = linearRegression(x, y);

        // 결과 출력
        std::cout << "Slope: " << slope << std::endl;
        std::cout << "Intercept: " << intercept << std::endl;

        return 0;
    }
