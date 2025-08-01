#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <map>
#include <cmath>
#include <iomanip>
#include <Eigen/Dense>
#include <torch/torch.h>

// Helper: Parse CSV (Date, Count)
struct Record {
    std::string date;
    double count;
};

std::vector<Record> load_csv(const std::string& filename) {
    std::vector<Record> data;
    std::ifstream file(filename);
    std::string line;
    std::getline(file, line); // skip header
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string date, count_str;
        std::getline(ss, date, ',');
        std::getline(ss, count_str, ',');
        if (!date.empty() && !count_str.empty()) {
            data.push_back({date, std::stod(count_str)});
        }
    }
    return data;
}

// MinMaxScaler
struct MinMaxScaler {
    double min, max;
    MinMaxScaler() : min(0), max(1) {}
    void fit(const std::vector<double>& v) {
        min = *std::min_element(v.begin(), v.end());
        max = *std::max_element(v.begin(), v.end());
    }
    std::vector<double> transform(const std::vector<double>& v) {
        std::vector<double> res;
        for (auto x : v) res.push_back((x - min) / (max - min));
        return res;
    }
    std::vector<double> inverse_transform(const std::vector<double>& v) {
        std::vector<double> res;
        for (auto x : v) res.push_back(x * (max - min) + min);
        return res;
    }
};

// Create dataset for LSTM
void create_dataset(const std::vector<double>& data, int time_step, std::vector<std::vector<double>>& X, std::vector<double>& y) {
    for (size_t i = 0; i + time_step < data.size(); ++i) {
        X.push_back(std::vector<double>(data.begin() + i, data.begin() + i + time_step));
        y.push_back(data[i + time_step]);
    }
}

// LSTM Model
struct LSTMNet : torch::nn::Module {
    torch::nn::LSTM lstm1, lstm2;
    torch::nn::Linear fc1, fc2;
    LSTMNet(int input_size, int hidden_size, int time_step)
        : lstm1(torch::nn::LSTMOptions(input_size, hidden_size).batch_first(true)),
          lstm2(torch::nn::LSTMOptions(hidden_size, hidden_size).batch_first(true)),
          fc1(hidden_size, 25), fc2(25, 1) {
        register_module("lstm1", lstm1);
        register_module("lstm2", lstm2);
        register_module("fc1", fc1);
        register_module("fc2", fc2);
    }
    torch::Tensor forward(torch::Tensor x) {
        auto out1 = lstm1->forward(x);
        auto out2 = lstm2->forward(std::get<0>(out1));
        auto out = torch::relu(fc1->forward(out2.output[:, -1, :]));
        out = fc2->forward(out);
        return out;
    }
};

int main() {
    std::string file_path = "250314Shenfeng/2023-open-data-dfb-ambulance.csv";
    // For demo, expects a CSV with Date,Count columns (pre-aggregated per day)
    auto data = load_csv(file_path);
    std::vector<double> counts;
    for (const auto& r : data) counts.push_back(r.count);

    // Scaling
    MinMaxScaler scaler;
    scaler.fit(counts);
    auto scaled = scaler.transform(counts);

    // Train/test split
    int train_size = static_cast<int>(scaled.size() * 0.8);
    std::vector<double> train_data(scaled.begin(), scaled.begin() + train_size);
    std::vector<double> test_data(scaled.begin() + train_size, scaled.end());

    // Create datasets
    int time_step = 10;
    std::vector<std::vector<double>> X_train, X_test;
    std::vector<double> y_train, y_test;
    create_dataset(train_data, time_step, X_train, y_train);
    create_dataset(test_data, time_step, X_test, y_test);

    // Convert to torch tensors
    auto vec2tensor = [](const std::vector<std::vector<double>>& X) {
        return torch::tensor(X).unsqueeze(-1).to(torch::kFloat32);
    };
    auto vec1tensor = [](const std::vector<double>& y) {
        return torch::tensor(y).unsqueeze(-1).to(torch::kFloat32);
    };
    auto X_train_tensor = vec2tensor(X_train);
    auto y_train_tensor = vec1tensor(y_train);
    auto X_test_tensor = vec2tensor(X_test);
    auto y_test_tensor = vec1tensor(y_test);

    // Model
    int input_size = 1, hidden_size = 50;
    LSTMNet model(input_size, hidden_size, time_step);
    model.train();
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(0.001));
    int epochs = 10; // For demo, use more for real training
    for (int epoch = 0; epoch < epochs; ++epoch) {
        auto output = model.forward(X_train_tensor);
        auto loss = torch::mse_loss(output, y_train_tensor);
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();
        std::cout << "Epoch " << epoch+1 << ", Loss: " << loss.item<float>() << std::endl;
    }

    // Predict
    model.eval();
    auto train_pred = model.forward(X_train_tensor).detach().squeeze().to(torch::kCPU);
    auto test_pred = model.forward(X_test_tensor).detach().squeeze().to(torch::kCPU);

    // Inverse transform
    std::vector<double> train_pred_vec(train_pred.data_ptr<float>(), train_pred.data_ptr<float>() + train_pred.numel());
    std::vector<double> y_train_vec(y_train_tensor.squeeze().data_ptr<float>(), y_train_tensor.squeeze().data_ptr<float>() + y_train_tensor.size(0));
    std::vector<double> test_pred_vec(test_pred.data_ptr<float>(), test_pred.data_ptr<float>() + test_pred.numel());
    std::vector<double> y_test_vec(y_test_tensor.squeeze().data_ptr<float>(), y_test_tensor.squeeze().data_ptr<float>() + y_test_tensor.size(0));
    auto train_pred_actual = scaler.inverse_transform(train_pred_vec);
    auto y_train_actual = scaler.inverse_transform(y_train_vec);
    auto test_pred_actual = scaler.inverse_transform(test_pred_vec);
    auto y_test_actual = scaler.inverse_transform(y_test_vec);

    // Output results (for plotting in Python or Excel)
    std::ofstream out("250322tf_train_results.csv");
    out << "Actual,Predicted\n";
    for (size_t i = 0; i < train_pred_actual.size(); ++i)
        out << y_train_actual[i] << "," << train_pred_actual[i] << "\n";
    out.close();
    std::ofstream out2("250322tf_test_results.csv");
    out2 << "Actual,Predicted\n";
    for (size_t i = 0; i < test_pred_actual.size(); ++i)
        out2 << y_test_actual[i] << "," << test_pred_actual[i] << "\n";
    out2.close();
    std::cout << "Results written to 250322tf_train_results.csv and 250322tf_test_results.csv" << std::endl;
    return 0;
}
