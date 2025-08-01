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

// Create sequences for RNN
void create_sequences(const std::vector<double>& data, int time_step, std::vector<std::vector<double>>& X, std::vector<double>& y) {
    for (size_t i = 0; i + time_step < data.size(); ++i) {
        X.push_back(std::vector<double>(data.begin() + i, data.begin() + i + time_step));
        y.push_back(data[i + time_step]);
    }
}

// RNN Model
struct RNNNet : torch::nn::Module {
    torch::nn::RNN rnn;
    torch::nn::Linear fc;
    RNNNet(int input_size, int hidden_size, int num_layers)
        : rnn(torch::nn::RNNOptions(input_size, hidden_size).num_layers(num_layers).batch_first(true)),
          fc(hidden_size, 1) {
        register_module("rnn", rnn);
        register_module("fc", fc);
    }
    torch::Tensor forward(torch::Tensor x) {
        auto out = rnn->forward(x);
        auto last = std::get<0>(out).index({torch::indexing::Slice(), -1, torch::indexing::Slice()});
        return fc->forward(last);
    }
};

int main() {
    std::string file_path = "250314Shenfeng/combined/group_7_combined.csv";
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
    // Create sequences
    int time_step = 10;
    std::vector<std::vector<double>> X_train, X_test;
    std::vector<double> y_train, y_test;
    create_sequences(train_data, time_step, X_train, y_train);
    create_sequences(test_data, time_step, X_test, y_test);
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
    int input_size = 1, hidden_size = 50, num_layers = 2;
    RNNNet model(input_size, hidden_size, num_layers);
    model.train();
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(0.001));
    int epochs = 20;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        auto output = model.forward(X_train_tensor);
        auto loss = torch::mse_loss(output.squeeze(), y_train_tensor.squeeze());
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
    std::ofstream out("250328rnngroup6_train_results.csv");
    out << "Actual,Predicted\n";
    for (size_t i = 0; i < train_pred_actual.size(); ++i)
        out << y_train_actual[i] << "," << train_pred_actual[i] << "\n";
    out.close();
    std::ofstream out2("250328rnngroup6_test_results.csv");
    out2 << "Actual,Predicted\n";
    for (size_t i = 0; i < test_pred_actual.size(); ++i)
        out2 << y_test_actual[i] << "," << test_pred_actual[i] << "\n";
    out2.close();
    // Forecast future values (next 30 days)
    std::vector<double> last_seq(scaled.end()-time_step, scaled.end());
    std::vector<double> future_preds;
    torch::NoGradGuard no_grad;
    auto last_tensor = torch::tensor(last_seq).unsqueeze(0).unsqueeze(-1).to(torch::kFloat32);
    for (int i = 0; i < 30; ++i) {
        auto next_pred = model.forward(last_tensor).item<float>();
        future_preds.push_back(next_pred);
        // Shift sequence and append next_pred
        std::vector<double> new_seq(last_seq.begin()+1, last_seq.end());
        new_seq.push_back(next_pred);
        last_seq = new_seq;
        last_tensor = torch::tensor(last_seq).unsqueeze(0).unsqueeze(-1).to(torch::kFloat32);
    }
    auto future_actual = scaler.inverse_transform(future_preds);
    std::ofstream out3("250328rnngroup6_future_results.csv");
    out3 << "Future_Predicted\n";
    for (size_t i = 0; i < future_actual.size(); ++i)
        out3 << future_actual[i] << "\n";
    out3.close();
    // Save model
    torch::save(model, "rnn_model_group6.pth");
    std::cout << "Results written to 250328rnngroup6_train_results.csv, 250328rnngroup6_test_results.csv, 250328rnngroup6_future_results.csv, and model saved as rnn_model_group6.pth" << std::endl;
    return 0;
}
