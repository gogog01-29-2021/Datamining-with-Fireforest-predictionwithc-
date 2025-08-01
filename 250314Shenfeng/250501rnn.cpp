#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <map>
#include <chrono>
#include <iomanip>
#include <Eigen/Dense>
#include <torch/torch.h>
#include <filesystem>

// Helper: Parse CSV (Date, Count)
struct Record {
    std::tm date;
    double count;
};

std::vector<Record> load_csv(const std::string& filename, int date_col_idx = 0, int count_col_idx = 1) {
    std::vector<Record> data;
    std::ifstream file(filename);
    std::string line;
    std::getline(file, line); // header
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        std::vector<std::string> cells;
        while (std::getline(ss, cell, ',')) cells.push_back(cell);
        if (cells.size() <= std::max(date_col_idx, count_col_idx)) continue;
        std::string date_str = cells[date_col_idx];
        std::string count_str = cells[count_col_idx];
        std::tm tm = {};
        if (date_str.find(":") != std::string::npos) {
            std::istringstream ss(date_str);
            ss >> std::get_time(&tm, "%d/%m/%Y %H:%M");
        } else {
            std::istringstream ss(date_str);
            ss >> std::get_time(&tm, "%d/%m/%Y");
        }
        if (!date_str.empty() && !count_str.empty()) {
            data.push_back({tm, std::stod(count_str)});
        }
    }
    return data;
}

// Convert tm to YYYY-MM-DD string
std::string tm_to_string(const std::tm& tm) {
    char buf[16];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d", &tm);
    return std::string(buf);
}

// Aggregate to daily counts
std::map<std::string, double> aggregate_daily(const std::vector<Record>& data) {
    std::map<std::string, double> daily_counts;
    for (const auto& r : data) {
        std::string day = tm_to_string(r.date);
        daily_counts[day] += 1.0;
    }
    return daily_counts;
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
    RNNNet(int input_size, int hidden_size, int output_size, int num_layers)
        : rnn(torch::nn::RNNOptions(input_size, hidden_size).num_layers(num_layers).batch_first(true)),
          fc(hidden_size, output_size) {
        register_module("rnn", rnn);
        register_module("fc", fc);
    }
    torch::Tensor forward(torch::Tensor x) {
        auto out_tuple = rnn->forward(x);
        auto out = std::get<0>(out_tuple);
        out = fc->forward(out.index({torch::indexing::Slice(), -1, torch::indexing::Slice()}));
        return out;
    }
};

int main() {
    // Dataset group (example: group1)
    std::vector<std::string> files = {
        "250314Shenfeng/datasets/dfb-fire-2023-opendata.csv",
        "250314Shenfeng/datasets/df-opendata-2016-to-2017-with-stn-area.csv",
        "250314Shenfeng/datasets/df-opendata-2020-to-2022-with-stn-area.csv",
        "250314Shenfeng/datasets/df-opendata-2018-to-2019-with-stn-area.csv"
    };
    std::vector<Record> all_data;
    for (const auto& file : files) {
        auto data = load_csv(file);
        all_data.insert(all_data.end(), data.begin(), data.end());
    }
    std::sort(all_data.begin(), all_data.end(), [](const Record& a, const Record& b) {
        return std::mktime(const_cast<std::tm*>(&a.date)) < std::mktime(const_cast<std::tm*>(&b.date));
    });
    auto daily_counts_map = aggregate_daily(all_data);
    std::vector<std::string> dates;
    std::vector<double> counts;
    for (const auto& kv : daily_counts_map) {
        dates.push_back(kv.first);
        counts.push_back(kv.second);
    }
    std::cout << "Start Date: " << dates.front() << std::endl;
    std::cout << "End Date: " << dates.back() << std::endl;
    std::cout << "Total Days: " << dates.size() << std::endl;
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
        return torch::tensor(y).to(torch::kFloat32);
    };
    auto X_train_tensor = vec2tensor(X_train);
    auto y_train_tensor = vec1tensor(y_train);
    auto X_test_tensor = vec2tensor(X_test);
    auto y_test_tensor = vec1tensor(y_test);
    // Model
    int input_size = 1, hidden_size = 50, output_size = 1, num_layers = 2;
    RNNNet model(input_size, hidden_size, output_size, num_layers);
    model.train();
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(0.001));
    int epochs = 20;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        auto output = model.forward(X_train_tensor);
        auto loss = torch::mse_loss(output.squeeze(), y_train_tensor);
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
    std::vector<double> y_train_vec(y_train_tensor.data_ptr<float>(), y_train_tensor.data_ptr<float>() + y_train_tensor.size(0));
    std::vector<double> test_pred_vec(test_pred.data_ptr<float>(), test_pred.data_ptr<float>() + test_pred.numel());
    std::vector<double> y_test_vec(y_test_tensor.data_ptr<float>(), y_test_tensor.data_ptr<float>() + y_test_tensor.size(0));
    auto train_pred_actual = scaler.inverse_transform(train_pred_vec);
    auto y_train_actual = scaler.inverse_transform(y_train_vec);
    auto test_pred_actual = scaler.inverse_transform(test_pred_vec);
    auto y_test_actual = scaler.inverse_transform(y_test_vec);
    // Output results to CSV
    std::ofstream out("250501rnn_train_results.csv");
    out << "Actual,Predicted\n";
    for (size_t i = 0; i < train_pred_actual.size(); ++i)
        out << y_train_actual[i] << "," << train_pred_actual[i] << "\n";
    out.close();
    std::ofstream out2("250501rnn_test_results.csv");
    out2 << "Actual,Predicted\n";
    for (size_t i = 0; i < test_pred_actual.size(); ++i)
        out2 << y_test_actual[i] << "," << test_pred_actual[i] << "\n";
    out2.close();
    // Forecast next 30 days
    std::vector<double> last_sequence(scaled.end() - time_step, scaled.end());
    auto last_seq_tensor = torch::tensor(last_sequence).unsqueeze(0).unsqueeze(-1).to(torch::kFloat32);
    std::vector<double> future_predictions;
    for (int i = 0; i < 30; ++i) {
        auto next_pred = model.forward(last_seq_tensor).item<float>();
        future_predictions.push_back(next_pred);
        // Update sequence
        std::vector<double> seq(last_sequence.begin() + 1, last_sequence.end());
        seq.push_back(next_pred);
        last_seq_tensor = torch::tensor(seq).unsqueeze(0).unsqueeze(-1).to(torch::kFloat32);
        last_sequence = seq;
    }
    auto future_pred_actual = scaler.inverse_transform(future_predictions);
    // Output future predictions to CSV
    std::ofstream out3("250501rnn_future_results.csv");
    out3 << "Day,Predicted\n";
    for (size_t i = 0; i < future_pred_actual.size(); ++i)
        out3 << i+1 << "," << future_pred_actual[i] << "\n";
    out3.close();
    // Metrics
    double sse = 0, mae = 0;
    for (size_t i = 0; i < test_pred_actual.size(); ++i) {
        double err = y_test_actual[i] - test_pred_actual[i];
        sse += err * err;
        mae += std::abs(err);
    }
    double mse = sse / test_pred_actual.size();
    double rmse = std::sqrt(mse);
    mae /= test_pred_actual.size();
    std::cout << "SSE: " << sse << std::endl;
    std::cout << "RMSE: " << rmse << std::endl;
    std::cout << "MAE: " << mae << std::endl;
    // Save model
    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);
    std::ostringstream oss;
    oss << "rnn_model_" << std::put_time(&tm, "%Y%m%d_%H%M%S") << ".pth";
    torch::save(model, oss.str());
    std::cout << "Model saved as " << oss.str() << std::endl;
    return 0;
}
