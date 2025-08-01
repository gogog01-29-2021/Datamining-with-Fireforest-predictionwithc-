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

// Helper: Parse CSV (Date, Count)
struct Record {
    std::tm date;
    double count;
};

std::vector<Record> load_csv(const std::string& filename) {
    std::vector<Record> data;
    std::ifstream file(filename);
    std::string line;
    std::getline(file, line); // skip header
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string date_str, count_str;
        std::getline(ss, date_str, ',');
        std::getline(ss, count_str, ',');
        std::tm tm = {};
        std::istringstream ssdate(date_str);
        ssdate >> std::get_time(&tm, "%d/%m/%Y");
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
    std::string file_path = "250314Shenfeng/combined/group_1_combined.csv";
    auto data = load_csv(file_path);
    std::sort(data.begin(), data.end(), [](const Record& a, const Record& b) {
        return std::mktime(const_cast<std::tm*>(&a.date)) < std::mktime(const_cast<std::tm*>(&b.date));
    });
    auto daily_counts_map = aggregate_daily(data);
    std::vector<std::string> dates;
    std::vector<double> counts;
    for (const auto& kv : daily_counts_map) {
        dates.push_back(kv.first);
        counts.push_back(kv.second);
    }
    // Scaling
    MinMaxScaler scaler;
    scaler.fit(counts);
    auto scaled = scaler.transform(counts);
    // Prepare last sequence
    int time_step = 10;
    std::vector<double> last_sequence(scaled.end() - time_step, scaled.end());
    auto last_seq_tensor = torch::tensor(last_sequence).unsqueeze(0).unsqueeze(-1).to(torch::kFloat32);
    // Load model
    int input_size = 1, hidden_size = 50, output_size = 1, num_layers = 2;
    RNNNet model(input_size, hidden_size, output_size, num_layers);
    torch::load(model, "rnn_model_20250430_0835.pth"); // change to your filename
    model.eval();
    // Predict next 30 days
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
    std::ofstream out("runmodel_future_results.csv");
    out << "Day,Predicted\n";
    for (size_t i = 0; i < future_pred_actual.size(); ++i)
        out << i+1 << "," << future_pred_actual[i] << "\n";
    out.close();
    std::cout << "Future predictions written to runmodel_future_results.csv" << std::endl;
    return 0;
}
