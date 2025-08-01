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

// Simple ARIMA (AR only, for demo)
std::vector<double> ar_predict(const std::vector<double>& train, int p, int steps) {
    std::vector<double> preds;
    for (int i = 0; i < steps; ++i) {
        double pred = 0.0;
        for (int j = 0; j < p; ++j) {
            if (train.size() + i - j - 1 >= 0)
                pred += train[train.size() + i - j - 1] / p;
        }
        preds.push_back(pred);
    }
    return preds;
}

double rmse(const std::vector<double>& actual, const std::vector<double>& pred) {
    double sum = 0.0;
    for (size_t i = 0; i < actual.size(); ++i) sum += std::pow(actual[i]-pred[i],2);
    return std::sqrt(sum/actual.size());
}

int main() {
    std::string file_path = "250314Shenfeng/combined/group_1_combined.csv";
    auto data = load_csv(file_path);
    std::vector<double> counts;
    for (const auto& r : data) counts.push_back(r.count);
    // Train/test split
    int train_size = static_cast<int>(counts.size() * 0.8);
    std::vector<double> train_data(counts.begin(), counts.begin() + train_size);
    std::vector<double> test_data(counts.begin() + train_size, counts.end());
    // ARIMA (AR only, p=3 for demo)
    int p = 3;
    auto test_pred = ar_predict(train_data, p, test_data.size());
    double test_rmse = rmse(test_data, test_pred);
    std::cout << "Test RMSE: " << test_rmse << std::endl;
    // Output results (for plotting)
    std::ofstream out("250328ARIMAtogroup_test_results.csv");
    out << "Actual,Predicted\n";
    for (size_t i = 0; i < test_pred.size(); ++i)
        out << test_data[i] << "," << test_pred[i] << "\n";
    out.close();
    std::cout << "Results written to 250328ARIMAtogroup_test_results.csv" << std::endl;
    return 0;
}
