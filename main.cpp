#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <random>
#include <map>
#include <string>
#include <Eigen/Dense>

// Helper functions for statistics

double mean(const std::vector<double>& v) {
    return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    if (n % 2 == 0)
        return (v[n/2 - 1] + v[n/2]) / 2.0;
    else
        return v[n/2];
}

double stddev(const std::vector<double>& v, bool sample = false) {
    double m = mean(v);
    double accum = 0.0;
    for (auto val : v) accum += (val - m) * (val - m);
    return std::sqrt(accum / (v.size() - (sample ? 1 : 0)));
}

double variance(const std::vector<double>& v, bool sample = false) {
    double m = mean(v);
    double accum = 0.0;
    for (auto val : v) accum += (val - m) * (val - m);
    return accum / (v.size() - (sample ? 1 : 0));
}

double quantile(std::vector<double> v, double q) {
    std::sort(v.begin(), v.end());
    double pos = (v.size() - 1) * q;
    size_t idx = static_cast<size_t>(pos);
    double frac = pos - idx;
    if (idx + 1 < v.size())
        return v[idx] * (1 - frac) + v[idx + 1] * frac;
    else
        return v[idx];
}

double min(const std::vector<double>& v) {
    return *std::min_element(v.begin(), v.end());
}

double max(const std::vector<double>& v) {
    return *std::max_element(v.begin(), v.end());
}

// IQR
std::pair<double, double> iqr(const std::vector<double>& v) {
    double q1 = quantile(v, 0.25);
    double q3 = quantile(v, 0.75);
    return {q1, q3};
}

double scipy_iqr(const std::vector<double>& v) {
    auto [q1, q3] = iqr(v);
    return q3 - q1;
}

// Skewness and kurtosis (using Eigen for convenience)
double skewness(const std::vector<double>& v) {
    Eigen::VectorXd x = Eigen::Map<const Eigen::VectorXd>(v.data(), v.size());
    double m = x.mean();
    double s = std::sqrt((x.array() - m).square().sum() / x.size());
    return ((x.array() - m).pow(3).sum() / x.size()) / std::pow(s, 3);
}

double kurtosis(const std::vector<double>& v, bool fisher = true) {
    Eigen::VectorXd x = Eigen::Map<const Eigen::VectorXd>(v.data(), v.size());
    double m = x.mean();
    double s = std::sqrt((x.array() - m).square().sum() / x.size());
    double k = ((x.array() - m).pow(4).sum() / x.size()) / std::pow(s, 4);
    return fisher ? k - 3 : k;
}

int main() {
    std::vector<double> s = {2, 2, 4, 5, 5, 5, 8, 9, 9, 9, 12};
    std::cout << "len: " << s.size() << std::endl;
    std::cout << "mean: " << mean(s) << std::endl;
    std::cout << "median: " << median(s) << std::endl;
    std::cout << "stddev (pop): " << stddev(s, false) << std::endl;
    std::cout << "stddev (sample): " << stddev(s, true) << std::endl;
    std::cout << "variance (pop): " << variance(s, false) << std::endl;
    std::cout << "variance (sample): " << variance(s, true) << std::endl;
    std::cout << "min: " << min(s) << std::endl;
    std::cout << "max: " << max(s) << std::endl;
    std::cout << "quantile 0.5: " << quantile(s, 0.5) << std::endl;
    std::cout << "quantile 0.25: " << quantile(s, 0.25) << std::endl;
    std::cout << "quantile 0.75: " << quantile(s, 0.75) << std::endl;
    auto [q1, q3] = iqr(s);
    std::cout << "IQR: " << q3 - q1 << std::endl;
    std::cout << "scipy.stats.iqr: " << scipy_iqr(s) << std::endl;
    std::cout << "skew: " << skewness(s) << std::endl;
    std::cout << "kurtosis (fisher=false): " << kurtosis(s, false) << std::endl;
    std::cout << "kurtosis (fisher=true): " << kurtosis(s, true) << std::endl;

    // Random normal
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(0, 1);
    std::vector<double> x(200);
    for (auto& xi : x) xi = d(gen);

    double mu = mean(x);
    double sigma = stddev(x, true);
    std::vector<double> Z(x.size());
    for (size_t i = 0; i < x.size(); ++i) Z[i] = (x[i] - mu) / sigma;

    // Outlier detection
    double Q1 = quantile(x, 0.25);
    double Q3 = quantile(x, 0.75);
    double IQR = Q3 - Q1;
    double lower_inner_fence = Q1 - 1.5 * IQR;
    double upper_inner_fence = Q3 + 1.5 * IQR;
    std::vector<double> outliers;
    for (auto xi : x) if (xi < lower_inner_fence || xi > upper_inner_fence) outliers.push_back(xi);
    double lower_outer_fence = Q1 - 3 * IQR;
    double upper_outer_fence = Q3 + 3 * IQR;
    std::vector<double> ext_outliers, mild_outliers;
    for (auto o : outliers) {
        if (o < lower_outer_fence || o > upper_outer_fence) ext_outliers.push_back(o);
        else mild_outliers.push_back(o);
    }
    std::cout << "outliers: " << outliers.size() << std::endl;
    std::cout << "ext_outliers: " << ext_outliers.size() << std::endl;
    std::cout << "mild_outliers: " << mild_outliers.size() << std::endl;

    // DataFrame and correlation (using Eigen)
    Eigen::MatrixXd df = Eigen::MatrixXd::Random(50, 3);
    Eigen::MatrixXd corr = (df.transpose() * df) / (df.rows() - 1);
    std::cout << "Correlation matrix:\n" << corr << std::endl;

    // One-hot encoding (manual example)
    std::vector<std::string> gender = {"Male", "Female", "Female"};
    std::vector<int> group = {1, 3, 2};
    std::map<std::string, int> gender_map = {{"Male", 0}, {"Female", 1}};
    std::vector<std::vector<int>> X_onehot(gender.size(), std::vector<int>(gender_map.size() + 3, 0));
    for (size_t i = 0; i < gender.size(); ++i) {
        X_onehot[i][gender_map[gender[i]]] = 1;
        X_onehot[i][2 + group[i] - 1] = 1;
    }
    std::cout << "One-hot encoded X:\n";
    for (const auto& row : X_onehot) {
        for (auto v : row) std::cout << v << " ";
        std::cout << std::endl;
    }

    // StandardScaler and MinMaxScaler (manual)
    std::vector<std::vector<double>> data = {{0,0},{0,0},{1,1},{1,1}};
    std::vector<double> mean_col(2, 0.0), std_col(2, 0.0), min_col(2, 1e9), max_col(2, -1e9);
    for (int j = 0; j < 2; ++j) {
        for (int i = 0; i < 4; ++i) {
            mean_col[j] += data[i][j];
            min_col[j] = std::min(min_col[j], data[i][j]);
            max_col[j] = std::max(max_col[j], data[i][j]);
        }
        mean_col[j] /= 4.0;
        for (int i = 0; i < 4; ++i) std_col[j] += (data[i][j] - mean_col[j]) * (data[i][j] - mean_col[j]);
        std_col[j] = std::sqrt(std_col[j] / 4.0);
    }
    std::cout << "StandardScaler transform [2,2]: ";
    for (int j = 0; j < 2; ++j) std::cout << (2 - mean_col[j]) / std_col[j] << " ";
    std::cout << std::endl;
    std::cout << "MinMaxScaler transform [2,2]: ";
    for (int j = 0; j < 2; ++j) std::cout << (2 - min_col[j]) / (max_col[j] - min_col[j]) << " ";
    std::cout << std::endl;

    return 0;
}
