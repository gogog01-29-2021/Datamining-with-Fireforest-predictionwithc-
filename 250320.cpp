#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <Eigen/Dense>
#include <map>
#include <iomanip>
#include <string>

// Helper: Print vector
void print_vec(const std::vector<double>& v, const std::string& name = "") {
    if (!name.empty()) std::cout << name << ": ";
    for (auto x : v) std::cout << x << " ";
    std::cout << std::endl;
}

int main() {
    std::vector<double> s = {2,2,4,5,5,5,8,9,9,9,12};
    std::cout << "len(s): " << s.size() << std::endl;
    double mean = std::accumulate(s.begin(), s.end(), 0.0) / s.size();
    std::cout << "mean: " << mean << std::endl;
    std::vector<double> sorted_s = s;
    std::sort(sorted_s.begin(), sorted_s.end());
    double median = sorted_s[sorted_s.size()/2];
    std::cout << "median: " << median << std::endl;
    double sq_sum = std::inner_product(s.begin(), s.end(), s.begin(), 0.0);
    double stddev = std::sqrt(sq_sum/s.size() - mean*mean);
    std::cout << "stddev (ddof=0): " << stddev << std::endl;
    double stddev1 = std::sqrt(sq_sum/(s.size()-1) - mean*mean);
    std::cout << "stddev (ddof=1): " << stddev1 << std::endl;
    double var = stddev*stddev;
    double var1 = stddev1*stddev1;
    std::cout << "var (ddof=0): " << var << std::endl;
    std::cout << "var (ddof=1): " << var1 << std::endl;
    std::cout << "min: " << *std::min_element(s.begin(), s.end()) << std::endl;
    std::cout << "max: " << *std::max_element(s.begin(), s.end()) << std::endl;
    auto quantile = [](std::vector<double> v, double q) {
        std::sort(v.begin(), v.end());
        double pos = (v.size()-1)*q;
        int idx = static_cast<int>(pos);
        double frac = pos-idx;
        if (idx+1 < v.size())
            return v[idx]*(1-frac) + v[idx+1]*frac;
        else
            return v[idx];
    };
    std::cout << "25th percentile: " << quantile(s,0.25) << std::endl;
    std::cout << "75th percentile: " << quantile(s,0.75) << std::endl;
    std::cout << "50th percentile: " << quantile(s,0.5) << std::endl;
    std::cout << "IQR: " << quantile(s,0.75)-quantile(s,0.25) << std::endl;

    // Skewness & Kurtosis (manual)
    double m3 = 0, m4 = 0;
    for (auto x : s) {
        m3 += std::pow(x-mean,3);
        m4 += std::pow(x-mean,4);
    }
    m3 /= s.size();
    m4 /= s.size();
    double skew = m3/std::pow(stddev,3);
    double kurt = m4/std::pow(stddev,4);
    std::cout << "skewness: " << skew << std::endl;
    std::cout << "kurtosis: " << kurt << std::endl;

    // Histogram
    int bins = 4;
    double minv = *std::min_element(s.begin(), s.end());
    double maxv = *std::max_element(s.begin(), s.end());
    double bin_width = (maxv-minv)/bins;
    std::vector<int> hist(bins,0);
    for (auto x : s) {
        int idx = std::min(static_cast<int>((x-minv)/bin_width), bins-1);
        hist[idx]++;
    }
    std::cout << "Histogram: ";
    for (auto h : hist) std::cout << h << " ";
    std::cout << std::endl;

    // Outlier detection (z-score)
    std::vector<double> z;
    for (auto x : s) z.push_back((x-mean)/stddev);
    std::cout << "Outliers (z<-3 or z>3): ";
    for (size_t i = 0; i < s.size(); ++i) {
        if (z[i]<-3 || z[i]>3) std::cout << s[i] << " ";
    }
    std::cout << std::endl;

    // Correlation
    Eigen::MatrixXd df = Eigen::MatrixXd::Random(50,3);
    Eigen::MatrixXd corr = df.transpose()*df;
    std::cout << "Correlation matrix:\n" << corr << std::endl;

    // One-hot encoding
    std::vector<std::string> genders = {"Male","Female","Female"};
    std::vector<int> groups = {1,3,2};
    std::map<std::string,int> gender_map = {{"Male",0},{"Female",1}};
    Eigen::MatrixXi X(3,2);
    for (int i = 0; i < 3; ++i) {
        X(i,0) = gender_map[genders[i]];
        X(i,1) = groups[i];
    }
    std::cout << "One-hot encoding:\n";
    for (int i = 0; i < X.rows(); ++i) {
        std::cout << (X(i,0)==0?"[1,0]":"[0,1]") << " Group: " << X(i,1) << std::endl;
    }

    // Scaling
    Eigen::MatrixXd data(4,2);
    data << 0,0,0,0,1,1,1,1;
    Eigen::RowVectorXd meanv = data.colwise().mean();
    Eigen::RowVectorXd stdv = ((data.rowwise()-meanv).array().square().colwise().sum()/(data.rows()-1)).sqrt();
    Eigen::MatrixXd scaled = (data.rowwise()-meanv).array().rowwise()/stdv.array();
    std::cout << "Standard scaled:\n" << scaled << std::endl;
    Eigen::RowVectorXd minv2 = data.colwise().minCoeff();
    Eigen::RowVectorXd maxv2 = data.colwise().maxCoeff();
    Eigen::MatrixXd minmax = (data.rowwise()-minv2).array().rowwise()/(maxv2-minv2).array();
    std::cout << "MinMax scaled:\n" << minmax << std::endl;
    return 0;
}
