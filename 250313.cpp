#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <algorithm>
#include <numeric>
#include <Eigen/Dense>
#include <iomanip>
#include <cmath>

// Helper: Print Eigen matrix
void print_matrix(const Eigen::MatrixXd& mat, const std::string& name = "") {
    if (!name.empty()) std::cout << name << ":\n";
    std::cout << mat << "\n";
}

int main() {
    // Series
    std::vector<int> s = {1,2,3,4,5,6,7,8,9,10};
    std::cout << "Series: ";
    for (auto v : s) std::cout << v << " ";
    std::cout << std::endl;
    std::cout << "Head: ";
    for (int i = 0; i < 3; ++i) std::cout << s[i] << " ";
    std::cout << std::endl;
    std::cout << "Tail: ";
    for (int i = s.size()-3; i < s.size(); ++i) std::cout << s[i] << " ";
    std::cout << std::endl;

    // DataFrame
    Eigen::MatrixXd df = Eigen::MatrixXd::Random(6,4); // 6x4 random
    std::vector<int> df_index = {11,12,14,15,16,17};
    std::vector<std::string> df_columns = {"A","B","C","D"};
    print_matrix(df, "df");

    // Column selection
    std::cout << "df['A']: ";
    for (int i = 0; i < df.rows(); ++i) std::cout << df(i,0) << " ";
    std::cout << std::endl;
    std::cout << "df[['A','B']]:\n";
    for (int i = 0; i < df.rows(); ++i) std::cout << df(i,0) << ", " << df(i,1) << std::endl;

    // dtypes
    std::cout << "All columns are double (Eigen::MatrixXd)" << std::endl;

    // Tail
    std::cout << "df.tail():\n";
    for (int i = df.rows()-3; i < df.rows(); ++i) {
        for (int j = 0; j < df.cols(); ++j) std::cout << df(i,j) << " ";
        std::cout << std::endl;
    }

    // Index, columns, values
    std::cout << "Index: ";
    for (auto idx : df_index) std::cout << idx << " ";
    std::cout << std::endl;
    std::cout << "Columns: ";
    for (auto col : df_columns) std::cout << col << " ";
    std::cout << std::endl;
    std::cout << "Values:\n";
    print_matrix(df);

    // Describe
    std::cout << "Describe:\n";
    for (int j = 0; j < df.cols(); ++j) {
        double mean = df.col(j).mean();
        double stddev = std::sqrt((df.col(j).array() - mean).square().sum() / (df.rows()-1));
        double min = df.col(j).minCoeff();
        double max = df.col(j).maxCoeff();
        std::cout << df_columns[j] << ": mean=" << mean << ", std=" << stddev << ", min=" << min << ", max=" << max << std::endl;
    }

    // Transpose
    print_matrix(df.transpose(), "df.T");

    // Sort columns (descending)
    std::cout << "Sort columns descending:\n";
    Eigen::MatrixXd df_sorted = df;
    std::reverse(df_columns.begin(), df_columns.end());
    for (int j = 0; j < df.cols(); ++j) {
        for (int i = 0; i < df.rows(); ++i) {
            df_sorted(i,j) = df(i,df.cols()-1-j);
        }
    }
    print_matrix(df_sorted);

    // Sort by column B
    std::vector<std::pair<double,int>> b_vals;
    for (int i = 0; i < df.rows(); ++i) b_vals.push_back({df(i,1),i});
    std::sort(b_vals.begin(), b_vals.end());
    std::cout << "Sort by B:\n";
    for (auto& p : b_vals) {
        for (int j = 0; j < df.cols(); ++j) std::cout << df(p.second,j) << " ";
        std::cout << std::endl;
    }

    // Integer DataFrame
    Eigen::MatrixXi df2 = Eigen::MatrixXi::NullaryExpr(6,4,[](){return rand()%3;});
    std::vector<std::string> df2_index = {"a","b","c","d","e","f"};
    print_matrix(df2.cast<double>(), "df2");

    // Sorts
    // ...sorts omitted for brevity...

    // Logical operator
    std::cout << "df[df['A']>0]:\n";
    for (int i = 0; i < df.rows(); ++i) {
        if (df(i,0)>0) {
            for (int j = 0; j < df.cols(); ++j) std::cout << df(i,j) << " ";
            std::cout << std::endl;
        }
    }

    // Add new column
    std::vector<std::string> E = {"one","one","two","three","four","three"};
    std::vector<int> F(df2.rows());
    for (int i = 0; i < df2.rows(); ++i) F[i] = df2(i,0) + df2(i,1);
    std::cout << "E: "; for (auto& e : E) std::cout << e << " "; std::cout << std::endl;
    std::cout << "F: "; for (auto& f : F) std::cout << f << " "; std::cout << std::endl;

    // Statistics
    std::cout << "Mean per row:\n";
    for (int i = 0; i < df.rows(); ++i) {
        double mean = df.row(i).mean();
        std::cout << "Row " << i << ": " << mean << std::endl;
    }
    // ...other stats omitted for brevity...

    // Value counts (categorical)
    std::map<std::string,int> value_counts;
    for (auto& e : E) value_counts[e]++;
    std::cout << "Value counts for E:\n";
    for (auto& kv : value_counts) std::cout << kv.first << ": " << kv.second << std::endl;

    // Groupby
    std::map<std::string,int> group_sum;
    for (int i = 0; i < E.size(); ++i) group_sum[E[i]] += df2(i,0);
    std::cout << "Groupby E, sum of A:\n";
    for (auto& kv : group_sum) std::cout << kv.first << ": " << kv.second << std::endl;

    // Missing value handling, concat, merge, image section omitted for brevity
    return 0;
}
