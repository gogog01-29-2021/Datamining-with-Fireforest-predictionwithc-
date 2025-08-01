#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <map>
#include <filesystem>
#include <sstream>
#include <Eigen/Dense>

namespace fs = std::filesystem;

struct CSVInfo {
    std::vector<std::string> columns;
    std::vector<std::string> dtypes;
    std::pair<int,int> shape;
};

CSVInfo analyze_csv(const std::string& file_path) {
    std::ifstream file(file_path);
    std::string line;
    CSVInfo info;
    if (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string col;
        while (std::getline(ss, col, ',')) info.columns.push_back(col);
    }
    int row_count = 0;
    while (std::getline(file, line)) ++row_count;
    info.shape = {row_count, (int)info.columns.size()};
    info.dtypes = std::vector<std::string>(info.columns.size(), "string/double"); // C++ does not infer dtypes
    return info;
}

int main() {
    std::string folder_path = "250314Shenfeng/datasets";
    std::map<std::string, CSVInfo> file_info;
    for (const auto& entry : fs::directory_iterator(folder_path)) {
        if (entry.path().extension() == ".csv") {
            try {
                file_info[entry.path().filename()] = analyze_csv(entry.path().string());
            } catch (...) {
                std::cout << "Error reading " << entry.path().filename() << std::endl;
            }
        }
    }
    std::cout << "Comparing CSV files...\n";
    for (const auto& [file, info] : file_info) {
        std::cout << "File: " << file << std::endl;
        std::cout << "Columns: ";
        for (auto& c : info.columns) std::cout << c << " ";
        std::cout << std::endl;
        std::cout << "Shape: (" << info.shape.first << ", " << info.shape.second << ")" << std::endl;
        std::cout << "Dtypes: ";
        for (auto& d : info.dtypes) std::cout << d << " ";
        std::cout << std::endl << std::string(50,'-') << std::endl;
    }
    // Check if all files have the same columns
    std::set<std::vector<std::string>> columns_set;
    for (const auto& [_, info] : file_info) columns_set.insert(info.columns);
    if (columns_set.size() == 1) std::cout << "All files have the same columns." << std::endl;
    else std::cout << "Files have different columns." << std::endl;
    // Check if all files have the same dtypes
    std::set<std::vector<std::string>> dtypes_set;
    for (const auto& [_, info] : file_info) dtypes_set.insert(info.dtypes);
    if (dtypes_set.size() == 1) std::cout << "All files have the same data types." << std::endl;
    else std::cout << "Files have different data types." << std::endl;
    return 0;
}
