#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <map>
#include <set>
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
    // Group files by structure (columns and dtypes)
    std::map<std::pair<std::vector<std::string>, std::vector<std::string>>, std::vector<std::string>> grouped_files;
    for (const auto& [file, info] : file_info) {
        auto key = std::make_pair(info.columns, info.dtypes);
        grouped_files[key].push_back(file);
    }
    std::cout << "Grouped files by structure:\n";
    int group_num = 1;
    for (const auto& [key, files] : grouped_files) {
        std::cout << "Group " << group_num++ << ":\n";
        std::cout << "Columns: ";
        for (auto& c : key.first) std::cout << c << " ";
        std::cout << std::endl;
        std::cout << "Data Types: ";
        for (auto& d : key.second) std::cout << d << " ";
        std::cout << std::endl;
        std::cout << "Files: ";
        for (auto& f : files) std::cout << f << " ";
        std::cout << std::endl << std::string(50,'-') << std::endl;
    }
    return 0;
}
