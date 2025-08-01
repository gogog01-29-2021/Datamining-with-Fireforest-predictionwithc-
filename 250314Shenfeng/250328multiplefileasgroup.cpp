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

void combine_csvs(const std::vector<std::string>& files, const std::string& folder_path, const std::string& output_file) {
    std::ofstream out(output_file);
    bool header_written = false;
    for (const auto& file : files) {
        std::ifstream in(folder_path + "/" + file);
        std::string line;
        bool first_line = true;
        while (std::getline(in, line)) {
            if (first_line) {
                if (!header_written) {
                    out << line << "\n";
                    header_written = true;
                }
                first_line = false;
            } else {
                out << line << "\n";
            }
        }
    }
    out.close();
}

int main() {
    std::string folder_path = "250314Shenfeng/datasets";
    std::string output_folder = "250314Shenfeng/combined";
    fs::create_directories(output_folder);
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
    int group_num = 1;
    for (const auto& [key, files] : grouped_files) {
        std::cout << "Combining files in Group " << group_num << ": ";
        for (auto& f : files) std::cout << f << " ";
        std::cout << std::endl;
        std::string output_file = output_folder + "/group_" + std::to_string(group_num) + "_combined.csv";
        combine_csvs(files, folder_path, output_file);
        std::cout << "Group " << group_num << " combined file saved as: " << output_file << std::endl;
        ++group_num;
    }
    return 0;
}
