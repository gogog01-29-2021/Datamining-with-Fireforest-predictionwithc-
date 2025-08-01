#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <map>
#include <algorithm>
#include <iomanip>
#include <ctime>
#include <Eigen/Dense>

struct Record {
    std::string date;
    std::string station_name;
    // ... add other columns as needed ...
};

// Parse CSV
std::vector<Record> load_csv(const std::string& filename) {
    std::vector<Record> data;
    std::ifstream file(filename);
    std::string line;
    std::getline(file, line); // skip header
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string date, station_name;
        std::getline(ss, date, ',');
        std::getline(ss, station_name, ',');
        // ...parse other columns as needed...
        if (!date.empty()) data.push_back({date, station_name});
    }
    return data;
}

// Parse date string to struct tm
std::tm parse_date(const std::string& date_str) {
    std::tm tm = {};
    strptime(date_str.c_str(), "%d/%m/%Y", &tm);
    return tm;
}

int main() {
    std::string file_path = "250314Shenfeng/2023-open-data-dfb-ambulance.csv";
    auto data = load_csv(file_path);
    std::cout << "Loaded " << data.size() << " records." << std::endl;
    // Print head
    for (size_t i = 0; i < std::min<size_t>(5, data.size()); ++i) {
        std::cout << data[i].date << ", " << data[i].station_name << std::endl;
    }
    // Print tail
    for (size_t i = data.size()-std::min<size_t>(5, data.size()); i < data.size(); ++i) {
        std::cout << data[i].date << ", " << data[i].station_name << std::endl;
    }
    // Value counts for Station Name
    std::map<std::string,int> station_counts;
    for (const auto& r : data) station_counts[r.station_name]++;
    std::cout << "Station Name value counts:\n";
    for (const auto& kv : station_counts) std::cout << kv.first << ": " << kv.second << std::endl;
    // Extract year/month/week/day
    std::map<int,int> year_counts, month_counts, week_counts;
    std::map<std::string,int> day_counts;
    for (const auto& r : data) {
        std::tm tm = parse_date(r.date);
        int year = tm.tm_year+1900;
        int month = tm.tm_mon+1;
        int week = tm.tm_yday/7+1;
        char buf[16];
        strftime(buf, sizeof(buf), "%A", &tm);
        std::string day(buf);
        year_counts[year]++;
        month_counts[month]++;
        week_counts[week]++;
        day_counts[day]++;
    }
    std::cout << "Year counts:\n";
    for (const auto& kv : year_counts) std::cout << kv.first << ": " << kv.second << std::endl;
    std::cout << "Month counts:\n";
    for (const auto& kv : month_counts) std::cout << kv.first << ": " << kv.second << std::endl;
    std::cout << "Week counts:\n";
    for (const auto& kv : week_counts) std::cout << kv.first << ": " << kv.second << std::endl;
    std::cout << "Day counts:\n";
    for (const auto& kv : day_counts) std::cout << kv.first << ": " << kv.second << std::endl;
    // Null checks omitted (not typical in C++ vector)
    // Visualization and seasonal decomposition omitted
    return 0;
}
