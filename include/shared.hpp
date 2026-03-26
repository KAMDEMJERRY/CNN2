# pragma once

# include <filesystem>
# include <chrono>
# include <ctime>
# include <sstream>

namespace fs = std::filesystem;
/**
 * retourne le dossier racine
 */
std::string relativePath(std::string filename) {
    try {
        fs::path currentDir = fs::current_path();
        std::string filePath = currentDir.string() + filename;
        return filePath;

    } catch (const fs::filesystem_error& e) {
        std::cerr << "Filesystem error: " << e.what() << std::endl;
    }
    return "";
}

std::string currentTime() {
    auto now = std::chrono::system_clock::now();
    std::time_t currentTime = std::chrono::system_clock::to_time_t(now);
    std::tm* localTime = std::localtime(&currentTime);
    std::stringstream ss;
    ss << std::put_time(localTime, "%Y-%m-%d %H:%M:%S");
    return ss.str();
}


    const std::string red = "\033[31m";
    const std::string green = "\033[32m";
    const std::string yellow = "\033[33m";
    const std::string blue = "\033[34m";
    const std::string reset = "\033[0m";