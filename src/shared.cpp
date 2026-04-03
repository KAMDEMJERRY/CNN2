#include "shared.hpp"
/**
 * retourne le dossier racine
 */

std::string red = "\033[31m";
std::string green = "\033[32m";
std::string yellow = "\033[33m";
std::string blue = "\033[34m";
std::string reset = "\033[0m";

std::string relativePath(std::string filename) {
    try {
        fs::path currentDir = fs::current_path();
        std::string filePath = currentDir.string() + filename;
        return filePath;

    }
    catch (const fs::filesystem_error& e) {
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

// =============================================================================
// Utilitaires
// =============================================================================
void section(const std::string& title) {
    const std::string bar(60, '=');
    std::cout << "\n" << bar << "\n  " << title << "\n" << bar << "\n";
}

void requireDir(const std::string& path) {
    if (!fs::exists(path))
        throw std::runtime_error("Répertoire introuvable : " + path);
}
