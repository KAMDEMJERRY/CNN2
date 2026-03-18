# pragma once

# include <filesystem>

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