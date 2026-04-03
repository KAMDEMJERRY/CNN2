# pragma once
# include <iostream>
# include <filesystem>
# include <chrono>
# include <ctime>
# include <sstream>

namespace fs = std::filesystem;

extern std::string red;
extern std::string green;
extern std::string yellow;
extern std::string blue;
extern std::string reset;

std::string relativePath(std::string filename);

std::string currentTime();

void section(const std::string& title);

void requireDir(const std::string& path);
