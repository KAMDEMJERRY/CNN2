# pragma once
# include <iostream>
# include <filesystem>
# include <chrono>
# include <ctime>
# include <sstream>
# include <vector>

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

// A streambuf that duplicates output to two buffers, with optional ANSI stripping on the second one.
class TeeBuffer : public std::streambuf {
public:
    TeeBuffer(std::streambuf* sb1, std::streambuf* sb2, bool stripColors2 = false);
protected:
    virtual int overflow(int c) override;
    virtual int sync() override;
private:
    std::streambuf *sb1, *sb2;
    bool stripColors2;
    // 0 = normal, 1 = saw ESC, 2 = in CSI (ESC [ ... final)
    int esc_state = 0;
};
