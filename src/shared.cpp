#include "shared.hpp"
#include <cstdio>
#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif
/**
 * retourne le dossier racine
 */

// Couleurs ANSI — initialisées au démarrage en fonction de stdout
std::string red;
std::string green;
std::string yellow;
std::string blue;
std::string reset;

// Initialisation: n'activer les séquences ANSI que si stdout est un terminal
namespace {
struct ColorInitializer {
    ColorInitializer() {
#ifdef _WIN32
        bool use_colors = _isatty(_fileno(stdout));
#else
        bool use_colors = isatty(fileno(stdout));
#endif
        if (use_colors) {
            red = "\033[31m";
            green = "\033[32m";
            yellow = "\033[33m";
            blue = "\033[34m";
            reset = "\033[0m";
        } else {
            red = green = yellow = blue = reset = "";
        }
    }
} color_initializer;
} // anonymous namespace

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

// =============================================================================
// TeeBuffer Implementation
// =============================================================================

TeeBuffer::TeeBuffer(std::streambuf* sb1, std::streambuf* sb2, bool stripColors2)
    : sb1(sb1), sb2(sb2), stripColors2(stripColors2), esc_state(0) {}

int TeeBuffer::overflow(int c) {
    if (c == EOF) return !EOF;

    // Always write to first buffer (console)
    if (sb1->sputc(c) == EOF) return EOF;

    // If we're not stripping colors, write to second buffer directly
    if (!stripColors2) {
        if (sb2->sputc(c) == EOF) return EOF;
        return c;
    }

    unsigned char uc = static_cast<unsigned char>(c);

    // State machine for ANSI escapes:
    // esc_state == 0 : normal
    // esc_state == 1 : saw ESC (0x1B)
    // esc_state == 2 : inside CSI (after ESC '['), skip until final byte 0x40..0x7E
    if (esc_state == 0) {
        if (uc == 0x1B) { // ESC
            esc_state = 1;
            // skip ESC itself from second buffer
            return c;
        }
        // normal char -> write to file
        if (sb2->sputc(c) == EOF) return EOF;
        return c;
    }

    if (esc_state == 1) {
        if (uc == '[') {
            // CSI introducer -> enter CSI state
            esc_state = 2;
            return c; // skip '['
        }
        // Other short escape sequence: consume this byte and return to normal
        esc_state = 0;
        return c; // skip this byte
    }

    // esc_state == 2 : inside CSI
    if (esc_state == 2) {
        // final byte of CSI is in 0x40..0x7E
        if (uc >= 0x40 && uc <= 0x7E) {
            esc_state = 0; // end of CSI
            return c; // skip final byte
        }
        // otherwise skip intermediate/parameter bytes
        return c;
    }

    // fallback: write
    if (sb2->sputc(c) == EOF) return EOF;
    return c;
}

int TeeBuffer::sync() {
    int r1 = sb1->pubsync();
    int r2 = sb2->pubsync();
    return (r1 == 0 && r2 == 0) ? 0 : -1;
}
