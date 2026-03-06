#pragma once
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cstdint>
#include <cstring>
#include <regex>
#include <iostream>
#include "Tensor.hpp"

// ─────────────────────────────────────────────────────────────────────────────
// Dtype détecté dans le header .npy
// ─────────────────────────────────────────────────────────────────────────────
enum class NpyDtype {
    FLOAT32,   // '<f4'
    FLOAT64,   // '<f8'
    INT16,     // '<i2'
    INT32,     // '<i4'
    INT64,     // '<i8'
    UINT8,     // '|u1'
    UINT16,    // '<u2'
    UNKNOWN
};

// ─────────────────────────────────────────────────────────────────────────────
// Métadonnées extraites du header
// ─────────────────────────────────────────────────────────────────────────────
struct NpyHeader {
    NpyDtype      dtype         = NpyDtype::UNKNOWN;
    bool          fortran_order = false;
    std::vector<int> shape;
    size_t        data_offset   = 0;   // position des données dans le fichier
    size_t        num_elements  = 0;   // produit de shape
    size_t        bytes_per_elem = 0;
    bool          is_little_endian = true;
};

// ─────────────────────────────────────────────────────────────────────────────
// NpyParser
// ─────────────────────────────────────────────────────────────────────────────
class NpyParser {
public:

    // ── API principale ────────────────────────────────────────────────────────

    // Charge un .npy et retourne un Tensor 5D (batch, 1, D, H, W)
    // Gère automatiquement : uint8 → /255, float64 → cast, int16/uint16 → cast
    static Tensor load(const std::string& filepath, bool verbose = false) {
        std::ifstream file(filepath, std::ios::binary);
        if (!file.is_open())
            throw std::runtime_error("[NpyParser] Cannot open: " + filepath);

        NpyHeader header = parseHeader(file, filepath);

        if (verbose) printHeader(header, filepath);

        Tensor tensor = allocateTensor(header);
        readAndCast(file, header, tensor);

        return tensor;
    }

    // Charge un .npy de labels et retourne un std::vector<int>
    static std::vector<int> loadLabels(const std::string& filepath, bool verbose = false) {
        std::ifstream file(filepath, std::ios::binary);
        if (!file.is_open())
            throw std::runtime_error("[NpyParser] Cannot open: " + filepath);

        NpyHeader header = parseHeader(file, filepath);

        if (verbose) printHeader(header, filepath);

        // Labels attendus : (N,) ou (N, 1)
        size_t n = header.num_elements;
        std::vector<int> labels(n);

        if (header.dtype == NpyDtype::INT64) {
            std::vector<int64_t> raw(n);
            file.read(reinterpret_cast<char*>(raw.data()), n * sizeof(int64_t));
            for (size_t i = 0; i < n; ++i) labels[i] = static_cast<int>(raw[i]);
        }
        else if (header.dtype == NpyDtype::INT32) {
            std::vector<int32_t> raw(n);
            file.read(reinterpret_cast<char*>(raw.data()), n * sizeof(int32_t));
            for (size_t i = 0; i < n; ++i) labels[i] = static_cast<int>(raw[i]);
        }
        else if (header.dtype == NpyDtype::UINT8) {
            std::vector<uint8_t> raw(n);
            file.read(reinterpret_cast<char*>(raw.data()), n * sizeof(uint8_t));
            for (size_t i = 0; i < n; ++i) labels[i] = static_cast<int>(raw[i]);
        }
        else {
            throw std::runtime_error("[NpyParser] Unsupported label dtype");
        }

        return labels;
    }

    // Affiche les infos du header sans charger les données
    static void inspect(const std::string& filepath) {
        std::ifstream file(filepath, std::ios::binary);
        if (!file.is_open())
            throw std::runtime_error("[NpyParser] Cannot open: " + filepath);
        NpyHeader header = parseHeader(file, filepath);
        printHeader(header, filepath);

        // Vérification taille fichier
        file.seekg(0, std::ios::end);
        size_t file_size = file.tellg();
        size_t expected  = header.data_offset + header.num_elements * header.bytes_per_elem;
        std::cout << "   File size     : " << file_size << " bytes" << std::endl;
        std::cout << "   Expected size : " << expected  << " bytes" << std::endl;
        if (file_size != expected)
            std::cout << "   ⚠️  Taille incohérente !" << std::endl;
        else
            std::cout << "   ✅ Fichier cohérent" << std::endl;
    }

private:

    // ── Parsing du header ─────────────────────────────────────────────────────

    static NpyHeader parseHeader(std::ifstream& file, const std::string& path) {
        // 1. Magic number
        char magic[6];
        file.read(magic, 6);
        if (std::string(magic, 6) != "\x93NUMPY")
            throw std::runtime_error("[NpyParser] Not a valid .npy file: " + path);

        // 2. Version
        uint8_t major, minor;
        file.read(reinterpret_cast<char*>(&major), 1);
        file.read(reinterpret_cast<char*>(&minor), 1);

        // 3. Longueur du header
        size_t header_len = 0;
        if (major == 1) {
            uint16_t hlen;
            file.read(reinterpret_cast<char*>(&hlen), 2);
            header_len = hlen;
        } else {
            uint32_t hlen;
            file.read(reinterpret_cast<char*>(&hlen), 4);
            header_len = hlen;
        }

        // 4. Lire le header ASCII
        std::string header_str(header_len, '\0');
        file.read(&header_str[0], header_len);

        // 5. Parser le dictionnaire
        NpyHeader header;
        header.data_offset = file.tellg();
        parseDict(header_str, header);

        return header;
    }

    static void parseDict(const std::string& dict, NpyHeader& header) {
        // ── descr ──
        std::regex descr_re(R"('descr'\s*:\s*'([^']+)')");
        std::smatch m;
        if (std::regex_search(dict, m, descr_re)) {
            std::string descr = m[1].str();
            header.is_little_endian = (descr[0] != '>');
            header.dtype            = parseDtype(descr);
            header.bytes_per_elem   = bytesPerElem(header.dtype);
        } else {
            throw std::runtime_error("[NpyParser] Cannot find 'descr' in header");
        }

        // ── fortran_order ──
        std::regex order_re(R"('fortran_order'\s*:\s*(True|False))");
        if (std::regex_search(dict, m, order_re)) {
            header.fortran_order = (m[1].str() == "True");
            if (header.fortran_order)
                std::cerr << "[NpyParser] ⚠️  fortran_order=True détecté, les données seront transposées" << std::endl;
        }

        // ── shape ──
        std::regex shape_re(R"('shape'\s*:\s*\(([^)]*)\))");
        if (std::regex_search(dict, m, shape_re)) {
            std::string shape_str = m[1].str();
            header.shape = parseShape(shape_str);
        } else {
            throw std::runtime_error("[NpyParser] Cannot find 'shape' in header");
        }

        // ── num_elements ──
        header.num_elements = 1;
        for (int d : header.shape) header.num_elements *= d;
    }

    static std::vector<int> parseShape(const std::string& s) {
        std::vector<int> shape;
        std::regex num_re(R"(\d+)");
        auto begin = std::sregex_iterator(s.begin(), s.end(), num_re);
        auto end   = std::sregex_iterator();
        for (auto it = begin; it != end; ++it)
            shape.push_back(std::stoi((*it).str()));
        return shape;
    }

    static NpyDtype parseDtype(const std::string& descr) {
        // Ignorer le premier caractère (endianness) sauf pour |u1
        std::string d = descr;
        if (d == "|u1" || d == "u1")  return NpyDtype::UINT8;
        if (d == "<f4" || d == "f4")  return NpyDtype::FLOAT32;
        if (d == "<f8" || d == "f8")  return NpyDtype::FLOAT64;
        if (d == "<i2")               return NpyDtype::INT16;
        if (d == "<i4")               return NpyDtype::INT32;
        if (d == "<i8")               return NpyDtype::INT64;
        if (d == "<u2")               return NpyDtype::UINT16;
        if (d == ">f4")               return NpyDtype::FLOAT32; // big-endian, swap géré
        if (d == ">f8")               return NpyDtype::FLOAT64;
        return NpyDtype::UNKNOWN;
    }

    static size_t bytesPerElem(NpyDtype dtype) {
        switch (dtype) {
            case NpyDtype::UINT8:   return 1;
            case NpyDtype::INT16:
            case NpyDtype::UINT16:  return 2;
            case NpyDtype::FLOAT32:
            case NpyDtype::INT32:   return 4;
            case NpyDtype::FLOAT64:
            case NpyDtype::INT64:   return 8;
            default: return 0;
        }
    }

    // ── Allocation du Tensor ──────────────────────────────────────────────────

    // Wrapping des shapes vers 5D (batch, 1, D, H, W)
    static Tensor allocateTensor(const NpyHeader& header) {
        const auto& s = header.shape;
        int b = 1, c = 1, d = 1, h = 1, w = 1;

        if (s.size() == 5) {         // (B, C, D, H, W) — déjà 5D
            b = s[0]; c = s[1]; d = s[2]; h = s[3]; w = s[4];
        } else if (s.size() == 4) {  // (N, D, H, W) — cas MedMNIST3D images
            b = s[0]; d = s[1]; h = s[2]; w = s[3];
        } else if (s.size() == 3) {  // (D, H, W) — volume unique
            d = s[0]; h = s[1]; w = s[2];
        } else if (s.size() == 2) {  // (N, 1) — labels (ne pas utiliser ici)
            b = s[0]; c = s[1];
        } else if (s.size() == 1) {  // (N,)
            b = s[0];
        }

        return Tensor(b, c, d, h, w);
    }

    // ── Lecture et cast vers float32 ──────────────────────────────────────────

    static void readAndCast(std::ifstream& file, const NpyHeader& header, Tensor& tensor) {
        size_t n = header.num_elements;
        float* dst = tensor.getData();

        switch (header.dtype) {

            case NpyDtype::FLOAT32: {
                // Lecture directe — memcpy
                file.read(reinterpret_cast<char*>(dst), n * sizeof(float));
                if (!header.is_little_endian) byteswap32(dst, n);
                break;
            }

            case NpyDtype::UINT8: {
                // uint8 [0,255] → float32 [0,1]
                std::vector<uint8_t> buf(n);
                file.read(reinterpret_cast<char*>(buf.data()), n);
                for (size_t i = 0; i < n; ++i)
                    dst[i] = static_cast<float>(buf[i]) / 255.0f;
                break;
            }

            case NpyDtype::FLOAT64: {
                std::vector<double> buf(n);
                file.read(reinterpret_cast<char*>(buf.data()), n * sizeof(double));
                for (size_t i = 0; i < n; ++i)
                    dst[i] = static_cast<float>(buf[i]);
                break;
            }

            case NpyDtype::INT16: {
                // int16 (Hounsfield CT) → float32, pas de normalisation ici
                std::vector<int16_t> buf(n);
                file.read(reinterpret_cast<char*>(buf.data()), n * sizeof(int16_t));
                for (size_t i = 0; i < n; ++i)
                    dst[i] = static_cast<float>(buf[i]);
                break;
            }

            case NpyDtype::UINT16: {
                std::vector<uint16_t> buf(n);
                file.read(reinterpret_cast<char*>(buf.data()), n * sizeof(uint16_t));
                for (size_t i = 0; i < n; ++i)
                    dst[i] = static_cast<float>(buf[i]) / 65535.0f;
                break;
            }

            case NpyDtype::INT32: {
                std::vector<int32_t> buf(n);
                file.read(reinterpret_cast<char*>(buf.data()), n * sizeof(int32_t));
                for (size_t i = 0; i < n; ++i)
                    dst[i] = static_cast<float>(buf[i]);
                break;
            }

            default:
                throw std::runtime_error("[NpyParser] Unsupported dtype for image loading");
        }

        if (!file)
            throw std::runtime_error("[NpyParser] Error reading data — file truncated?");
    }

    // ── Utilitaires ───────────────────────────────────────────────────────────

    static void byteswap32(float* data, size_t n) {
        auto* p = reinterpret_cast<uint32_t*>(data);
        for (size_t i = 0; i < n; ++i) {
            uint32_t v = p[i];
            p[i] = ((v & 0xFF000000) >> 24) |
                   ((v & 0x00FF0000) >>  8) |
                   ((v & 0x0000FF00) <<  8) |
                   ((v & 0x000000FF) << 24);
        }
    }

    static void printHeader(const NpyHeader& h, const std::string& path) {
        auto dtypeStr = [](NpyDtype d) -> std::string {
            switch(d) {
                case NpyDtype::FLOAT32: return "float32";
                case NpyDtype::FLOAT64: return "float64";
                case NpyDtype::INT16:   return "int16";
                case NpyDtype::INT32:   return "int32";
                case NpyDtype::INT64:   return "int64";
                case NpyDtype::UINT8:   return "uint8";
                case NpyDtype::UINT16:  return "uint16";
                default:                return "unknown";
            }
        };

        std::cout << "─────────────────────────────────────────────" << std::endl;
        std::cout << "📂 " << path << std::endl;
        std::cout << "   dtype         : " << dtypeStr(h.dtype) << std::endl;
        std::cout << "   fortran_order : " << (h.fortran_order ? "True ⚠️" : "False ✅") << std::endl;
        std::cout << "   shape         : (";
        for (size_t i = 0; i < h.shape.size(); ++i) {
            std::cout << h.shape[i];
            if (i + 1 < h.shape.size()) std::cout << ", ";
        }
        std::cout << ")" << std::endl;
        std::cout << "   num_elements  : " << h.num_elements << std::endl;
        std::cout << "   data_offset   : " << h.data_offset << " bytes" << std::endl;
        std::cout << "─────────────────────────────────────────────" << std::endl;
    }
};