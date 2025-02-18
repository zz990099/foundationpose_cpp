#pragma once

#include <vector>
#include <fstream>
#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;

inline std::vector<fs::path> get_files_in_directory(const fs::path& directory) {
    std::vector<fs::path> files;
    // 检查目录是否存在
    if (!fs::exists(directory) || !fs::is_directory(directory)) {
        std::cerr << "Directory does not exist or is not a directory." << std::endl;
        return files;
    }

    // 递归遍历目录
    for (const auto& entry : fs::recursive_directory_iterator(directory)) {
        if (entry.is_regular_file()) {
            files.push_back(fs::absolute(entry.path()));
        }
    }

    return files;
}