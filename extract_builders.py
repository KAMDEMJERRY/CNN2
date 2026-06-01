import os

with open('src/main.cpp', 'r') as f:
    lines = f.readlines()

ranges = [
    (11, 54),     # Configs
    (59, 80),     # 2D
    (141, 164),   # 3D
    (228, 258),   # 3D Sparse
    (346, 403),   # 3D Attn
    (503, 549),   # 3D Sparse Attn
    (642, 669),   # ConvNeXt Dense
    (736, 761)    # ConvNeXt Sparse
]

extracted = []
lines_to_keep = set(range(1, len(lines) + 1))

# For configs
config_lines = lines[10:54]
extracted.append("".join(config_lines))
for i in range(11, 55):
    lines_to_keep.remove(i)

for start, end in ranges[1:]:
    func_lines = lines[start-1:end]
    func_lines[0] = func_lines[0].replace('static CNN', 'inline CNN')
    extracted.append("".join(func_lines))
    for i in range(start, end + 1):
        lines_to_keep.remove(i)

# Create include/ModelBuilders.hpp
header_content = """#pragma once

#include "CNNLIB.hpp"
#include <vector>
#include <memory>
#include <chrono>

""" + "\n".join(extracted)

with open('include/ModelBuilders.hpp', 'w') as f:
    f.write(header_content)

new_main_lines = []
includes_added = False
for i, line in enumerate(lines):
    if (i + 1) in lines_to_keep:
        new_main_lines.append(line)
        if not includes_added and line.strip() == '#include "CNNLIB.hpp"':
            new_main_lines.append('#include "ModelBuilders.hpp"\n')
            includes_added = True

with open('src/main.cpp', 'w') as f:
    f.writelines(new_main_lines)
