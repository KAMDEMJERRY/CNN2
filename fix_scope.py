import sys

with open('src/main.cpp', 'r') as f:
    text = f.read()

block = """enum class Pipeline { CNN2D, CNN3D, CNN3D_SPARSE, CNN3D_ATTN, CNN3D_SPARSE_ATTN, CNN3D_CONVNEXT_DENSE, CNN3D_CONVNEXT_SPARSE };
static Pipeline ACTIVE_PIPELINE = Pipeline::CNN2D;
static std::chrono::high_resolution_clock::time_point START_TIME;"""

if block in text:
    text = text.replace(block + "\n\n", "")
    text = text.replace(block + "\n", "")
    text = text.replace(block, "")
    
    idx = text.find('#include "ModelBuilders.hpp"\n')
    if idx != -1:
        idx += len('#include "ModelBuilders.hpp"\n')
        text = text[:idx] + "\n" + block + "\n" + text[idx:]

with open('src/main.cpp', 'w') as f:
    f.write(text)

