import re

with open('src/main.cpp', 'r') as f:
    text = f.read()

pat = re.compile(r'enum class Pipeline \{.*?\};\n*(static Pipeline ACTIVE_PIPELINE = Pipeline::CNN2D;\nstatic std::chrono::high_resolution_clock::time_point START_TIME;)\n*', re.DOTALL)

match = pat.search(text)
if match:
    decl = match.group(0)
    text = text.replace(decl, "")
    
    idx = text.find('#include "ModelBuilders.hpp"\n')
    if idx != -1:
        idx += len('#include "ModelBuilders.hpp"\n')
        text = text[:idx] + "\n" + decl + "\n" + text[idx:]
        with open('src/main.cpp', 'w') as f:
            f.write(text)
        print("Fixed scope")
    else:
        print("Could not find #include ModelBuilders.hpp")
else:
    print("Could not find block")
