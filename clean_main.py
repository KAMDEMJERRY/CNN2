import sys

with open('src/main.cpp', 'r') as f:
    text = f.read()

def remove_funcs_by_prefix(text, prefix):
    while True:
        idx = text.find(prefix)
        if idx == -1:
            break
        # find the opening brace
        brace_idx = text.find('{', idx)
        if brace_idx == -1:
            break
        
        brace_count = 1
        end_idx = brace_idx + 1
        while brace_count > 0 and end_idx < len(text):
            if text[end_idx] == '{':
                brace_count += 1
            elif text[end_idx] == '}':
                brace_count -= 1
            end_idx += 1
        
        if end_idx < len(text) and text[end_idx] == '\n':
            end_idx += 1
        text = text[:idx] + text[end_idx:]
    return text

text = remove_funcs_by_prefix(text, "static CNN buildModel")
text = remove_funcs_by_prefix(text, "CNN buildModel")

with open('src/main.cpp', 'w') as f:
    f.write(text)

print("cleaned")
