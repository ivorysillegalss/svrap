import re
content = open('greedy.cpp', 'r', encoding='utf-8').read()
content = re.sub(r'std::to_string\([a-zA-Z_]+\.first\)\s*\+\s*\",\s*\"\s*\+\s*std::to_string\([a-zA-Z_]+\.second\)', lambda m: m.group(0).split('.')[0] + ')', content)
with open('greedy.cpp', 'w', encoding='utf-8') as f:
    f.write(content)
