import json
import re

# Full absolute paths from CWD PythonProjects
source_path = 'main_fixed/kjv.json/kjv-master/json/verses-1769.json'
output_path = 'main_fixed/kjv.json'

print('Parsing KJV...')

with open(source_path, 'r', encoding='utf-8') as f:
    flat_verses = json.load(f)

print(f'Loaded {len(flat_verses)} verses')

# Flat list matching load_sample_verses()
verses = []
for ref, text in flat_verses.items():
    text = text.strip()
    if text:
        verses.append({'ref': ref.strip(), 'text': text})

print(f'Cleaned {len(verses)} verses')

with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(verses, f, indent=1)  # compact for size

print('Created main_fixed/kjv.json')
print('Sample:', verses[0])

