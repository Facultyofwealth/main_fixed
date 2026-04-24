import json
import re
import os

# Paths
source_path = 'kjv.json/kjv-master/json/verses-1769.json'
output_path = 'kjv.json'

# Load flat verses
with open(source_path, 'r', encoding='utf-8') as f:
    flat_verses = json.load(f)

print(f'Loaded {len(flat_verses)} flat verses')

# Parse to nested: {'Genesis': {'1': [{'verse': 1, 'text': '...'}], ...}}
bible = {}
for ref, text in flat_verses.items():
    if not text.strip():
        continue
    # Parse "Genesis 1:1"
    match = re.match(r'^(.+?)\s+(\d+):(\d+)$', ref.strip())
    if match:
        book = match.group(1)
        ch = match.group(2)
        v = int(match.group(3))
        if book not in bible:
            bible[book] = {}
        if ch not in bible[book]:
            bible[book][ch] = []
        bible[book][ch].append({'verse': v, 'text': text})

# Save
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(bible, f, indent=2, ensure_ascii=False)

print(f'Created {output_path} with {sum(len(chs) for book in bible.values() for chs in book.values())} verses')
print('Books:', list(bible.keys())[:5], '...')

