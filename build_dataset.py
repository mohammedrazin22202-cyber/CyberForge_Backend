"""
Run this script if you update DataSet.xlsx.
It regenerates dataset.json used by the chatbot backend.
"""

import pandas as pd
import json
import os

XLSX_PATH    = os.path.join(os.path.dirname(__file__), 'DataSet.xlsx')
OUTPUT_PATH  = os.path.join(os.path.dirname(__file__), 'dataset.json')

KEY_SENTENCE_COLS = [
    'Core Intent',
    'Recruiter Response 1', 'Recruiter Response 2', 'Recruiter Response 3',
    'Casual Response 1',    'Casual Response 2',    'Casual Response 3',
    'Technical Response 1', 'Technical Response 2', 'Technical Response 3',
    'Short Trigger 1', 'Short Trigger 2',
    'Typo 1', 'Typo 2', 'Fuzzy 1', 'Fuzzy 2',
]

KEYWORD_COLS = (
    [f'Technical {i}' for i in range(1, 16)] +
    [f'Casual {i}'    for i in range(1, 16)] +
    ['Fuzzy 1.1','Fuzzy 2.1'] + [f'Fuzzy {i}' for i in range(3, 11)] +
    ['Typo 1.1','Typo 2.1']   + [f'Typo {i}'  for i in range(3, 11)]
)

df = pd.read_excel(XLSX_PATH)
data = []

for _, row in df.iterrows():
    entry = {
        'file_name':     str(row['File Name']).strip(),
        'key_sentences': [],
        'keywords':      [],
    }
    for col in KEY_SENTENCE_COLS:
        val = str(row.get(col, '')).strip()
        if val and val.lower() != 'nan':
            entry['key_sentences'].append(val.lower())
    for col in KEYWORD_COLS:
        val = str(row.get(col, '')).strip()
        if val and val.lower() != 'nan':
            entry['keywords'].append(val.lower())
    data.append(entry)

with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f'Exported {len(data)} entries to dataset.json')
