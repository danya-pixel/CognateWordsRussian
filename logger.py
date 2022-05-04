import os
from pathlib import Path

base_dir = Path('logs')

if not base_dir.exists():
    os.makedirs(base_dir)

root_file = base_dir / 'root.csv'
cognate_file = base_dir / 'cognate.csv'


def save_root(word, correct_word):
    with open(root_file, 'a') as f:
        f.write(f'{word};{correct_word}\n')


def save_cognate(word1, word2, status: bool):
    with open(cognate_file, 'a') as f:
        f.write(f'{word1};{word2};{1 if status else 0}\n')
