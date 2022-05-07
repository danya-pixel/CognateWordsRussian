import os
from pathlib import Path
import csv

base_dir = Path('logs')

if not base_dir.exists():
    os.makedirs(base_dir)

root_file = base_dir / 'root.csv'
cognate_file = base_dir / 'cognate.csv'


def save_root(word, correct_word):
    with open(root_file, 'a') as f:
        f.write(f'{word};{correct_word}\n')


def save_cognate(word_1, word_2, siamese_prob, heurisic_predict, status: bool):

    log = [word_1, word_2, siamese_prob, heurisic_predict, status]
    with open(cognate_file, 'a') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(log)