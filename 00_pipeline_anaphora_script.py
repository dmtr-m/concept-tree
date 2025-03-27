# %%
import os
from pathlib import Path

subdir='cs.AI'

input_directory = Path("/home/kdemyokhin_1/concept-tree-course-work/articles_parsed/arxiv-txt-cs/"+subdir)
output_directory = Path("/home/kdemyokhin_1/concept-tree-course-work/articles_anaphora_resolved/arxiv-txt-cs/"+subdir)

# %%
import spacy
import coreferee

nlp = spacy.load('en_core_web_trf')
nlp.add_pipe('coreferee')


def resolve_coreference(text):
    doc = nlp(text)
    refined_text = []
    for i in range(len(doc)):
        res = doc._.coref_chains.resolve(doc[i])
        if res is None:
            refined_text += [doc[i]]
        else:
            refined_text += res
    
    return ' '.join([d.text for d in refined_text])


def process_file(input_path, output_path):
    with open(input_path, "r") as f:
        lines = f.readlines()
    
    resolved_lines = []
    for line in lines:
        resolved_lines.append(resolve_coreference(line))

    with open(output_path, 'w') as f:
        for line in resolved_lines:
            f.write(f"{line}")

# %%
# Получаем список всех txt файлов рекурсивно (включая поддиректории)
input_files = list(input_directory.rglob("*.txt"))

# Формируем список выходных файлов, сохраняя структуру поддиректорий
output_files = []
for file in input_files:
    # Вычисляем относительный путь файла относительно input_directory
    relative_path = file.relative_to(input_directory)
    # Формируем путь к файлу в выходной директории
    out_file = output_directory / relative_path
    # Создаем директорию, если её ещё нет
    out_file.parent.mkdir(parents=True, exist_ok=True)
    output_files.append(out_file)

# %%
import warnings
warnings.filterwarnings('ignore')

# %%
from tqdm import tqdm

for input_path, output_path in tqdm(zip(input_files, output_files), total=len(input_files)):
    if os.path.exists(output_path):
        continue
    process_file(input_path, output_path)