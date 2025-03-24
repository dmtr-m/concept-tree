# %%
from bs4 import BeautifulSoup
from concurrent.futures import ProcessPoolExecutor
import re
import os


# %%
def clean_html(path_to_input: str, path_to_output: str) -> str:
    # Open html file with text
    with open(path_to_input, "r", encoding="utf-8") as file:
        soup = BeautifulSoup(file, "lxml")

    # --------- CLEANUP: REMOVE unwanted elements ---------
    # Find all <math> elements (MathML equations)
    math_tags = soup.find_all("math")

    # Replace each <math> with [[[formula]]]
    for tag in math_tags:
        tag.replace_with(" [[[formula]]] ")

    # Find all <cite> elements (MathML equations)
    cite_tags = soup.find_all("cite")

    # Replace each <cite> with [[[cite]]]
    for tag in cite_tags:
        tag.replace_with(" [[[cite]]] ")

    # --------- BODY TEXT ---------
    # You can extract body text from paragraphs or specific divs
    paragraphs = soup.find_all(["p", "div"], class_="ltx_para")
    text_parts = [
        p.get_text(separator=" ", strip=True)
        for p in paragraphs
        if p.get_text(strip=True)
    ]

    # Combine into full text
    text = "\n\n".join(text_parts)

    text = re.sub(
        r"Eqs?\.\s*\(\s*\d+\s*\)(?:\s*[–-]\s*\(\s*\d+\s*\))?",
        "[[[Equation Reference]]]",
        text,
    )
    text = re.sub(
        r"Figs?\.\s*(?:S)?\d+(?:\s*(?:\([a-z]\)|\([a-z]\)-\([a-z]\)))?(?:\s*(?:and|–|-)\s*(?:S)?\d+(?:\s*\([a-z]\))?)?",
        "[[[Figure Reference]]]",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"(?:Sec(?:tion)?\.?|Appendix)\s+(?:[A-Z]+|\d+)(?:\.(?:\d+|[A-Z]+)){0,3}(?:\s+in\s+\[?SI\]?)?",
        "[[[Sequence Reference]]]",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\(\s*\d+\s*\)", "", text)

    os.makedirs(
        os.path.dirname(path_to_output), exist_ok=True
    )  # Create output directory if it doesn't exist
    # Write prcoessed text to file
    with open(path_to_output, "w", encoding="utf-8") as out_f:
        out_f.write(text)


# %%
from pathlib import Path

# pathlist = Path("data/").glob('*.html')
# for path in pathlist:
#     path_in_str = str(path)
#     output_file = path_in_str[:-5] + ".txt"
#     clean_html(path_in_str, output_file)


# %%
def process_files_in_parallel():
    base_input_dir = Path(
        "/home/kdemyokhin_1/concept-tree-course-work/articles_raw/arxiv-html-cs/"
    )  # Исходная директория с входными файлами
    base_output_dir = Path(
        "/home/kdemyokhin_1/concept-tree-course-work/articles_parsed/arxiv-txt-cs/"
    )  # Целевая директория

    tasks = []

    # Проходим по всем входным файлам
    for path in base_input_dir.rglob(
        "*.html"
    ):  # rglob рекурсивно находит все HTML-файлы
        relative_path = path.relative_to(
            base_input_dir
        )  # Получаем относительный путь от базовой директории
        output_path = base_output_dir / relative_path.with_suffix(
            ".txt"
        )  # Меняем расширение на .txt
        tasks.append((str(path), str(output_path)))

    # Используем ProcessPoolExecutor для параллельной обработки
    with ProcessPoolExecutor() as executor:
        # Запускаем задачи в пуле процессов
        executor.map(lambda task: clean_html(task[0], task[1]), tasks)


if __name__ == "__main__":
    process_files_in_parallel()
