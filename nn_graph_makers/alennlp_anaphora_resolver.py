import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from allennlp.predictors.predictor import Predictor
from tqdm import tqdm

# Инициализация модели AllenNLP для разрешения кореференций
def load_coref_model():
    print("Loading coreference resolution model...")
    model_url = "https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2021.03.10.tar.gz"
    predictor = Predictor.from_path(model_url)
    return predictor

# Функция для разрешения анафоры в тексте
def resolve_anaphora(text, predictor):
    try:
        result = predictor.predict(document=text)
        resolved_text = text
        clusters = result["clusters"]
        for cluster in clusters:
            main_mention = " ".join(result["document"][cluster[0][0]:cluster[0][1] + 1])
            for mention in cluster[1:]:
                mention_text = " ".join(result["document"][mention[0]:mention[1] + 1])
                resolved_text = resolved_text.replace(mention_text, main_mention)
        return resolved_text
    except Exception as e:
        print(f"Error resolving anaphora: {e}")
        return text  # Возвращаем исходный текст в случае ошибки

# Функция для обработки одного файла
def process_file(input_path, output_path, predictor):
    try:
        # Чтение текста из входного файла
        with open(input_path, "r", encoding="utf-8") as file:
            text = file.read()

        # Разрешение анафоры
        resolved_text = resolve_anaphora(text, predictor)

        # Сохранение результата в выходной файл
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as file:
            file.write(resolved_text)

        print(f"Processed: {input_path} -> {output_path}")
    except Exception as e:
        print(f"Error processing {input_path}: {e}")

# Основная функция для параллельной обработки файлов
def process_files_multithreaded(
    workers_count, input_paths, output_paths, predictor
):
    """
    Обрабатывает файлы многопоточно, разрешая анафору в каждом файле.
    
    :param workers_count: Количество потоков для обработки.
    :param input_paths: Список путей до входных файлов.
    :param output_paths: Список путей до выходных файлов.
    """
    if len(input_paths) != len(output_paths):
        raise ValueError("Количество входных файлов должно совпадать с количеством выходных файлов.")

    # Используем ThreadPoolExecutor для многопоточной обработки
    with ThreadPoolExecutor(max_workers=workers_count) as executor:
        futures = []
        for input_path, output_path in zip(input_paths, output_paths):
            future = executor.submit(process_file, input_path, output_path, predictor)
            futures.append(future)

        # Ожидаем завершения всех задач с прогресс-баром
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing files"):
            try:
                future.result()
            except Exception as e:
                print(f"Error during processing: {e}")