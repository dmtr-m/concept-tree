import os
from tqdm import tqdm

# requires corenlpserver https://nlp.stanford.edu/software/stanford-corenlp-4.5.8.zip
os.environ["CORENLP_HOME"] = (
    "/home/kdemyokhin_1/concept-tree-course-work/concept-tree/nn_graph_makers/stanford-corenlp-4.5.8"
)

# from directed_graph.graph import Graph, visualize_graph_with_equivalent_elements

import pandas as pd
from stanza.server import CoreNLPClient
from concurrent.futures import ThreadPoolExecutor, as_completed


class OpenIETriplesMaker:
    def __init__(
        self, endpoint, max_length=5, max_char_count=100, memory="4G", threads=None
    ):
        """
        Инициализация объекта для извлечения триплетов.
        :param endpoint: Адрес сервера CoreNLP (например, "http://localhost:9000").
        :param max_length: Максимальная длина элементов триплета (в словах).
        :param max_char_count: Максимальное количество символов в элементе триплета.
        :param memory: Объем памяти для CoreNLPClient (строка, например, "4G").
        :param threads: Количество потоков для CoreNLPClient. Если None, используется количество ядер CPU.
        """
        self.endpoint = endpoint
        self.max_length = max_length
        self.max_char_count = max_char_count
        self.memory = memory  # Память передается как строка (например, "4G")
        self.threads = threads if threads is not None else os.cpu_count()
        print(f"Using {self.threads} threads and memory: {self.memory}")

        # Настройка properties
        self.properties = {
            "annotators": "tokenize,ssplit,pos,lemma,ner,openie",
            "openie.triple.strict": "true",
            "openie.splitConjuncts": "true",
        }

        # Инициализация CoreNLPClient (без автоматического запуска сервера)
        self.client = CoreNLPClient(
            annotators=self.properties["annotators"],
            threads=self.threads,
            memory=self.memory,  # Передаем память как строку
            max_char_length=self.max_char_count,  # Передаем максимальную длину символов
            timeout=60000,
            be_quiet=True,
            start_server=False,  # Явно отключаем автозапуск сервера
            endpoint=self.endpoint,  # Указываем endpoint
        )
        print("CoreNLPClient initialized. Server must be started manually.")

    def __start_server(self):
        """
        Явный запуск сервера CoreNLP.
        """
        self.client.start()
        print("CoreNLP server started.")

    def __stop_server(self):
        """
        Остановка сервера CoreNLP.
        """
        self.client.stop()
        print("CoreNLP server stopped.")

    def extract_triples(self, text):
        """
        Извлечение и фильтрация триплетов из текста.
        :param text: Исходный текст.
        :return: Список отфильтрованных триплетов с POS для relation.
        """
        triples = []

        # Аннотация текста
        document = self.client.annotate(text)

        # Извлечение и фильтрация триплетов
        for sentence in document.sentence:
            # Создаем карты для лемматизации и POS
            word_to_lemma = {
                token.word.lower(): token.lemma.lower() for token in sentence.token
            }
            word_to_pos = {token.word.lower(): token.pos for token in sentence.token}

            for triple in sentence.openieTriple:
                # Лемматизация элементов триплета
                subject_lemma = " ".join(
                    [
                        word_to_lemma.get(word.lower(), word.lower())
                        for word in triple.subject.split()
                    ]
                )
                relation_lemma = " ".join(
                    [
                        word_to_lemma.get(word.lower(), word.lower())
                        for word in triple.relation.split()
                    ]
                )
                object_lemma = " ".join(
                    [
                        word_to_lemma.get(word.lower(), word.lower())
                        for word in triple.object.split()
                    ]
                )

                # Определение POS для relation с учетом контекста
                relation_pos = []
                for word in triple.relation.split():
                    if word.lower() in word_to_pos:
                        relation_pos.append(word_to_pos[word.lower()])
                    else:
                        relation_pos.append("UNKNOWN")
                relation_pos = " ".join(relation_pos)

                # Проверка длины элементов перед добавлением
                if (
                    len(subject_lemma.split()) <= self.max_length
                    and len(relation_lemma.split()) <= self.max_length
                    and len(object_lemma.split()) <= self.max_length
                ):
                    triples.append(
                        {
                            "subject": subject_lemma,
                            "relation": relation_lemma,
                            "relation_pos": relation_pos,  # Добавляем POS для relation
                            "object": object_lemma,
                        }
                    )

        return triples

    def process_file(self, input_file, output_file):
        """
        Обработка одного файла: чтение текста, извлечение триплетов и сохранение результатов.
        :param input_file: Путь к входному файлу.
        :param output_file: Путь к выходному файлу.
        """
        print(f"Processing file: {input_file}")
        # Чтение текста из входного файла
        with open(input_file, "r", encoding="utf-8") as f:
            text = f.read()

        # Извлечение триплетов
        triples = self.extract_triples(text)

        # Сохранение триплетов в выходной файл
        df = pd.DataFrame(
            triples, columns=["subject", "relation", "relation_pos", "object"]
        )
        df.to_pickle(output_file, compression="gzip")
        print(f"Saved triples to: {output_file}")

    def process_files(self, input_files, output_files):
        """
        Обработка списка входных файлов и сохранение результатов в выходные файлы с использованием многопоточности.
        :param input_files: Список путей к входным файлам.
        :param output_files: Список путей к выходным файлам (должен быть того же размера, что и input_files).
        """
        if len(input_files) != len(output_files):
            raise ValueError("Количество входных и выходных файлов должно совпадать.")

        self.__start_server()

        # Используем ThreadPoolExecutor для многопоточной обработки
        with ThreadPoolExecutor(max_workers=self.threads) as executor:
            futures = [
                executor.submit(self.process_file, input_file, output_file)
                for input_file, output_file in zip(input_files, output_files)
            ]

            # Ожидаем завершения всех задач
            for future in tqdm(as_completed(futures)):
                try:
                    future.result()  # Проверяем на наличие исключений
                except Exception as e:
                    print(f"Error processing file: {e}")

        self.__stop_server()
