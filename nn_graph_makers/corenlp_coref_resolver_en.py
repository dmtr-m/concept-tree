import os

# requires corenlpserver https://nlp.stanford.edu/software/stanford-corenlp-4.5.8.zip
os.environ["CORENLP_HOME"] = (
    "/home/kdemyokhin_1/concept-tree-course-work/concept-tree/nn_graph_makers/stanford-corenlp-4.5.8"
    # "/home/simon/Desktop/concept-tree/concept-tree/nn_graph_makers/stanford-corenlp-4.5.8"
)

from tqdm import tqdm
from stanza.server import CoreNLPClient

from concurrent.futures import ThreadPoolExecutor, as_completed

from nn_graph_makers.find_free_ports import find_free_ports

import time


class CoreferenceResolver:
    def __init__(
        self, endpoint, threads, max_char_length, memory=6, algorithm="statistical"
    ):
        self.endpoint = endpoint
        self.threads = threads
        self.max_char_length = max_char_length
        self.memory_in_gs = memory
        self.algorithm = algorithm
        self.client = None

    def resolve_coreferences(self, text):
        """
        Разрешение анафорических ссылок в тексте.
        :param text: Исходный текст.
        :return: Текст с разрешенными анафорическими ссылками.
        """
        if not self.client:
            raise RuntimeError(
                "CoreNLPClient is not started. Call start_client() first."
            )

        # Аннотируем текст
        document = self.client.annotate(text)

        # Создаем список для хранения замен
        replacements = []

        # Проходим по всем цепочкам кореференции
        for chain in document.corefChain:
            # Получаем основное упоминание (representative mention)
            representative_mention = chain.mention[chain.representative]
            representative_sentence = document.sentence[
                representative_mention.sentenceIndex
            ]
            representative_tokens = representative_sentence.token[
                representative_mention.beginIndex : representative_mention.endIndex
            ]
            representative_text = " ".join(
                token.word for token in representative_tokens
            )

            # Проверяем, является ли основное упоминание сущностью OBJECT (например, существительное)
            is_object = any(
                token.pos.startswith("NN")  # Noun или Proper Noun
                for token in representative_tokens
            )

            # Проходим по всем упоминаниям в цепочке
            for mention in chain.mention:
                if mention.mentionID != representative_mention.mentionID:
                    mention_sentence = document.sentence[mention.sentenceIndex]
                    mention_tokens = mention_sentence.token[
                        mention.beginIndex : mention.endIndex
                    ]
                    mention_text = " ".join(token.word for token in mention_tokens)

                    # Проверяем, является ли упоминание притяжательным местоимением
                    is_possessive_pronoun = any(
                        token.pos == "PRP$"
                        or token.word.lower()
                        in {"my", "your", "his", "her", "its", "our", "their"}
                        for token in mention_tokens
                    )

                    # Если это притяжательное местоимение и основное упоминание — сущность OBJECT
                    if is_possessive_pronoun and is_object:
                        replacement_text = f"{representative_text}'s"
                    else:
                        replacement_text = representative_text

                    # Добавляем замену в список
                    replacements.append(
                        {
                            "sentence_index": mention.sentenceIndex,
                            "start": mention_sentence.token[
                                mention.beginIndex
                            ].beginChar,
                            "end": mention_sentence.token[mention.endIndex - 1].endChar,
                            "replacement": replacement_text,
                        }
                    )

        # Применяем замены в тексте (с конца к началу, чтобы избежать смещения индексов)
        resolved_text = text
        for replacement in sorted(replacements, key=lambda x: x["start"], reverse=True):
            resolved_text = (
                resolved_text[: replacement["start"]]
                + replacement["replacement"]
                + resolved_text[replacement["end"] :]
            )

        return resolved_text

    def process_files(self, input_paths, output_paths, verbose=True):
        """
        Обрабатывает список файлов, разрешая анафорические ссылки.
        :param input_paths: Список путей до входных файлов.
        :param output_paths: Список путей для сохранения обработанных файлов.
        :param algorithm: Алгоритм разрешения анафоры: neural, statistical
        """
        if len(input_paths) != len(output_paths):
            raise ValueError("Количество входных и выходных файлов должно совпадать.")

        # Настройка properties
        properties = {
            "annotators": "tokenize,ssplit,pos,lemma,ner,parse,coref",
            "coref.algorithm": self.algorithm,
        }

        def process_file(input_path, output_path):
            try:
                # Читаем текст из входного файла
                with open(input_path, "r", encoding="utf-8") as file:
                    text = file.read()

                # Разрешаем анафорические ссылки
                resolved_text = self.resolve_coreferences(text)

                # Сохраняем результат в выходной файл
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, "w", encoding="utf-8") as file:
                    file.write(resolved_text)

                if verbose:
                    print(f"Processed: {input_path} -> {output_path}")
                return True
            except Exception as e:
                print(f"Error processing file {input_path}: {e}")
                return False

        # Поднимаем контекстный менеджер только один раз
        with CoreNLPClient(
            properties=properties,
            threads=self.threads,
            endpoint=self.endpoint,
            timeout=600000,  # Увеличиваем таймаут
            be_quiet=True,
            max_char_length=self.max_char_length,
            memory=f"{self.memory_in_gs}G",
        ) as self.client:
            results = []
            try:
                with ThreadPoolExecutor(max_workers=self.threads) as executor:
                    futures = []
                    for i, (input_path, output_path) in enumerate(
                        zip(input_paths, output_paths)
                    ):
                        future = executor.submit(process_file, input_path, output_path)
                        futures.append(future)

                    # Ждем завершени.test_files/я всех задач
                    for future in tqdm(as_completed(futures)):
                        results.append(future.result())
            except Exception as e:
                print(e)

            return results


# Пример использования
if __name__ == "__main__":
    input_paths = [f"{os.getcwd()}/test_files/file{i}.txt" for i in range(1, 100 + 1)]

    # Список выходных файлов
    output_paths = [
        f"{os.getcwd()}/test_files/file{i}_processed.txt" for i in range(1, 100 + 1)
    ]

    # Общее количество потоков
    total_threads = 16
    num_servers = 2

    for path in output_paths:
        if os.path.exists(path):
            os.remove(path)

    # start_time = time.time()
    # # Обрабатываем файлы
    # process_files_parallel(
    #     total_threads,
    #     num_servers,
    #     input_paths,
    #     output_paths,
    #     algorithm="neural",
    #     verbose=False,
    # )
    # end_time = time.time()
    # print("Elapsed time:", end_time - start_time)

    # for path in output_paths:
    #     if os.path.exists(path):
    #         os.remove(path)

    # start_time = time.time()
    # resolver = CoreferenceResolver(
    #     properties=None,
    #     endpoint=f"http://localhost:{find_free_ports(1)[0]}",
    #     threads=total_threads,
    #     max_char_length=10000,
    #     memory=2,
    # )
    # resolver.process_files(input_paths, output_paths, "neural", verbose=False)
    # end_time = time.time()
    # print("Elapsed time:", end_time - start_time)
