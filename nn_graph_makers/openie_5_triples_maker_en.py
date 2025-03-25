import subprocess
import socket
import time
import requests


class OpenIE5Processor:
    def __init__(
        self,
        server_url="http://localhost:8080",
        jar_path="target/openie-assembly.jar",
        memory="4g",
    ):
        """
        Инициализация процессора OpenIE-5.
        :param server_url: URL сервера OpenIE-5.    
        :param jar_path: Путь к JAR-файлу сервера OpenIE-5.
        :param memory: Максимальный объем памяти для сервера (например, "4g").
        """
        self.server_url = server_url
        self.jar_path = jar_path
        self.memory = memory
        self.server_process = None

    def is_port_open(self, port, host="localhost", timeout=1):
        """
        Проверка, открыт ли порт.
        :param port: Номер порта.
        :param host: Хост (по умолчанию localhost).
        :param timeout: Таймаут для проверки (в секундах).
        :return: True, если порт открыт, иначе False.
        """
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except (socket.timeout, ConnectionRefusedError):
            return False

    def start_server(self):
        """
        Запуск сервера OpenIE-5.
        """
        if self.server_process is not None:
            print("Сервер уже запущен.")
            return

        try:
            print(f"Запуск сервера OpenIE-5 на {self.server_url}...")
            self.server_process = subprocess.Popen(
                [
                    "java",
                    f"-Xmx{self.memory}",
                    "-jar",
                    self.jar_path,
                    "--httpPort=8080",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            # Проверка доступности порта
            port = int(self.server_url.split(":")[-1])  # Извлекаем номер порта из URL
            while not self.is_port_open(port):
                print("Ожидание запуска сервера...")
                time.sleep(1)  # Проверяем каждую секунду

            print("Сервер успешно запущен.")
        except Exception as e:
            print(f"Ошибка при запуске сервера: {e}")
            self.stop_server()
            raise

    def stop_server(self):
        """
        Остановка сервера OpenIE-5.
        """
        if self.server_process is None:
            print("Сервер не запущен.")
            return

        try:
            print("Остановка сервера OpenIE-5...")
            self.server_process.terminate()
            self.server_process.wait()  # Дожидаемся завершения процесса
            print("Сервер успешно остановлен.")
        except Exception as e:
            print(f"Ошибка при остановке сервера: {e}")
        finally:
            self.server_process = None

    def extract_triples(self, text):
        """
        Извлечение триплетов с помощью OpenIE-5.
        :param text: Исходный текст.
        :return: Список триплетов.
        """
        if self.server_process is None:
            raise RuntimeError(
                "Сервер OpenIE-5 не запущен. Вызовите метод start_server()."
            )

        # Отправка POST-запроса к серверу OpenIE-5
        url = f"{self.server_url}/getExtraction"
        headers = {"Content-Type": "application/json"}
        data = {"text": text}

        try:
            response = requests.post(url, json=data)
            if response.status_code != 200:
                raise Exception(f"Ошибка при извлечении триплетов: {response.text}")

            # Парсинг JSON-ответа
            extractions = response.json()
            triples = []

            for extraction in extractions:
                subject = extraction["subject"]
                relation = extraction["relation"]
                obj = extraction["object"]
                triples.append(
                    {"subject": subject, "relation": relation, "object": obj}
                )

            return triples

        except Exception as e:
            print(f"Ошибка при взаимодействии с сервером: {e}")
            return []

    def __enter__(self):
        """
        Метод, вызываемый при входе в контекстный менеджер.
        """
        self.start_server()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Метод, вызываемый при выходе из контекстного менеджера.
        """
        self.stop_server()


# Пример использования
if __name__ == "__main__":
    # Создание экземпляра класса через контекстный менеджер
    with OpenIE5Processor() as processor:
        # Исходный текст
        text = (
            "Albert Einstein developed the theory of relativity. "
            "He was a brilliant physicist."
        )

        # Извлечение триплетов
        triples = processor.extract_triples(text)

        # Вывод результатов
        print("Извлеченные триплеты:")
        for triple in triples:
            print(triple)
