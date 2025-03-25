import socket

def find_free_ports(num_ports, start_port=9000):
    """
    Находит указанное количество свободных портов, начиная с start_port.
    :param num_ports: Количество свободных портов для поиска.
    :param start_port: Начальный порт для поиска.
    :return: Список свободных портов.
    """
    free_ports = []
    current_port = start_port

    while len(free_ports) < num_ports:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                # Попытка связать сокет с портом
                sock.bind(("localhost", current_port))
                free_ports.append(current_port)
            except OSError:
                # Если порт занят, переходим к следующему
                pass
            finally:
                current_port += 1

    return free_ports