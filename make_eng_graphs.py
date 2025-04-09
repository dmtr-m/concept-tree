import spacy
import spacy.cli
from nltk.stem import WordNetLemmatizer
from graph.higher_dim_graph import Graph
from graph.graph import visualize_graph
from graph.edge import Edge
from graph.vertex import Vertex

# Загружаем модель Spacy
# spacy.cli.download("en_core_web_lg")
nlp = spacy.load("en_core_web_lg")
lemmatizer = WordNetLemmatizer()

def get_syntactic_relations(doc):
    """
    Извлекает синтаксические связи (именные группы, включая прилагательные и артикли) из текста.
    Обрабатывает сочетания глагола с предлогом как единое ребро.
    Обрабатывает сочинение (conj), создавая дополнительные связи.
    """
    chunks = []  # Список для хранения именных групп
    relations = []  # Список для хранения связей (субъект, глагол, объект)
    subjects = {}  # Словарь для хранения подлежащих
    conjunctions = {}  # Словарь для хранения связей conj (сочинение)
    chunk_to_text = {}  # Связываем root токены с текстом именных групп

    # Добавляем именные группы (NOUN CHUNKS) и нормализуем их
    for chunk in doc.noun_chunks:
        normalized_chunk = ' '.join([lemmatizer.lemmatize(token.text.lower(), pos='n') for token in chunk if token.text.lower() not in ['the', 'a', 'an']])
        chunks.append((chunk.start_char, chunk.end_char, chunk, normalized_chunk, chunk.root.head, chunk.root.dep_))
        chunk_to_text[chunk.root] = normalized_chunk  # Связываем root токен с нормализованным текстом

    # Обрабатываем сочинение (conj)
    for token in doc:
        if token.dep_ == "conj" and token.head in chunk_to_text:
            head_text = chunk_to_text[token.head]
            conj_text = chunk_to_text.get(token, None)  # Используем уже обработанный chunk

            if head_text and conj_text:
                conjunctions.setdefault(head_text, []).append(conj_text)

    # Обрабатываем союзы для существительных
    for token in doc:
        if token.dep_ == "conj" and token.head.pos_ == "NOUN":
            head_text = chunk_to_text.get(token.head, token.head.text.lower())
            conj_text = chunk_to_text.get(token, token.text.lower())

            # Добавляем рёбра для сочинённых существительных
            relations.append((head_text, "and", conj_text))
            print(f"Adding conjunction edge: {head_text} --[and]--> {conj_text}")

    # Добавляем подлежащие
    for chunk in chunks:
        if chunk[5] == 'nsubj':
            subject_text = chunk_to_text.get(chunk[2].root, chunk[3])  # Используем нормализованный текст
            subjects.setdefault(chunk[4], []).append(subject_text)

            # Добавляем conj-подлежащие
            if subject_text in conjunctions:
                subjects[chunk[4]].extend(conjunctions[subject_text])

    # Добавляем связи для глаголов и предлогов
    for i, chunk in enumerate(chunks):
        # Связи с глаголами
        if chunk[4].pos_ == 'VERB' and chunk[5] != 'nsubj':
            subject_list = subjects.get(chunk[4], [])
            object_text = chunk_to_text.get(chunk[2].root, chunk[3])  # Используем нормализованный текст

            for subject in subject_list:
                relations.append((subject, chunk[4].text, object_text))

                # Добавляем conj-объекты
                if object_text in conjunctions:
                    for conj in conjunctions[object_text]:
                        relations.append((subject, chunk[4].text, conj))

        # Обрабатываем сочетания глаголов и предлогов как единое ребро
        if chunk[4].pos_ == 'VERB' and i + 1 < len(chunks):
            next_chunk = chunks[i + 1]
            if next_chunk[4].pos_ == 'ADP':  # Если следующий элемент - предлог
                subject_list = subjects.get(chunk[4], [])
                relation_text = f"{chunk[4].text} {next_chunk[4].text}"
                object_text = chunk_to_text.get(next_chunk[2].root, next_chunk[3])

                for subject in subject_list:
                    relations.append((subject, relation_text, object_text))

                    # Добавляем conj-объекты
                    if object_text in conjunctions:
                        for conj in conjunctions[object_text]:
                            relations.append((subject, relation_text, conj))

    # Добавляем связи для предлогов и объектов
    for token in doc:
        if token.dep_ == "prep" and token.head.pos_ == "NOUN":
            prep_text = token.text
            object_text = None

            # Ищем объект предлога (pobj)
            for child in token.children:
                if child.dep_ == "pobj":
                    object_text = chunk_to_text.get(child, child.text.lower())

            if object_text:
                head_text = chunk_to_text.get(token.head, token.head.text.lower())  # Нормализуем ключ
                subject_list = [head_text]

                # Добавляем все сочинённые существительные
                if head_text in conjunctions:
                    subject_list.extend(conjunctions[head_text])

                # Добавляем рёбра для каждого существительного в subject_list
                for subject in subject_list:
                    relations.append((subject, prep_text, object_text))
                    print(f"Adding edge: {subject} --[{prep_text}]--> {object_text}")

    return relations


def find_nearest_vertex(token, chunk_to_vertex):
    """
    Ищет ближайшую вершину для токена, поднимаясь вверх по дереву зависимостей.
    """
    visited = set()
    while token and token not in chunk_to_vertex:
        if token in visited or token.head == token:
            return None  # Защита от бесконечного цикла
        visited.add(token)
        token = token.head  # Поднимаемся выше в дереве
    return chunk_to_vertex.get(token)


def process_token(token, graph, chunk_to_vertex, conjunctions):
    """
    Обрабатывает токен, добавляя рёбра между существующими вершинами (именными группами).
    """
    for child in token.children:
        head_vertex = find_nearest_vertex(token, chunk_to_vertex)
        child_vertex = find_nearest_vertex(child, chunk_to_vertex)

        if head_vertex and child_vertex and token.pos_ == "ADP":  # Обрабатываем только предлоги
            edge_tuples = [(head_vertex.concept, child_vertex.concept, token.text)]
            print(f"Processing preposition edge: {head_vertex.concept} --[{token.text}]--> {child_vertex.concept}")

            # Добавляем союзы между существительными
            if head_vertex.concept in conjunctions:
                conj_list = conjunctions[head_vertex.concept]
                print(f"Found conjunctions for {head_vertex.concept}: {conj_list}")
                for conj in conj_list:
                    graph.add_edge(conj, child_vertex.concept, token.text, 1, 0)

            # Проверяем, есть ли у существительного сочинённые элементы
            if head_vertex.concept in conjunctions:
                conj_list = conjunctions[head_vertex.concept]
                print(f"Found conjunctions for {head_vertex.concept}: {conj_list}")
                for conj in conj_list:
                    edge_tuples.append((conj, child_vertex.concept, token.text))

            # Проверяем наличие рёбер и добавляем только новые
            existing_edges = {(e.agent_1, e.agent_2, e.meaning) for e in graph.edges}
            for edge_tuple in edge_tuples:
                if edge_tuple not in existing_edges:
                    print(f"Adding edge: {edge_tuple[0]} --[{edge_tuple[2]}]--> {edge_tuple[1]}")
                    graph.add_edge(edge_tuple[0], edge_tuple[1], edge_tuple[2], 1, 0)
                else:
                    print(f"Edge already exists: {edge_tuple[0]} --[{edge_tuple[2]}]--> {edge_tuple[1]}")

        process_token(child, graph, chunk_to_vertex, conjunctions)

# Тестирование на предложении
# text = """How are living organisms different from inanimate matter? There are obvious answers in terms
# of the chemical composition and structure, but when it comes to the central processes in the evolution
# of life, the distinction is far less obvious. In the tradition of Darwin-Wallace, it is tempting to
# posit that life is defined by evolution through the survival of the fittest. However, the
# uniqueness of this process to life could be questioned because the entire history of the universe
# consists of changes where the most stable structures survive. The process of
# replication itself is not truly unique to biology either: crystals do replicate. On the macroscopic
# scales of space and time, however, life clearly is a distinct phenomenon. To objectively define
# the features that distinguish life from other phenomena that occur mostly in the universe, it seems
# important to examine the key processes of biological evolution within the framework of
# theoretical physics."""
text = 'The cat and the dog eat fish and meat. The elephant and the bear on the mat. The monkey on the table and under the roof.'
doc = nlp(text)

graph1 = Graph()
chunk_to_vertex = {}
conjunctions = {}

# Добавляем вершины только для именных групп (исключаем глаголы и наречия)
for chunk in doc.noun_chunks:
    vertex = Vertex(chunk.text.lower(), [chunk.text.lower()])
    
    # Проверяем, существует ли вершина с таким концептом
    if vertex.concept not in graph1.vertices:
        print(f"Adding vertex: {vertex.concept}")
        graph1.add_vertex(vertex.concept, vertex.words_of_concept)
    
    # Привязываем root токен к вершине
    chunk_to_vertex[chunk.root.lower] = vertex

# Получаем синтаксические связи с помощью get_syntactic_relations
relations = get_syntactic_relations(doc)
# Добавляем рёбра в граф
for relation in relations:
    # Предполагаем, что каждая связь состоит из трёх элементов: (subject, verb, object)
    subject, verb, object_text = relation
    # Проверяем, существуют ли вершины для этих элементов, и добавляем рёбра
    if subject not in graph1.vertices:
        graph1.add_vertex(subject, [subject])
    if object_text not in graph1.vertices:
        graph1.add_vertex(object_text, [object_text])
    
    # Добавляем рёбра между вершинами
    print(f"Adding edge: {subject} --[{verb}]--> {object_text}")
    graph1.add_edge(subject, object_text, verb, 1, 0)

# чистим граф от пустых вершин
graph_res = Graph()
for concept in graph1.vertices.keys():
    if graph1.get_vertex_edges(concept):
        graph_res.add_vertex(concept)
for edge in graph1.edges:
    graph_res.add_edge(edge.agent_1, edge.agent_2, edge.meaning, 0, 0)

# Выводим граф
print("Graph structure:")
print("\nEdges:")
for concept in graph_res.vertices.keys():
    print(f"Concept: {concept}")
    print(graph_res.get_vertex_edges(concept))

print("\nVertices:")
for concept in graph_res.vertices.keys():
    print(graph_res.vertices[concept])

visualize_graph(graph_res)

# TODO resolve anaphora, adding conj edge, process conjunction