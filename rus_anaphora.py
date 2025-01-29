import spacy
from spacy.tokens import Span
from spacy import displacy

from pymorphy3 import MorphAnalyzer

import types
from types import NoneType


nlp = spacy.load("ru_core_news_lg")
morph = MorphAnalyzer()


def head_in_named_entity(doc, span): # на вход подаются документ и именованная сущность, которая в нём содержится
    span_parts = span.text.split(' ')
    head = None
    heads = [[], []] # [[token], [head]]
    for span_part in span_parts:
        for token in doc: # перебираем токены, потому что именно они, в отличие от строк, содержат всю информацию
            if span_part == token.text:
                heads[0].append(token)
                heads[1].append(token.head)
    for i in range(len(span_parts)):
        if heads[1][i] not in heads[0]: # вершиной является то, что не зависит от других слов, входящих в именованную сущность
            head = heads[0][i]
    return head, [_.split('=')[1] for _ in str(head.morph).split('|')], head.head, head.dep_, heads

def normalize_noun_phrase(doc, np): # на вход подаётся документ и именная группа
    head, morphology, parent, dep, np_parts = head_in_named_entity(doc, np)
    ana = morph.parse(head.text)[0]
    res = ''
    for i, np_part in enumerate(np_parts[0]):
        np_part_head = np_parts[1][i]
        if np_part == head:
            np_part = ana.normal_form
        else:
            np_part = morph.parse(np_part.text)[0]
            pos = str(np_part.tag).split(',')[0].split(' ')[0]
            if pos == 'ADJF' and np_part_head == head:
                gender, number = str(ana.normalized.tag).split(',')[2].split()
                np_part = np_part.inflect({gender, 'nomn'})[0]
            else:
                np_part = np_part.word
        res += np_part + ' '
    return res.strip()

def get_syntactic_relations(doc):
    chunks = [] # [((индекс первого символа, индекс последнего символа), чанк в тексте, нормализованный чанк, морфологические признаки чанка, родитель чанка, тип зависимости}
    res = [] # [(Концепция1, глагол, Концепция2)]
    subs_and_preds = {} # {сказуемое: подлежащее}
    for ent in doc.ents: # добавляем именованные сущности
        chars = (ent.start_char, ent.end_char)
        chunks.append((chars, ent, normalize_noun_phrase(doc, ent)) + head_in_named_entity(doc, ent)[1:-1])
    for token in doc: # добавляем существительные
        if token.pos_ == 'NOUN':
            morph = [_.split('=')[1] for _ in str(token.morph).split('|')]
            chars = (token.idx, token.idx + len(token.text))
            chunks.append((chars, token, token.lemma_, morph, token.head, token.dep_))
    chunks.sort(key=lambda x: x[0])
    for token in doc: # решаем анафору
        if token.pos_ == 'PRON':
            morph = [_.split('=')[1] for _ in str(token.morph).split('|')]
            for chunk in chunks:
                if chunk[0][0] < token.idx and chunk[3][2:4] == morph[1:3]:
                    chars = (token.idx, token.idx + len(token.text))
                    pron_chunk = (chars, token, normalize_noun_phrase(doc, chunk[1]), morph, token.head, token.dep_)
            chunks.append(pron_chunk)
    for chunk in chunks:
        if chunk[5] == 'nsubj':
            subs_and_preds[chunk[4]] = chunk[2]
    for chunk in chunks:
        if (type(chunk[1]) == Span or chunk[1].pos_ == 'NOUN' or chunk[1].pos_ == 'PRON') and chunk[4].pos_ == 'VERB' and chunk[5] != 'nsubj':
            res.append((subs_and_preds[chunk[4]], chunk[4].text, chunk[2], chunk[5]))
    return res


if __name__ == "__main__":
    text = '''
        Мама мыла оконную раму. Из неё выпало стекло. Оно разбилось о пол.
        На полу спал наш пёс шарик. Он услышал звук бьющегося стекла. Шарик залаял на маму.
        Мама включила телевизор. На Первом Канале выступал Григорий Лепс.
        Недавно Правительство РФ присвоило ему звание Народного Артиста РФ.
    '''
    doc = nlp(text)
    print(get_syntactic_relations(doc))
