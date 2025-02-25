import spacy
import spacy.cli
from spacy.tokens import Span

from pymorphy3 import MorphAnalyzer

import types
from types import NoneType


spacy.cli.download("ru_core_news_lg")
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
            if (pos == 'ADJF' or 'Surn' in str(np_part.tag)) and np_part_head == head:
                gender, number = str(ana.normalized.tag).split(',')[2].split()[-1], str(ana.normalized.tag).split(',')[3].split()[-1]
                np_part = np_part.inflect({gender, 'nomn'})[0]
            else:
                np_part = np_part.word
        res += np_part + ' '
    return res.strip()


def get_syntactic_relations(doc):
    chunks = [] # [(токен, (индекс первого символа, индекс последнего символа), чанк в тексте, нормализованный чанк, морфологические признаки чанка, родитель чанка, тип зависимости)]
    d_chunks = {}
    res = [] # [(Концепция1, глагол, Концепция2)]
    subs_and_preds = {} # {сказуемое: подлежащее}
    for ent in doc.ents: # добавляем именованные сущности
        head, morph, parent, dep = head_in_named_entity(doc, ent)[:-1]
        chars = (ent.start_char, ent.end_char)
        chunks.append((head, chars, ent, normalize_noun_phrase(doc, ent), morph, parent, dep))
    for token in doc: # добавляем существительные
        if token.pos_ in ['NOUN', 'PROPN']:
            morph = [_.split('=')[1] for _ in str(token.morph).split('|')]
            chars = (token.idx, token.idx + len(token.text))
            chunks.append((token, chars, token, token.lemma_, morph, token.head, token.dep_))
    chunks.sort(key=lambda x: x[1])
    for token in doc: # решаем анафору
        if token.pos_ == 'PRON':
            morph = [_.split('=')[1] for _ in str(token.morph).split('|')]
            pron_chunk = None
            for chunk in chunks:
                if chunk[1][0] < token.idx and chunk[4][2:4] == morph[1:3]:
                    chars = (token.idx, token.idx + len(token.text))
                    pron_chunk = (token, chars, token, normalize_noun_phrase(doc, chunk[2]), morph, token.head, token.dep_)
            if pron_chunk is not None:
                chunks.append(pron_chunk)
    chunks.sort(key=lambda x: x[1])
    for chunk in chunks:
        d_chunks[chunk[0]] = chunk[1:]
    for chunk in d_chunks.values(): # Концепции1
        if chunk[5] == 'nsubj':
            if any(child.dep_ == 'conj' for child in chunk[1].children):
                for child in chunk[1].children:
                    if child.dep_ == 'conj':
                        for grandchild in child.children:
                            if grandchild.dep_ == 'cc':
                                conj = grandchild
                                break
                res.append((chunk[2], conj.text, d_chunks[child][2], 'conj'))
                subs_and_preds[chunk[4]] = (chunk[2], conj.text, d_chunks[child][2])
            else:
                subs_and_preds[chunk[4]] = chunk[2]
        elif chunk[4].dep_ in ['acl', 'acl:relcl'] and chunk[4] not in subs_and_preds and chunk[4].head.pos_ != 'PRON':
            subs_and_preds[chunk[4]] = d_chunks[chunk[4].head][2]
        elif chunk[4].dep_ == 'conj' or chunk[4].dep_ == 'advcl' and chunk[4] not in subs_and_preds:
            if not any(child.dep_ == 'nsubj' and child.pos == 'NOUN' for child in chunk[4].children):
                morph1, morph2 = {}, {}
                for (verb, morph) in [(chunk[4], morph1), (chunk[4].head, morph2)]:
                    for _ in str(verb.morph).split('|'):
                        key, value = _.split('=')
                        morph[key] = value
                if morph1['Number'] == morph2['Number']:
                    try:
                        if morph1['Gender'] == morph2['Gender']:
                            if chunk[4].head in subs_and_preds:
                                subs_and_preds[chunk[4]] = subs_and_preds[chunk[4].head]
                    except Exception:
                        if chunk[4].head in subs_and_preds:
                            subs_and_preds[chunk[4]] = subs_and_preds[chunk[4].head]
    for chunk in chunks:
        if (type(chunk[2]) == Span or chunk[2].pos_ in ['NOUN', 'PROPN', 'PRON']) and (chunk[5].pos_ == 'VERB' and 'nsubj' not in chunk[6]):
            if any(child.dep_ == 'conj' for child in chunk[2].children):
                for child in chunk[2].children:
                    if child.dep_ == 'conj':
                        for grandchild in child.children:
                            if grandchild.dep_ == 'cc':
                                conj = grandchild
                                break
                res.append((subs_and_preds[chunk[5]], chunk[5].text, (chunk[3], conj.text, d_chunks[child][2])))
            else:
                res.append((subs_and_preds[chunk[5]], chunk[5].text, chunk[3], chunk[6]))
    return res


if __name__ == "__main__":
    text = '''Конечно, дарвинизм в его изначальном виде столкнулся с более значительными и
непосредственными проблемами, чем вопрос о достаточности естественного отбора: Дарвин и
его ранние последователи не имели представления о механизмах наследования и о том, будут
ли когда-либо открыты механизмы, согласующиеся со сценарием Дарвина. В этом смысле здание теории Дарвина висело в воздухе. Повторное открытие законов генетики в начале
XX века и развитие теоретической и экспериментальной популяционной генетики обеспечило твердое основание для дарвиновской теории эволюции. Было показано, что, без
сомнения, популяции эволюционируют посредством процесса, в котором дарвиновский
естественный отбор играет важнейшую роль.
'''
    doc = nlp(text)
    print(get_syntactic_relations(doc))