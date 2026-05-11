import random
import nltk
nltk.download('wordnet', quiet=True)
from nltk.corpus import wordnet as wn

def build_definition_cache():
    cache = {}
    for synset in wn.all_synsets('n'):
        word = synset.lemmas()[0].name().replace('_', ' ').lower()
        cache[word] = synset.definition()
    return cache

def get_wordnet_docs(limit=None):
    docs = []
    for synset in wn.all_synsets('n'):
        word = synset.lemmas()[0].name().replace('_', ' ')
        definition = synset.definition()
        docs.append(f"Q: what does {word} mean? A: {definition}")
        docs.append(f"Q: define {word} A: {definition}")
        docs.append(f"Q: definition of {word} A: {definition}")
        docs.append(f"Q: what is the meaning of {word}? A: {definition}")
    random.shuffle(docs)
    if limit is not None:
        docs = docs[:limit]
    return docs

def get_docs_from_cache(cache):
    docs = []
    for word, definition in cache.items():
        docs.append(f"Q: what does {word} mean? A: {definition}")
        docs.append(f"Q: define {word} A: {definition}")
        docs.append(f"Q: definition of {word} A: {definition}")
        docs.append(f"Q: what is the meaning of {word}? A: {definition}")
    random.shuffle(docs)
    return docs