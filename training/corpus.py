import nltk.corpus
import random
import re

re_word = re.compile(r'^[a-zA-Z0-9].*$')

def word_join(words: list[str]) -> str:
    word_spaces = []
    for w in words:
        if len(word_spaces) > 0 and re_word.match(w):
            word_spaces.append(' ')
        word_spaces.append(w)
    return ''.join(word_spaces)

def get_corpus(seed:int, n:int):
    c = nltk.corpus.brown
    s = c.sents()
    random.seed(seed)
    randoms = random.sample(range(len(s)), k=n)
    return [word_join(s[r]) for r in randoms]

