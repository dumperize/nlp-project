from itertools import combinations
import networkx as nx
import nltk


PART = 0.1

def unique_words_similarity(words_1, words_2):
    words_1 = set(words_1)
    words_2 = set(words_2)

    if not len(words_1) or not len(words_2):
        return 0.0
    return len(words_1.intersection(words_2)) / (len(words_1) + len(words_2))

def get_text_rank(text):
    sentences = nltk.sent_tokenize(text)
    n_sentences = len(sentences)

    if n_sentences == 1:
        return text

    words = [[token for token in nltk.word_tokenize(sentence)] for sentence in sentences]
    # lemmatization ?

    pairs = combinations(range(n_sentences), 2)
    scores = [(i, j, unique_words_similarity(words[i], words[j])) for i,j in pairs]

    g = nx.Graph()
    g.add_weighted_edges_from(scores)

    page_rank = nx.pagerank(g)
    result = [(i, page_rank[i], sentence) for i, sentence in enumerate(sentences) if i in page_rank]
    
    result.sort(key = lambda x: x[1], reverse=True)

    n_summary_sentences = max(int(n_sentences * PART), 1)
    result = result[:n_summary_sentences]
    
    result.sort(key = lambda x: x[0]) # original sort

    return " ".join([sentence for i, pr, sentence in result])
