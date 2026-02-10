# -*- coding: utf-8 -*-
"""
----------------------------------------------------------------------------------------------------------------------
Program to train the Word2vec model
----------------------------------------------------------------------------------------------------------------------
Created on: 22 June 2020
---------------------------------------------------------------------------------------------------------------------
"""
import constants
import re
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.util import bigrams, trigrams, ngrams
from gensim.models import word2vec, KeyedVectors

import nltk
nltk.download('punkt')


def clean_sentence(sentence):
    """pre-process the sentence
    :param sentence: string
    :return: preprocessed string
    """
    sentence = sentence.lower().strip()
    sentence = re.sub('[^a-z_#+.]+', ' ',sentence)
    sentence = sentence.replace('u+','')
    return re.sub(r'\s{2,}', ' ', sentence)


def tokenize(doc, stop):
    """Tokenization of document text

    :param doc: document text in string format
    :param stop: list of stop words
    :return sentences: list of tokens after preprocessing
    """
    sentences = []
    doc_sent = sent_tokenize(doc)
    for i in doc_sent:
        tokens = [t for t in word_tokenize(clean_sentence(i)) if t not in stop and t != '.']
        bigrm = [' '.join(g) for g in list(bigrams(tokens))]
        trigrm = [' '.join(g) for g in list(trigrams(tokens))]
        ngrm = [' '.join(g) for g in list(ngrams(tokens, 4))]
        tokens = tokens + bigrm + trigrm + ngrm
        sentences.append(tokens)
    return sentences


def train_model(sentences):
    """Training the Word2vec model

    :param sentences: list of list of tokens
    :return model: trained Word2vec model
    """
    model = word2vec.Word2Vec(sentences, vector_size=200, min_count=2, sg=1, window=8)
    try:
        model.save(constants.MODEL_PATH)
    except Exception as e:
        print(f"Error saving model: {e}")
    return model


def get_most_sim(word, top, model):
    """ to find the top most similar keywords of the word text

    :param word: string
    :param top: integer
    :param model: object of a word2vec model
    """
    sim = model.wv.most_similar(word, topn=top)
    print('The top {} most similar keywords of the word {} are-'.format(top, word))
    for i in sim:
        print('{} with score {}'.format(i[0], i[1]))

def main():
    with open(constants.CORPUS_PATH, 'r', encoding='utf-8') as file:
        data = file.read()

    with open(constants.STOP_WORD_PATH, 'rb') as file:
        stop = str(file.read())
    
    stop = stop.split()
    sent_tokens = tokenize(data, stop)
    model = train_model(sent_tokens)
    # get_most_sim('machine learning', 10, model)


if __name__ == "__main__":
    main()