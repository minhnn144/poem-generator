import numpy as np
from numpy.linalg import norm
import pickle
import underthesea

import spacy
from spacy.language import Language

import os
import re

root = os.getenv("PYTHONPATH")


class VNlp:
    def __init__(self, lang="vi", tokenize=0):
        """Initial a vnlp processor

        Keyword Arguments:
            lang {str}      -- language label (default: {"vi"})
            tokenize {int}  -- select default and custom tokenizer (default: {"0"})

        """
        self.nlp = spacy.blank(lang)
        if tokenize == 0:
            pass
        elif tokenize == 1:
            # self.tokenizer = underthesea.word_tokenize()
            self.tokenizer = self.nlp.Defaults.create_tokenizer(nlp=self.nlp)
            pass

    def load_model(self, model):
        """Load model from model string

        Arguments:
            model {str} -- model
        """
        self.nlp = spacy.load(model)

    def add_vector(self, word, vector):
        """Add more word with vector to vocabulary
        
        Arguments:
            word {str} -- word text
            vector {list[num]} -- vector of word
        """
        vector = np.asarray(vector, dtype='f')
        if vector.size != len(self.to_vector("hi")):
            print('vector size not fit!')
            return
        self.nlp.vocab.set_vector(word, vector)

    def load_copus(self, vectors_corpus):
        """Load data from corpus file

        Arguments:
            vectors_corpus {str} -- corpus file location
        """
        with open(root + vectors_corpus, 'rb') as file_:
            header = file_.readline()
            nr_row, nr_dim = header.split()
            self.nlp.vocab.reset_vectors(width=int(nr_dim))
            for line in file_:
                line = line.rstrip().decode('utf8')
                pieces = line.rsplit(' ', int(nr_dim))
                word = pieces[0]
                vector = np.asarray([float(v)
                                     for v in pieces[1:]], dtype='f')
                # add the vectors to the vocab
                self.nlp.vocab.set_vector(word, vector)

    @staticmethod
    def cosine_distance(vector1, vector2):
        """Caculate cosine distance of 2 vectors

        Arguments:
            vector1 {list(num)} -- first vector
            vector2 {list(num)} -- second vector

        Returns:
            num -- cosine distance
        """
        return np.inner(vector1, vector2) / (norm(vector1) * norm(vector2))

    def to_vector(self, sent):
        """get vector from sentence

        Arguments:
            sent {str} -- input sentence

        Returns:
            vector -- vector
        """
        # print(sent)
        return list(self.nlp(sent).vector)

    @staticmethod
    def normalization(document: str, black_list: str = "[\W_]+") -> str:
        """normalize a document with remove black list character and multiple space

        Arguments:
            document {str} -- input document

        Keyword Arguments:
            black_list {str} -- black list character regex pattern (default: {"[\W_]+"})

        Returns:
            str -- document string after normalize
        """
        document = re.sub(black_list, " ", document.lower())
        return re.sub("\s+", " ", document)

    def get_closest(self, vector):
        """Get closest word from input vector with cosine distance
        
        Arguments:
            vector {list(num)} -- input vector
        
        Returns:
            str -- output word
        """
        if len(vector) != len(self.to_vector("hi")):
            print("size of input vector not fit!")
            return
        vector = np.asarray(vector, dtype='f')
        key = self.nlp.vocab.vectors.most_similar(np.asarray([vector]), n=1)
        text = self.nlp.vocab.strings[key[0][0][0]]
        return text

    @staticmethod
    def tokenize(sentence, rm_stopwords=True):
        """split a input string to token
        
        Arguments:
            sentence {str} -- input string
        
        Keyword Arguments:
            rm_stopwords {bool} -- remove stopwords or not (default: {True})
        
        Returns:
            list(str) -- output list of tokens
        """

        if rm_stopwords:
            pass
        else:
            pass
        return list(underthesea.word_tokenize(sentence))
        # return list(self.nlp.tokenizer(sentence))

    def to_disk(self, path):
        """save model to disk

        Arguments:
            path {str} -- save location
        """
        self.nlp.to_disk(root + path)

    def to_bin(self, path):
        """save model to bin file

        Arguments:
            path {str} -- save file location
        """
        with open(root + path, 'wb') as f:
            pickle.dump(self.nlp.to_bytes(), f)

    def from_disk(self, path):
        """load model from file

        Arguments:
            path {str} -- save file location
        """
        self.nlp.from_disk(root + path)

    def from_bin(self, path):
        """load model from bin file

        Arguments:
            path {str} -- save file locaiton
        """
        with open(root + path, 'rb') as f:
            self.nlp.from_bytes(pickle.load(f))
