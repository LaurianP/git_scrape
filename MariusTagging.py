# -*- coding: utf-8 -*-
"""
Created on Mon May 28 23:05:23 2018

@author: popaz
"""
import nltk
from nltk.tokenize import WhitespaceTokenizer
from nltk.corpus import brown

class MariusTagger: 
    def __init__(self):
        self.train_tagged_sents = brown.tagged_sents()
        self.default_tagger  = nltk.DefaultTagger('NN')
        self.unigram_tagger  = nltk.UnigramTagger(self.train_tagged_sents, backoff = self.default_tagger )
        self.bigram_tagger   = nltk.BigramTagger(self.train_tagged_sents, backoff = self.unigram_tagger )
        self.trigram_tagger  = nltk.TrigramTagger(self.train_tagged_sents, backoff = self.bigram_tagger) 

    def tagText(self, text):
        myTokens = WhitespaceTokenizer().tokenize(text)
        result = self.trigram_tagger.tag(myTokens)
        return(result)
