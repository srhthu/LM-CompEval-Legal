"""Parse open generation text to pre-defined labels"""
import numpy as np
from rank_bm25 import BM25Okapi
import jieba
from typing import List

class BM25_Parser:
    def __init__(self, label2id):
        self.all_labels = list(label2id.keys())
        corpus = [list(jieba.cut(k, cut_all = True)) for k in self.all_labels]
        self.bm25 = BM25Okapi(corpus)
    
    def __call__(self, choices: List[str]):
        """
        Given several sampled outputs, map them to a label.
        """
        tokenized_choices = [list(jieba.cut(c, cut_all = True)) for c in choices]
        cho_score = [self.bm25.get_scores(c) for c in tokenized_choices]
        # average the similarity score across all outputs
        cho_s = np.mean(cho_score, axis = 0)
        # select the 
        top_idx = np.argsort(cho_s)[-1]
        return self.all_labels[top_idx]