# Text-summarization
Text-summarization (extractive) using the TextRank Algorithm

import nltk

import numpy as np

import pandas as pd

import re

nltk.download('punkt')

from nltk.tokenize import sent_tokenize

from nltk.corpus import stopwords

from gensim.models import Word2Vec

from scipy import spatial

import networkx as nx


text= input("Enter text")


sentences=sent_tokenize(text)

nltk.download('stopwords')

!pip install textstat

import textstat

print("Num sentences:", textstat.sentence_count(text))

print("Num words:", len(text.split()))

sentences_clean=[re.sub(r'[^\w\s]','',sentence.lower()) for sentence in sentences]

stop_words = stopwords.words('english')

sentence_tokens=[[words for words in sentence.split(' ') if words not in stop_words] for sentence in sentences_clean]

w2v=Word2Vec(sentence_tokens,vector_size=1,min_count=1,epochs=1000)

sentence_embeddings=[[w2v.wv[word][0] for word in words] for words in sentence_tokens]

max_len=max([len(tokens) for tokens in sentence_tokens])

sentence_embeddings=[np.pad(embedding,(0,max_len-len(embedding)),'constant') for embedding in sentence_embeddings]

similarity_matrix = np.zeros([len(sentence_tokens), len(sentence_tokens)])

for i,row_embedding in enumerate(sentence_embeddings):
    
    for j,column_embedding in enumerate(sentence_embeddings):
        
        similarity_matrix[i][j]=1-spatial.distance.cosine(row_embedding,column_embedding)

nx_graph = nx.from_numpy_array(similarity_matrix)

scores = nx.pagerank(nx_graph)


summary=int(input("enter the number of sentences you want to display:"))

top_sentence={sentence:scores[index] for index,sentence in enumerate(sentences)}

top=dict(sorted(top_sentence.items(), key=lambda x: x[1], reverse=True)[:summary])

for sent in sentences:
    
    if sent in top.keys():
        
        print(sent)
