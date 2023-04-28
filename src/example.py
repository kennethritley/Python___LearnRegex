# Example partially based on: https://kavita-ganesan.com/tfidftransformer-tfidfvectorizer-usage-differences/#Tfidfvectorizer-Usage
'''
This is just a little script that tests whether all the bits-and-pieces
have been installed for the Chat Bot exercise

Author: KEN
Date:   2022.04.04
'''

import nltk
import numpy
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

document1 = "This is a document talking about dogs. Dogs are animals."
sent_tokens = nltk.sent_tokenize(document1)  # converts to list of sentences
sentence = "Zombies are also animals."
sent_tokens.append(sentence)  # add sentence
sent_tokens.append("What are zombies?")  # add another sentence

## TF

cv=CountVectorizer()
word_count_vector=cv.fit_transform(sent_tokens)
print(word_count_vector.shape)

## IDF

tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
tfidf_transformer.fit(word_count_vector)

# print idf values
df_idf = pd.DataFrame(tfidf_transformer.idf_, index=cv.get_feature_names(), columns=["idf_weights"])

print(df_idf)

## TF-IDF

tf_idf_vector=tfidf_transformer.transform(word_count_vector)  # computes tfidf as tf*idf

## Printing the results for the first document
feature_names = cv.get_feature_names()

# get tfidf vector for first document
first_document_vector = tf_idf_vector[0]

#print the scores
df = pd.DataFrame(first_document_vector.T.todense(), index=feature_names, columns=["tfidf"])
df.sort_values(by=["tfidf"],ascending=False)

print(df)

## Comparing the newly added sentence to the existing sentences

# compare the first element from the right to the rest of the documents
vals = cosine_similarity(tf_idf_vector[-1], tf_idf_vector)
vals = vals.flatten()  # returns a copy of the array collapsed into one dimension
closest = numpy.amax(vals[:-1])  # skip last one, since it is itself (similarity = 1)
closestIndex = int(numpy.where(vals == numpy.amax(vals[:-1]))[0])  # index of the max element
print("Newly added sentence: ", sent_tokens[-1])
print("The closest sentence is: ", sent_tokens[closestIndex])











