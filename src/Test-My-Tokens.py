#--------------------------------------------------------------------
# Tokenize and tag some text, from https://www.nltk.org/
# This demonstrates from functionality from the Natural Language Toolkit.
# Note that this toolkit contains the Penn Treebank, which is a database
# of sentences taken from the Wall Street Journal. By having a common
# set of sentences for testing purposes, this helps NL researchers
# to compare their results.
#--------------------------------------------------------------------

import nltk

# Put this here in case it has not already been downloaded
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('treebank')

#--------------------------------------------------------------------
# Tokenize and tag some text, from https://www.nltk.org/
#--------------------------------------------------------------------

sentence = """At eight o'clock on Thursday morning ... Arthur didn't feel very good."""
tokens = nltk.word_tokenize(sentence)
print ("\nHere are the tokens in the string")
print(tokens)

print ("\nNow we take the tokens")
tagged = nltk.pos_tag(tokens)
print(tagged)
print ("\nHere are tagged tokens 0 to 6")
print(tagged[0:6])

#--------------------------------------------------------------------
# Use entities, from https://www.nltk.org/
#--------------------------------------------------------------------

entities = nltk.chunk.ne_chunk(tagged)
print ("\nHere are entities made from the tokens above")
print(entities)

#--------------------------------------------------------------------
# Display a parse tree
#--------------------------------------------------------------------

from nltk.corpus import treebank
t = treebank.parsed_sents('wsj_0001.mrg')[0]
# Print the parse tree to the console
print(t)
# Print a graph of the parse tree
print ("\nWATCH OUT!  A new window should pop up!")
t.draw()

#--------------------------------------------------------------------
# Print some sample sentences from the Penn Treebank
#--------------------------------------------------------------------

# Get the list of file IDs in the Treebank corpus
file_ids = treebank.fileids()
# Count the number of sentences
sentence_count = sum(len(treebank.sents(file_id)) for file_id in file_ids)
print(f"The Treebank corpus contains {sentence_count} sentences.")

# Needed by Ken due to some SSL error, otherwise don't use
# import ssl
# # Bypass SSL certificate verification
# try:
#     _create_unverified_https_context = ssl._create_unverified_context
# except AttributeError:
#     pass
# else:
#     ssl._create_default_https_context = _create_unverified_https_context

# Print out some sentences
import random

# Download the Treebank dataset - watch for this on your hardrive!
nltk.download('treebank')

# Get the list of file IDs in the Treebank corpus
file_ids = treebank.fileids()

# Get all sentences in the Treebank corpus
all_sentences = [sentence for file_id in file_ids for sentence in treebank.sents(file_id)]

# Randomly sample 20 sentences
random_sentences = random.sample(all_sentences, 20)

# Print the randomly selected sentences
for i, sentence in enumerate(random_sentences, start=1):
    print(f"\nSentence {i}: \n{' '.join(sentence)}")

