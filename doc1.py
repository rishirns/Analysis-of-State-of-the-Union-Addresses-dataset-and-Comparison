
# coding: utf-8

# In[37]:

import nltk
import string
from urllib import request


file1= open('/Users/rishi/Desktop/NLP_Data/Assignment_1/state_union_part1.txt','r')


raw = file1.read()

#text of the book is separated into tokens with word tokenizer
#and converted all the characters to lowercase
SUAtokens = nltk.word_tokenize(raw)
SUAwords = [w.lower() for w in SUAtokens]

#print first 200 words which is tokenized and are in lowercase
print("Tokenized and lowercase")
print(SUAwords[:200])


import re

def alpha_filter(w):
  # pattern to match word of non-alphabetical characters
  pattern = re.compile('^[^a-z]+$')
  if (pattern.match(w)):
    return True
  else:
    return False

SUAwords= [w for w in SUAwords if not alpha_filter(w)]
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize

#to get the stopwords list
stopwords =nltk.corpus.stopwords.words('english')  + list(string.punctuation)
print("List of stopwords")
print(stopwords) 

morestopword=['upon','would','say','u','shall','\'s','could','must','us','also']
stopwords += morestopword

#to remove the stopwords
stoppedSUAwords = [w for w in SUAwords if not w in stopwords]
filtered_words = []
for w in SUAwords:
    if w not in stopwords:
        filtered_words.append(w)

porter = nltk.PorterStemmer()
lancaster = nltk.LancasterStemmer()
wnl=nltk.WordNetLemmatizer()
from nltk.stem import SnowballStemmer
snowball_stemmer = SnowballStemmer('english')

SUAPstem = [porter.stem(t) for t in stoppedSUAwords]
print('Porter\n', SUAPstem[:200])

SUALstem = [lancaster.stem(t) for t in stoppedSUAwords]
print('Lancaster\n', SUALstem[:200])

SUALemma = [wnl.lemmatize(t) for t in stoppedSUAwords]
print('WordNet Lemmatizer\n', SUALemma[:200])

SUASnstem = [snowball_stemmer.stem(t) for t in stoppedSUAwords]
print('Snowball Stemmer\n', SUASnstem[:200])

#tokenized, lowercase list without stopwords
print("tokenized, lowercase list without stopwords")
print(filtered_words[:200]) 

from nltk import FreqDist
from nltk.collocations import *
#list the top 50 words by frequency (normalized by the length of the document)
SUAdist = FreqDist(filtered_words)
SUAitems = SUAdist.most_common(50)
print("top 50 words by frequency")
for item in SUAitems:
    print(item[0], '\t', item[1]/len(SUAwords))

SUAbigrams = list(nltk.bigrams(SUAwords))
print("Sample Bigrams")
print(SUAbigrams[:50])

bigram_measures = nltk.collocations.BigramAssocMeasures()
finder = BigramCollocationFinder.from_words(SUAwords)
scored = finder.score_ngrams(bigram_measures.raw_freq)
print ("without any filter")
for bscore in scored[:30]:
    print (bscore)
    
finder2 = BigramCollocationFinder.from_words(SUAwords)
finder2.apply_freq_filter(2)
scored = finder2.score_ngrams(bigram_measures.raw_freq)
print ("removed low frequency words")
for bscore in scored[:30]:
    print (bscore)

finder.apply_word_filter(lambda w: w in stopwords)
scored = finder.score_ngrams(bigram_measures.raw_freq)
print ("Bigrams after removing stopwords")
for bscore in scored[:50]:
    print (bscore)

finder3 = BigramCollocationFinder.from_words(SUAwords)
scored = finder3.score_ngrams(bigram_measures.pmi)
print ("pmi on raw")
for bscore in scored[:30]:
    print (bscore)
  
finder3.apply_freq_filter(5)
scored = finder3.score_ngrams(bigram_measures.pmi)
print ("pmi data with minimum frequency")
for bscore in scored[:50]:
    print (bscore)
    


    





# In[ ]:



