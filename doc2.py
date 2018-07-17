
# coding: utf-8

# In[19]:

import nltk
import string
from urllib import request

#text from online gutenberg

file2= open('/Users/rishi/Desktop/NLP_Data/Assignment_1/state_union_part2.txt','r')
# response = request.urlopen(url)
raw = file2.read()
#decode('utf8')

#text of the book is separated into tokens with word tokenizer
#and converted all the characters to lowercase
#ABtokens = nltk.word_tokenize(raw)
SUBtokens = nltk.word_tokenize(raw)
SUBwords = [w.lower() for w in SUBtokens]

#print first 200 words which is tokenized and are in lowercase
print("Tokenized and lowercase")
print(SUBwords[:200])
import re

def alpha_filter(w):
  # pattern to match word of non-alphabetical characters
  pattern = re.compile('^[^a-z]+$')
  if (pattern.match(w)):
    return True
  else:
    return False

SUBwords= [w for w in SUBwords if not alpha_filter(w)]

from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize

#to print the stopwords list
stopwords =nltk.corpus.stopwords.words('english')  + list(string.punctuation)

print("List of stopwords")
print(stopwords) 

morestopword=['upon','would','say','u','shall','\'s','could','also','us','must']
stopwords += morestopword

#to remove the stopwords
stoppedSUBwords = [w for w in SUBwords if not w in stopwords]
filtered_words = []
for w in SUBwords:
    if w not in stopwords:
        filtered_words.append(w)

#tokenized, lowercase list without stopwords
print("tokenized, lowercase list without stopwords")
print(filtered_words[:200]) 

porter = nltk.PorterStemmer()
lancaster = nltk.LancasterStemmer()
wnl=nltk.WordNetLemmatizer()
from nltk.stem import SnowballStemmer
snowball_stemmer = SnowballStemmer('english')

SUBPstem = [porter.stem(t) for t in stoppedSUBwords]
print('Porter\n', SUBPstem[:200])

SUBLstem = [lancaster.stem(t) for t in stoppedSUBwords]
print('Lancaster\n', SUBLstem[:200])

SUBLemma = [wnl.lemmatize(t) for t in stoppedSUBwords]
print('WordNet Lemmatizer\n', SUBLemma[:200])

SUBSnstem = [snowball_stemmer.stem(t) for t in stoppedSUBwords]
print('Snowball Stemmer\n', SUBSnstem[:200])




from nltk import FreqDist
from nltk.collocations import *
#list the top 50 words by frequency (normalized by the length of the document)
SUBdist = FreqDist(filtered_words)
SUBitems = SUBdist.most_common(50)
print("top 50 words by frequency")
for item in SUBitems:
    print(item[0], '\t', item[1]/len(SUBwords))

SUBbigrams = list(nltk.bigrams(filtered_words))
print("Sample Bigrams")
print(SUBbigrams[:50])

bigram_measures = nltk.collocations.BigramAssocMeasures()
finder = BigramCollocationFinder.from_words(SUBwords)
scored = finder.score_ngrams(bigram_measures.raw_freq)
#print(type(scored))
#first = scored[0]
#print(type(first), first)
print ("without any filter")
for bscore in scored[:50]:
    print (bscore)
    
finder2 = BigramCollocationFinder.from_words(SUBwords)
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


finder3 = BigramCollocationFinder.from_words(SUBwords)
scored = finder3.score_ngrams(bigram_measures.pmi)
print ("pmi on raw")
for bscore in scored[:50]:
    print (bscore)

finder3.apply_freq_filter(5)
scored = finder3.score_ngrams(bigram_measures.pmi)
print ("pmi on filtered data")
for bscore in scored[:50]:
    print (bscore)

finderS = BigramCollocationFinder.from_words(SUBPstem)
finderS.apply_word_filter(lambda w: w in stopwords)
scored = finderS.score_ngrams(bigram_measures.raw_freq)
print ("Special Porter frequency words")
for bscore in scored[:50]:
    print (bscore)
    
finderSP = BigramCollocationFinder.from_words(SUBPstem)
finderSP.apply_freq_filter(5)
scored = finderSP.score_ngrams(bigram_measures.pmi)
print ("Special Porter PMI words")
for bscore in scored[:50]:
    print (bscore)


    




# In[ ]:




# In[ ]:



