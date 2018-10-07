from nltk.corpus import wordnet as wn

s = wn.synset('compete.v.01')
print(s.hypernym_paths())