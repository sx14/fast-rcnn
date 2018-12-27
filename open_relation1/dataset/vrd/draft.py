from nltk.corpus import wordnet as wn

synsets = wn.synsets('glasses')
i = 0
print(synsets)
print(synsets[i])
print(synsets[i].definition())