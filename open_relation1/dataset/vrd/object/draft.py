from nltk.corpus import wordnet as wn

synsets = wn.synsets('with')
print(synsets)
i = 0

print(synsets[i].hypernym_paths())
print(synsets[i].definition())