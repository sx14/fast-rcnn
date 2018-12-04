import h5py
myFile = h5py.File('dataset/word_vec_wn.h5', 'r')
# The '...' means retrieve the whole tensor
data = myFile['word_vec']
print(data[0])