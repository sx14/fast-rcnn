require 'nngraph'
require 'nn'
require 'dpnn'
require 'Dataset'
require 'hdf5'

FOR_VS = true
featureDimension = 300

if FOR_VS then
  datasetPath = 'exp_dataset/contrastive_trans.t7'
  weights = torch.load('word_embedding_weights_vs.t7')
else
  datasetPath = 'dataset/contrastive_trans.t7'
  weights = torch.load('word_embedding_weights_wn.t7')
end

dataset = torch.load(datasetPath)
lookup = nn.LookupTable(dataset.numEntities, featureDimension)
lookup.weight = weights:double()
fs = torch.Tensor(dataset.numEntities, featureDimension)
embedding = nn.Sequential():add(lookup)
for i=1, dataset.numEntities do
  input = torch.Tensor({i})
  f = embedding:forward(input):clone()
  fs[i] = f
end
if FOR_VS then
  myFile = hdf5.open('word_vec_vs.h5', 'w')
else
  myFile = hdf5.open('word_vec_wn.h5', 'w')
end
myFile:write('word_vec', torch.Tensor(fs))
myFile:close()
