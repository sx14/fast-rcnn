require 'nn'
require 'dpnn'

local HypernymImage, parent = torch.class('nn.HypernymImage', 'nn.Sequential')

function HypernymImage:__init(fc7Dimension, wordVecDimension, wordVecWeights, wordSum)
    parent.__init(self)
    -- fc7 -> wordVecSpace
    local fc7Linear = nn.Linear(fc7Dimension, wordVecDimension)
    local fc7Embedding = nn.Sequential():add(fc7Linear)
    -- wordVec
    local lookup = nn.LookupTable(wordSum, wordVecDimension)
    lookup.weight = wordVecWeights:double()
    -- fraze lookup table parameters
    lookup:zeroGradParameters()
    local wordEmbedding = nn.Sequential():add(lookup)
    -- self: takes two input words, outputs a probability that the first is a hypernym of the second
    self:add(nn.ParallelTable():add(fc7Embedding):add(wordEmbedding))
    self:add(nn.CSubTable())
    -- 1,1,1,1
    -- 2,2,2,2
    self:add(nn.ReLU()) -- max(0,(x - y))
    self:add(nn.Power(2)) -- 1,1,1,1
    self:add(nn.Sum(2))   -- 4
    self.visualFeatureModule = fc7Linear
end

