import mpebpm
import numpy as np
import pytest
import torch

def test_NoisyTopicModel():
  p = 100
  k = 10
  m = mpebpm.topics.NoisyTopicModel(p, k)
  assert m.F.shape == torch.Size([k, p])
  assert m.H.shape == torch.Size([k, p])
  assert m.encoder.forward(torch.zeros([1, p])).shape == torch.Size([1, k])

def test_NoisyTopicModel_fit():
  n = 50
  p = 100
  k = 10
  rng = np.random.default_rng(1)
  x = rng.poisson(lam=10, size=(n, p))
  m = mpebpm.topics.NoisyTopicModel(p, k)
  m.fit(x, batch_size=n, num_epochs=1)
  assert m.F.shape == torch.Size([k, p])
  assert m.H.shape == torch.Size([k, p])
  assert m.encoder.forward(torch.zeros([1, p])).shape == torch.Size([1, k])
