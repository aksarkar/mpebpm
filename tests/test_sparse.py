import mpebpm.sparse
import numpy as np
import pytest
import scipy.sparse as ss
import torch

@pytest.fixture
def csr_matrix():
  np.random.seed(0)
  x = ss.random(m=10, n=5, density=0.1).tocsr()
  return x

def test_CSRTensor(csr_matrix):
  x = csr_matrix
  xt = mpebpm.sparse.CSRTensor(x.data, x.indices, x.indptr, x.shape)
  assert np.isclose(xt.data.numpy(), x.data).all()
  assert (xt.indices.numpy() == x.indices).all()
  assert tuple(xt.shape) == x.shape

def test_CSRTensor___getitem__(csr_matrix):
  x = csr_matrix
  x0 = x[[0]].tocoo()
  xt = mpebpm.sparse.CSRTensor(x.data, x.indices, x.indptr, x.shape)
  # Important: coalesce() is required by torch.sparse.FloatTensor
  xt0 = xt[[0]].coalesce()
  assert x0.shape == tuple(xt0.shape)
  assert (x0.row == xt0.indices()[0].numpy()).all()
  assert (x0.col == xt0.indices()[1].numpy()).all()
  assert np.isclose(x0.data, xt0.values().numpy()).all()

