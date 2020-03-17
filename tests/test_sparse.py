import mpebpm.sparse
import numpy as np
import pytest
import torch
import torch.utils.data as td

from fixtures import *

def test_CSRTensor(csr_matrix):
  x = csr_matrix
  xt = mpebpm.sparse.CSRTensor(x.data, x.indices, x.indptr, x.shape)
  assert np.isclose(xt.data.numpy(), x.data).all()
  assert (xt.indices.numpy() == x.indices).all()
  assert (xt.indptr.numpy() == x.indptr).all()
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

@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
def test_CSRTensor_cuda(csr_matrix):
  x = csr_matrix
  xt = mpebpm.sparse.CSRTensor(x.data, x.indices, x.indptr, x.shape).cuda()
  assert np.isclose(xt.data.cpu().numpy(), x.data).all()
  assert (xt.indices.cpu().numpy() == x.indices).all()
  assert (xt.indptr.cpu().numpy() == x.indptr).all()

def test_SparseDataset__get_item__(simulate_point_gamma):
  x, s, log_mu, log_phi, logodds, l0 = simulate_point_gamma
  x = torch.tensor(x)
  s = torch.tensor(s)
  data = mpebpm.sparse.SparseDataset(x, s)
  y = data[0]
  assert y == 0

def test_SparseDataset_collate_fn(simulate_point_gamma):
  x, s, log_mu, log_phi, logodds, l0 = simulate_point_gamma
  x = torch.tensor(x)
  s = torch.tensor(s)
  data = mpebpm.sparse.SparseDataset(x, s)
  batch_size = 10
  y, t = data.collate_fn(range(batch_size))
  if torch.cuda.is_available():
    assert (y.cpu() == x[:batch_size]).all()
    assert (t.cpu() == s[:batch_size]).all()
  else:
    assert (y == x[:batch_size]).all()
    assert (t == s[:batch_size]).all()

def test_SparseDataset_collate_fn_CSRTensor(simulate_point_gamma):
  x, s, log_mu, log_phi, logodds, l0 = simulate_point_gamma
  xs = ss.csr_matrix(x)
  xt = mpebpm.sparse.CSRTensor(xs.data, xs.indices, xs.indptr, xs.shape)
  s = torch.tensor(s)
  data = mpebpm.sparse.SparseDataset(xt, s)
  batch_size = 10
  y, t = data.collate_fn(range(batch_size))
  if torch.cuda.is_available():
    assert (y.cpu().to_dense().numpy() == x[:batch_size]).all()
  else:
    assert (y.to_dense().numpy() == x[:batch_size]).all()

def test_SparseDataset_collate_fn_shuffle(simulate_point_gamma):
  x, s, log_mu, log_phi, logodds, l0 = simulate_point_gamma
  x = torch.tensor(x)
  s = torch.tensor(s)
  data = mpebpm.sparse.SparseDataset(x, s)
  idx = [10, 20, 30, 40, 50]
  y, t = data.collate_fn(idx)
  if torch.cuda.is_available():
    assert (y.cpu() == x[idx]).all()
    assert (t.cpu() == s[idx]).all()
  else:
    assert (y == x[idx]).all()
    assert (t == s[idx]).all()

def test_SparseDataset_DataLoader(simulate_point_gamma):
  x, s, log_mu, log_phi, logodds, l0 = simulate_point_gamma
  x = torch.tensor(x)
  s = torch.tensor(s)
  batch_size = 10
  sparse_data = mpebpm.sparse.SparseDataset(x, s)
  data = td.DataLoader(sparse_data, batch_size=batch_size, shuffle=False, collate_fn=sparse_data.collate_fn)
  y, t = next(iter(data))
  if torch.cuda.is_available():
    assert (y.cpu() == x[:batch_size]).all()
    assert (t.cpu() == s[:batch_size]).all()
  else:
    assert (y == x[:batch_size]).all()
    assert (t == s[:batch_size]).all()
