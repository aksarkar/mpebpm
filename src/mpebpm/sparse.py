"""Support for sparse tensors

Implement the minimal functionality needed to use a CSR-format matrix in
torch.utils.data.TensorDataset. In our application, we don't need gradients
with respect to the sparse matrix, only sparse-dense matrix multiplication in
the forward pass.

"""
import torch
import torch.utils.data as td

class CSRTensor:
  """Placeholder implementation of sparse 2-tensor in CSR format

  This implementation only supports extracting rows by a list of indices

  """
  def __init__(self, data, indices, indptr, shape, dtype=torch.float):
    self.data = torch.tensor(data, dtype=dtype)
    self.indices = torch.tensor(indices, dtype=torch.long)
    self.indptr = torch.tensor(indptr, dtype=torch.long)
    self.shape = torch.Size(shape)

  def __getitem__(self, idx):
    """Return tensor containing rows idx

    Construct a sparse tensor using the CSR index, then convert to dense. This
    is not a fully-featured __getitem__ (like numpy arrays or torch tensors),
    and only supports iterable idx

    """
    return torch.sparse.FloatTensor(
      torch.cat([
        torch.stack([torch.full(((self.indptr[i + 1] - self.indptr[i]).item(),), j, dtype=torch.long),
                     self.indices[self.indptr[i]:self.indptr[i + 1]]]) for j, i in enumerate(idx)], dim=1),
      torch.cat([self.data[self.indptr[i]:self.indptr[i + 1]] for i in idx]),
      size=[len(idx), self.shape[1]]).to_dense().squeeze()

  def cuda(self):
    self.data = self.data.cuda()
    self.indices = self.indices.cuda()
    self.indptr = self.indptr.cuda()
    return self

class SparseDataset(td.Dataset):
  """Specialized dataset type for zipping sparse and dense tensors"""
  def __init__(self, *tensors):
    super().__init__()
    self.tensors = tensors
    self.n = min(t.shape[0] for t in self.tensors)

  def __getitem__(self, index):
    """Dummy implementation of __getitem__

    torch.utils.DataLoader.__next__() calls:

    batch = self.collate_fn([self.dataset[i] for i in indices])

    This is too slow, so instead of actually returning the data, like:

    start = self.indptr[index]
    end = self.indptr[index + 1]
    return (
      torch.sparse.FloatTensor(
        # Important: sparse indices are long in Torch
        torch.stack([torch.zeros(end - start, dtype=torch.long, device=self.indices.device), self.indices[start:end]]),
        # Important: this needs to be 1d before collate_fn
        self.data[start:end], size=[1, self.p]).to_dense().squeeze(),
      self.s[index]
    )

    and then concatenating in collate_fn, just return the index.
    """
    return index
    
  def __len__(self):
    return self.n

  def collate_fn(self, indices):
    """Return a minibatch of items

    Construct the entire sparse tensor in one shot, and then convert to
    dense. This is *much* faster than the default:

    torch.stack([data[i] for i in indices])

    """
    return [t[indices] for t in self.tensors]
