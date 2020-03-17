import torch

class CSRTensor:
  """Placeholder implementation of sparse 2-tensor in CSR format

  Implement the minimal functionality needed to use a CSR-format matrix in
  torch.utils.data.TensorDataset. In our application, we don't need gradients
  with respect to the sparse matrix, only sparse-dense matrix multiplication in
  the forward pass.

  """
  def __init__(self, data, indices, indptr, shape, dtype=torch.float):
    self.data = torch.tensor(data, dtype=dtype)
    self.indices = torch.tensor(indices, dtype=torch.long)
    self.indptr = torch.tensor(indptr, dtype=torch.long)
    self.shape = torch.Size(shape)

  def __getitem__(self, idx):
    """Return torch.sparse.FloatTensor containing rows idx

    This is not a fully-featured __getitem__ (like numpy arrays or torch
    tensors), and only supports iterable idx

    """
    return torch.sparse.FloatTensor(
      torch.cat([
        torch.stack([torch.full(((self.indptr[i + 1] - self.indptr[i]).item(),), j, dtype=torch.long),
                     self.indices[self.indptr[i]:self.indptr[i + 1]]]) for j, i in enumerate(idx)], dim=1),
      torch.cat([self.data[self.indptr[i]:self.indptr[i + 1]] for i in idx]),
      size=[len(idx), self.shape[1]])
