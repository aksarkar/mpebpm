"""Empirical Bayes Poisson Means via SGD

These implementations are specialized for two scenarios:

1. Fitting p EBPM problems on n samples in parallel, where n, p may be large

2. Fitting p * k EBPM problems in parallel, where the n samples are assumed to
   be drawn from a discrete (known) choice of k different priors (for each
   gene)

"""
import mpebpm.sparse
import numpy as np
import os.path
import scipy.sparse as ss
import torch
import torch.utils.data as td
import torch.utils.tensorboard as tb

def _pois_llik(x, s, log_mean, beta, design, onehot=None):
  """Return ln p(x_i | s_i, g)

  Assume a design matrix of covariates with effects beta to be
  estimated. Without the design matrix, the MLE is analytic and we do not need
  to use this as a subroutine.

  x_i ~ Poisson(s_i lambda_i)
  lambda_i ~ g = delta_0(lambda_i - mu)

  x - [n, p] tensor
  s - [n, 1] tensor
  log_mean - [k, p] tensor (default: k = 1)
  log_inv_disp - [k, p] tensor (default: k = 1)
  onehot - [n, k] tensor

  """
  if onehot is None:
    mean = torch.matmul(s, torch.exp(log_mean))
  else:
    # This is OK for minibatches, but not for batch GD
    mean = s * torch.matmul(onehot, torch.exp(log_mean))
  mean *= torch.exp(torch.matmul(design, beta))
  return x * torch.log(mean) - mean - torch.lgamma(x + 1)

def _nb_llik(x, s, log_mean, log_inv_disp, beta=None, onehot=None, design=None):
  """Return ln p(x_i | s_i, g)

  x_i ~ Poisson(s_i lambda_i)
  lambda_i ~ g = Gamma(exp(log_inv_disp), exp(log_mean - log_inv_disp))

  x - [n, p] tensor
  s - [n, 1] tensor
  log_mean - [k, p] tensor (default: k = 1)
  log_inv_disp - [k, p] tensor (default: k = 1)
  onehot - [n, k] tensor

  """
  if onehot is None:
    mean = torch.matmul(s, torch.exp(log_mean))
    inv_disp = torch.exp(log_inv_disp)
  else:
    # This is OK for minibatches, but not for batch GD
    mean = s * torch.matmul(onehot, torch.exp(log_mean))
    inv_disp = torch.matmul(onehot, torch.exp(log_inv_disp))
  if beta is not None:
    assert design is not None
    mean *= torch.exp(torch.matmul(design, beta))
  return (x * torch.log(mean / inv_disp)
          - x * torch.log(1 + mean / inv_disp)
          - inv_disp * torch.log(1 + mean / inv_disp)
          # Important: these terms are why we use inverse dispersion
          + torch.lgamma(x + inv_disp)
          - torch.lgamma(inv_disp)
          - torch.lgamma(x + 1))

def _zinb_llik(x, s, log_mean, log_inv_disp, logodds, beta=None, onehot=None, design=None):
  """Return ln p(x_i | s_i, g)

  x_i ~ Poisson(s_i lambda_i)
  lambda_i ~ g = sigmoid(logodds) \\delta_0(.) + sigmoid(-logodds) Gamma(exp(log_inv_disp), exp(log_mean - log_inv_disp))

  x - [n, p] tensor
  s - [n, 1] tensor
  log_mean - [k, p] tensor (default: k = 1)
  log_inv_disp - [k, p] tensor (default: k = 1)
  logodds - [k, p] tensor (default: k = 1)
  onehot - [n, k] tensor

  """
  if onehot is not None:
    logodds = torch.matmul(onehot, logodds)
  nb_llik = _nb_llik(x, s, log_mean, log_inv_disp, beta=beta, onehot=onehot, design=design)
  softplus = torch.nn.functional.softplus
  case_zero = -softplus(-logodds) + softplus(nb_llik - logodds)
  case_non_zero = -softplus(logodds) + nb_llik
  return torch.where(torch.lt(x, 1), case_zero, case_non_zero)

def _check_args(x, s, onehot, design, init, lr, batch_size, num_epochs, log_dir):
  """Check input dimensions and learning parameters, and return a DataLoader"""
  n, p = x.shape
  if s is None:
    s = torch.tensor(x.sum(axis=1).reshape(-1, 1), dtype=torch.float)
  elif s.shape != (n, 1):
    raise ValueError(f'shape mismatch (s): expected {(n, 1)}, got {s.shape}')
  elif not isinstance(s, torch.FloatTensor):
    s = torch.tensor(s, dtype=torch.float)
  else:
    s = s
  if any(s == 0):
    raise ValueError(f'all size factors must be > 0')
  if ss.issparse(x):
    if not ss.isspmatrix_csr(x):
      x = x.tocsr()
    x = mpebpm.sparse.CSRTensor(x.data, x.indices, x.indptr, x.shape, dtype=torch.float)
  elif not isinstance(x, torch.Tensor):
    x = torch.tensor(x, dtype=torch.float)
  if onehot is not None:
    if onehot.shape[0] != n:
      raise ValueError(f'shape mismatch (onehot): expected ({n}, k), got {onehot.shape}')
    k = onehot.shape[1]
    if not ss.issparse(onehot):
      onehot = ss.csr_matrix(onehot)
    onehot = mpebpm.sparse.CSRTensor(onehot.data, onehot.indices, onehot.indptr, onehot.shape, dtype=torch.float)
  else:
    # Important: don't use onehot = torch.ones(n) because this might be big
    k = 1
  if design is not None:
    if design.shape[0] != n:
      raise ValueError(f'shape mismatch (onehot): expected ({n}, k), got {onehot.shape}')
    m = design.shape[1]
    if ss.issparse(design):
      design = ss.csr_matrix(design)
      design = mpebpm.sparse.CSRTensor(design.data, design.indices, design.indptr, design.shape, dtype=torch.float)
    else:
      design = torch.tensor(design, dtype=torch.float)
  else:
    # Important: don't use design = torch.zeros(n)
    m = 0
  if torch.cuda.is_available():
    x = x.cuda()
    s = s.cuda()
    if onehot is not None:
      onehot = onehot.cuda()
    if design is not None:
      design = design.cuda()
  tensors = [x, s]
  if onehot is not None:
    tensors.append(onehot)
  if design is not None:
    tensors.append(design)
  data = mpebpm.sparse.SparseDataset(*tensors)
  if init is None:
    pass
  elif init[0].shape != (k, p):
    raise ValueError(f'shape mismatch (log_mu): expected {(k, p)}, got {init[0].shape}')
  elif init[1].shape != (k, p):
    raise ValueError(f'shape mismatch (log_phi): expected {(k, p)}, got {init[1].shape}')
  elif len(init) > 2:
    raise ValueError(f'expected two values in init, got {len(init)}')
  if lr <= 0:
    raise ValueError('lr must be >= 0')
  if batch_size < 1:
    raise ValueError('batch_size must be >= 1')
  if num_epochs < 1:
    raise ValueError('num_epochs must be >= 1')
  if log_dir is not None and not os.path.exists(log_dir):
    os.makedirs(log_dir)
  return data, n, p, k, m, log_dir

def _sgd(data, onehot, design, llik, params, lr=1e-2, batch_size=100, num_epochs=100, shuffle=False, num_workers=0, log_dir=None, key=None):
  """SGD subroutine

  x - [n, p] tensor
  s - [n, 1] tensor
  llik - function returning [n, p] tensor
  params - list of tensor [1, p]

  """
  if log_dir is not None:
    writer = tb.SummaryWriter(log_dir=log_dir)
  # Important: this is required for EBPMDataset, but TensorDataset does not
  # define collate_fn.
  #
  # The default value of this kwarg in td.DataLoader.__init__ has changed since
  # torch 0.4.1 (which we are using)
  collate_fn = getattr(data, 'collate_fn', td.dataloader.default_collate)
  data = td.DataLoader(data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                       # Important: only CPU memory can be pinned
                       pin_memory=not torch.cuda.is_available(),
                       collate_fn=collate_fn)
  global_step = 0
  opt = torch.optim.RMSprop(params, lr=lr)
  for epoch in range(num_epochs):
    for batch in data:
      opt.zero_grad()
      x = batch.pop(0)
      s = batch.pop(0)
      if onehot is not None:
        l = batch.pop(0)
      else:
        l = None
      if design is not None:
        z = batch.pop(0)
      else:
        z = None
      # Important: params are assumed to be provided in the order assumed by llik
      loss = -llik(x, s, *params, onehot=l, design=z).sum()
      if torch.isnan(loss):
        raise RuntimeError('nan loss')
      if log_dir is not None:
        writer.add_scalar(f'loss/{key}', loss, global_step)
        for k, v in zip(['log_mu', 'log_phi', 'logodds'], params):
          writer.add_scalar(f'params/{key}/{k}_norm', torch.norm(v), global_step)
      global_step += 1
      loss.backward()
      opt.step()
  if torch.cuda.is_available:
    result = [p.cpu().detach().numpy() for p in params]
  else:
    result = [p.detach().numpy() for p in params]
  # Clean up GPU memory
  del data
  del params[:]
  return result

def ebpm_point(x, s=None, onehot=None, design=None, lr=1e-2, batch_size=100, num_epochs=100, shuffle=False, log_dir=None):
  """Return fitted parameters assuming g is a point mass

  Note that if design is None, then the solution is analytic.

  Parameters:

  x - array-like [n, p]
  s - array-like [n, 1]
  onehot - array-like [n, m]
  design - array-like [n, q]
  lr - learning rate
  batch_size - number of data points for minibatch SGD
  num_epochs - number of passes through the data
  shuffle - randomly sample data points in each minibatch
  log_dir - output directory for tensorboard

  Return:

  log mu - array [m, p]

  """
  data, n, p, k, m, log_dir = _check_args(x, s, onehot, design, None, lr, batch_size, num_epochs, log_dir)
  if design is None:
    if onehot is None:
      return np.log(x.sum(axis=0, keepdims=True)) - np.log(s.sum())
    else:
      return np.log(onehot.T @ x) - np.log(onehot.T @ s)
  else:
    if torch.cuda.is_available():
      device = 'cuda'
    else:
      device = 'cpu'
    log_mean = torch.zeros([k, p], dtype=torch.float, requires_grad=True, device=device)
    beta = torch.zeros([m, p], dtype=torch.float, requires_grad=True, device=device)
    params = [log_mean, beta]
    return _sgd(data, onehot=onehot, design=design, llik=_pois_llik,
                params=params, key='pois', lr=lr, batch_size=batch_size,
                num_epochs=num_epochs, shuffle=shuffle, log_dir=log_dir)

def ebpm_gamma(x, s=None, onehot=None, design=None, init=None, lr=1e-2, batch_size=100, num_epochs=100, shuffle=False, log_dir=None):
  """Return fitted parameters assuming g is a Gamma distribution

  Parameters:

  x - array-like [n, p]
  s - array-like [n, 1]
  onehot - array-like [n, m]
  design - array-like [n, q]
  init - (log_mu, log_phi) [1, p]
  lr - learning rate
  batch_size - number of data points for minibatch SGD
  num_epochs - number of passes through the data
  shuffle - randomly sample data points in each minibatch
  log_dir - output directory for tensorboard

  Return:

  log mu - array [m, p]
  neg_log phi - array[m, p]

  """
  data, n, p, k, m, log_dir = _check_args(x, s, onehot, design, init, lr, batch_size, num_epochs, log_dir)
  if torch.cuda.is_available():
    device = 'cuda'
  else:
    device = 'cpu'
  if init is None:
    log_mean = torch.zeros([k, p], dtype=torch.float, requires_grad=True, device=device)
    log_inv_disp = torch.zeros([k, p], dtype=torch.float, requires_grad=True, device=device)
  else:
    log_mean = torch.tensor(init[0], dtype=torch.float, requires_grad=True, device=device)
    log_inv_disp = torch.tensor(init[1], dtype=torch.float, requires_grad=True, device=device)
  params = [log_mean, log_inv_disp]
  if design is not None:
    beta = torch.zeros([m, p], dtype=torch.float, requires_grad=True, device=device)
    params.append(beta)
  return _sgd(data, onehot=onehot, design=design, llik=_nb_llik,
              params=params, key='gamma', lr=lr, batch_size=batch_size,
              num_epochs=num_epochs, shuffle=shuffle, log_dir=log_dir)

def ebpm_point_gamma(x, s=None, onehot=None, design=None, init=None, lr=1e-2, batch_size=100, num_epochs=100, shuffle=False, log_dir=None):
  """Return fitted parameters assuming g is a point-Gamma distribution


  Parameters:

  x - array-like [n, p]
  s - array-like [n, 1]
  onehot - array-like [n, m]
  design - array-like [n, q]
  init - (log_mu, log_phi) [1, p]
  lr - learning rate
  batch_size - number of data points for minibatch SGD
  num_epochs - number of passes through the data
  shuffle - randomly sample data points in each minibatch
  log_dir - output directory for tensorboard

  Return:

  log mu - array [m, p]
  neg_log phi - array[m, p]
  logit_pi - array[m, p]

  """
  data, n, p, k, m, log_dir = _check_args(x, s, onehot, design, init, lr, batch_size, num_epochs, log_dir)
  if init is None:
    init = ebpm_gamma(x, s, onehot=onehot, lr=lr, batch_size=batch_size, num_epochs=num_epochs, shuffle=shuffle, log_dir=log_dir)
  if torch.cuda.is_available():
    device = 'cuda'
  else:
    device = 'cpu'
  log_mean = torch.tensor(init[0], dtype=torch.float, requires_grad=True, device=device)
  log_inv_disp = torch.tensor(init[1], dtype=torch.float, requires_grad=True, device=device)
  # Important: start pi_j near zero
  logodds = torch.full([k, p], -8, dtype=torch.float, requires_grad=True, device=device)
  params = [log_mean, log_inv_disp, logodds]
  if design is not None:
    # Do not warm start this
    beta = torch.zeros([m, p], dtype=torch.float, requires_grad=True, device=device)
    params.append(beta)
  return _sgd(data, onehot=onehot, design=design, llik=_zinb_llik,
              params=params, key='point_gamma', lr=lr, batch_size=batch_size,
              num_epochs=num_epochs, shuffle=shuffle, log_dir=log_dir)
