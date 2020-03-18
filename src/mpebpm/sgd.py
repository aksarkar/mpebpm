"""Empirical Bayes Poisson Means via SGD

These implementations are specialized for two scenarios:

1. Fitting p EBPM problems on n samples in parallel, where n, p may be large

2. Fitting p * k EBPM problems in parallel, where the n samples are assumed to
   be drawn from a discrete (known) choice of k different priors (for each
   gene)

"""
import mpebpm.sparse
import scipy.sparse as ss
import torch
import torch.utils.data as td

def _nb_llik(x, s, log_mean, log_inv_disp, onehot=None):
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
  return (x * torch.log(mean / inv_disp)
          - x * torch.log(1 + mean / inv_disp)
          - inv_disp * torch.log(1 + mean / inv_disp)
          # Important: these terms are why we use inverse dispersion
          + torch.lgamma(x + inv_disp)
          - torch.lgamma(inv_disp)
          - torch.lgamma(x + 1))

def _zinb_llik(x, s, log_mean, log_inv_disp, logodds, onehot=None):
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
  nb_llik = _nb_llik(x, s, log_mean, log_inv_disp)
  softplus = torch.nn.functional.softplus
  case_zero = -softplus(-logodds) + softplus(nb_llik - logodds)
  case_non_zero = -softplus(logodds) + nb_llik
  return torch.where(torch.lt(x, 1), case_zero, case_non_zero)

def _check_args(x, s, onehot, init, lr, batch_size, max_epochs):
  """Check x, s, onehot and return a DataLoader"""
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
    onehot = mpebpm.sparse.CSRTensor(onehot.data, onehot.indices, onehot.indptr, dtype=torch.float)
  else:
    # Important: don't use onehot = torch.ones(n) because this might be big
    k = 1
  if torch.cuda.is_available:
    x = x.cuda()
    s = s.cuda()
    if onehot is not None:
      onehot = onehot.cuda()
  if onehot is not None:
    data = mpebpm.sparse.SparseDataset(x, s, onehot)
  else:
    data = mpebpm.sparse.SparseDataset(x, s)
  if init is None:
    pass
  elif init[0].shape != (k, p):
    raise ValueError(f'shape mismatch (log_mu): expected {(k, p)}, got {init[0].shape}')
  elif init[1].shape != (k, p):
    raise ValueError(f'shape mismatch (log_phi): expected {(k, p)}, got {init[1].shape}')
  elif len(init) > 2:
    raise ValueError('expected two values in init, got {len(init)}')
  if lr <= 0:
    raise ValueError('lr must be >= 0')
  if batch_size < 1:
    raise ValueError('batch_size must be >= 1')
  if max_epochs < 1:
    raise ValueError('max_epochs must be >= 1')
  return data, n, p, k

def _sgd(data, onehot, llik, params, lr=1e-2, batch_size=100, max_epochs=100, num_workers=0, verbose=False, trace=False):
  """SGD subroutine

  x - [n, p] tensor
  s - [n, 1] tensor
  llik - function returning [n, p] tensor
  params - list of tensor [1, p]

  """
  # Important: this is required for EBPMDataset, but TensorDataset does not
  # define collate_fn.
  #
  # The default value of this kwarg in td.DataLoader.__init__ has changed since
  # torch 0.4.1 (which we are using)
  collate_fn = getattr(data, 'collate_fn', td.dataloader.default_collate)
  data = td.DataLoader(data, batch_size=batch_size, num_workers=num_workers,
                       # Important: only CPU memory can be pinned
                       pin_memory=not torch.cuda.is_available(),
                       collate_fn=collate_fn)
  opt = torch.optim.RMSprop(params, lr=lr)
  param_trace = []
  loss = None
  for epoch in range(max_epochs):
    for (x, s) in data:
      opt.zero_grad()
      # Important: params are assumed to be provided in the order assumed by llik
      loss = -llik(x, s, *params).sum()
      if torch.isnan(loss):
        raise RuntimeError('nan loss')
      loss.backward()
      opt.step()
      if trace:
        # Important: this only works for scalar params
        param_trace.append([p.item() for p in params] + [loss.item()])
    if verbose:
      print(f'Epoch {epoch}:', loss.item())
  if torch.cuda.is_available:
    result = [p.cpu().detach().numpy() for p in params]
  else:
    result = [p.detach().numpy() for p in params]
  # Clean up GPU memory
  del params[:]
  result.append(loss.item())
  if trace:
    result.append(param_trace)
  return result

def ebpm_gamma(x, s=None, onehot=None, init=None, lr=1e-2, batch_size=100, max_epochs=100, verbose=False, trace=False):
  """Return fitted parameters and marginal log likelihood assuming g is a Gamma
distribution

  Important: returns log mu and -log phi

  x - array-like [n, p]
  s - array-like [n, 1]
  init - (log_mu, log_phi) [1, p]

  """
  data, n, p, k = _check_args(x, s, onehot, init, lr, batch_size, max_epochs)
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
  return _sgd(data, onehot=onehot, llik=_nb_llik, params=[log_mean, log_inv_disp],
              lr=lr, batch_size=batch_size, max_epochs=max_epochs, verbose=verbose,
              trace=trace)

def ebpm_point_gamma(x, s=None, onehot=None, init=None, lr=1e-2, batch_size=100, max_epochs=100, verbose=False, trace=False):
  """Return fitted parameters and marginal log likelihood assuming g is a Gamma
distribution

  Important: returns log mu, -log phi, logit pi

  x - array-like [n, p]
  s - array-like [n, 1]
  init - (log_mu, -log_phi) [1, p]

  """
  data, n, p, k = _check_args(x, s, onehot, init, lr, batch_size, max_epochs)
  if init is None:
    if verbose:
      print('Fitting ebpm_gamma to get initialization')
    res = ebpm_gamma(x, s, onehot=onehot, lr=lr, batch_size=batch_size, max_epochs=max_epochs, verbose=verbose)
    init = res[:-1]
  if torch.cuda.is_available():
    device = 'cuda'
  else:
    device = 'cpu'
  log_mean = torch.tensor(init[0], dtype=torch.float, requires_grad=True, device=device)
  log_inv_disp = torch.tensor(init[1], dtype=torch.float, requires_grad=True, device=device)
  # Important: start pi_j near zero (on the logit scale)
  logodds = torch.full([k, p], -8, dtype=torch.float, requires_grad=True, device=device)
  return _sgd(data, onehot=onehot, llik=_zinb_llik, params=[log_mean, log_inv_disp, logodds],
              lr=lr, batch_size=batch_size, max_epochs=max_epochs,
              verbose=verbose, trace=trace)
