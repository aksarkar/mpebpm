"""Massively parallel EBPM for model-based clustering

"""
import mpebpm.sgd
import torch
import torch.utils.data as td
import torch.utils.tensorboard as tb

def _nb_mix_llik(x, s, log_mean, log_inv_disp):
  r"""Return \sum_j ln p(x_ij | s_i, z_ik = 1, g_jk)

  x_ij ~ Poisson(s_i lambda_i)
  lambda_ij \mid z_ik = 1 ~ g_jk = Gamma(exp(log_inv_disp_jk), exp(log_mean_jk - log_inv_disp_jk))

  x - [n, p] tensor
  s - [n, 1] tensor
  log_mean - [k, p] tensor
  log_inv_disp - [k, p] tensor

  """
  # [n, 1, p]
  x = x.unsqueeze(1)
  # [n, 1, 1] * [k, p] = [n, k, p]
  mean = s.unsqueeze(-1) * torch.exp(log_mean)
  # [1, k, p]
  inv_disp = torch.exp(log_inv_disp).unsqueeze(0)
  L = (x * torch.log(mean / inv_disp)
       - x * torch.log(1 + mean / inv_disp)
       - inv_disp * torch.log(1 + mean / inv_disp)
       + torch.lgamma(x + inv_disp)
       - torch.lgamma(inv_disp)
       - torch.lgamma(x + 1))
  return L.sum(dim=2)

def _nb_mix_loss(z, x, s, k, log_mean, log_inv_disp, eps=1e-16):
  L = _nb_mix_llik(x, s, log_mean, log_inv_disp)
  # TODO: fixed uniform prior
  return -(z * (L - torch.log(k))).mean()

def ebpm_gam_mix_em(x, s, k, y=None, lr=1e-2, num_epochs=100, max_em_iters=10, shuffle=False, num_workers=0, log_dir=None):
  data, n, p, _, _, log_dir = mpebpm.sgd._check_args(x, s, None, None, None, lr, x.shape[0], num_epochs, log_dir)
  collate_fn = getattr(data, 'collate_fn', td.dataloader.default_collate)
  # TODO: very hard to do minibatch SGD in this scheme, since we need to batch
  # parameter z along with the data.
  data = td.DataLoader(data, batch_size=n, shuffle=shuffle, num_workers=num_workers,
                       # Important: only CPU memory can be pinned
                       pin_memory=not torch.cuda.is_available(),
                       collate_fn=collate_fn)

  if torch.cuda.is_available():
    device = 'cuda:0'
  else:
    device = 'cpu:0'
  z = torch.rand([n, k], dtype=torch.float, requires_grad=False, device=device)
  z /= z.sum(dim=1, keepdim=True)
  log_mean = torch.log(torch.matmul(z.T, torch.tensor(x, dtype=torch.float).cuda())) - torch.log(torch.matmul(z.T, torch.tensor(s, dtype=torch.float).cuda()))
  log_mean.requires_grad = True
  log_inv_disp = torch.zeros([k, p], dtype=torch.float, requires_grad=True, device=device)
  k = torch.tensor(k, dtype=torch.float, requires_grad=False, device=device)
  params = [log_mean, log_inv_disp]

  if log_dir is not None:
    writer = tb.SummaryWriter(log_dir=log_dir)
  global_step = 0
  for t in range(max_em_iters):
    # Important: this needs to restart after each E step
    opt = torch.optim.RMSprop(params, lr=lr)
    for _ in range(num_epochs):
      for x, s in data:
        opt.zero_grad()
        loss = _nb_mix_loss(z, x, s, k, log_mean, log_inv_disp)
        if torch.isnan(loss):
          raise RuntimeError('nan loss')
        loss.backward(retain_graph=True)
        opt.step()
        if log_dir is not None:
          writer.add_scalar(f'loss/nll', loss, global_step)
        global_step += 1
    L = _nb_mix_llik(x, s, log_mean, log_inv_disp)
    z = torch.nn.functional.softmax(L, dim=1)
    with torch.no_grad():
      update = _nb_mix_loss(z, x, s, k, log_mean, log_inv_disp)
      assert update <= loss
      if y is not None:
        l = torch.min(torch.nn.functional.binary_cross_entropy(z, y),
                      torch.nn.functional.binary_cross_entropy(1 - z, y))
        if log_dir is not None:
          writer.add_scalar(f'loss/cross_entropy', l, t)
  params.append(z)
  if torch.cuda.is_available:
    result = [p.detach().cpu().numpy() for p in params]
  else:
    result = [p.detach().numpy() for p in params]
  del data
  del params[:]
  return result

class EBPMGammaMix(torch.nn.Module):
  """Amortized inference for model-based clustering"""
  def __init__(self, p, k, log_mean=None, log_inv_disp=None, hidden_dim=128):
    """Initialize the model

    p - number of genes
    k - number of clusters
    hidden_dim - hidden layer size in the encoder network

    """
    super().__init__()
    self.encoder = torch.nn.Sequential(
      torch.nn.Linear(p, hidden_dim),
      torch.nn.ReLU(),
      torch.nn.Linear(hidden_dim, k),
      torch.nn.Softmax(dim=1),
    )
    if log_mean is None:
      log_mean = torch.randn([k, p])
    else:
      log_mean = torch.tensor(log_mean, dtype=torch.float)
    if log_inv_disp is None:
      log_inv_disp = torch.zeros([k, p])
    else:
      log_inv_disp = torch.tensor(log_inv_disp, dtype=torch.float)
    self.log_mean = torch.nn.Parameter(log_mean)
    self.log_inv_disp = torch.nn.Parameter(log_inv_disp)
    self.k = torch.tensor(k, dtype=torch.float)

  def forward(self, x):
    return self.encoder(x)

  def fit(self, x, s, y=None, lr=1e-2, batch_size=100, num_pretrain=50, num_epochs=100, shuffle=False, num_workers=0, log_dir=None):
    """Fit the model

    x - array-like [n, p]
    s - array-like [n, 1]
    y - one-hot encoded true labels (for testing) [n, k]

    """
    if torch.cuda.is_available():
      self.cuda()
    # This moves input tensors to the GPU if available.
    data, n, p, _, _, log_dir = mpebpm.sgd._check_args(x, s, None, y, None, lr, batch_size, num_epochs, log_dir)
    collate_fn = getattr(data, 'collate_fn', td.dataloader.default_collate)
    data = td.DataLoader(data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                         # Important: only CPU memory can be pinned
                         pin_memory=not torch.cuda.is_available(),
                         collate_fn=collate_fn)
    if log_dir is not None:
      writer = tb.SummaryWriter(log_dir=log_dir)
    opt = torch.optim.RMSprop(self.parameters(), lr=lr)
    global_step = 0
    for t in range(num_epochs):
      if t < num_pretrain:
        self.log_mean.requires_grad = False
        self.log_inv_disp.requires_grad = False
      else:
        self.log_mean.requires_grad = True
        self.log_inv_disp.requires_grad = True
      for batch in data:
        x = batch.pop(0)
        s = batch.pop(0)
        if batch:
          y = batch.pop(0)
        else:
          y = None
        opt.zero_grad()
        z = self.encoder.forward(x)
        err = -_nb_mix_loss(z, x, s, self.log_mean, self.log_inv_disp)
        kl = (z * (torch.log(z + 1e-16) + torch.log(self.k))).sum()
        elbo = err - kl
        loss = -elbo
        loss.retain_grad()
        if torch.isnan(loss):
          raise RuntimeError('nan loss')
        loss.backward()
        opt.step()
        if log_dir is not None:
          writer.add_scalar(f'loss/err', err, global_step)
          writer.add_scalar(f'loss/kl', kl, global_step)
          writer.add_scalar(f'loss/elbo', elbo, global_step)
          with torch.no_grad():
            lz = torch.nn.functional.binary_cross_entropy(z, y)
            lx = _nb_mix_llik(x, s, self.log_mean, self.log_inv_disp).sum(dim=0)
          writer.add_scalar('loss/nll/diff', lx[0] - lx[1], global_step)
          if y is not None:
            writer.add_scalar(f'loss/cross_entropy', lz, global_step)
          writer.add_scalar('encoder/w1_l2', torch.norm(self.encoder[0].weight.grad), global_step)
          writer.add_scalar('encoder/w2_l2', torch.norm(self.encoder[2].weight.grad), global_step)
        global_step += 1
    del data
    return self
