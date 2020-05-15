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

def _nb_mix_loss(z, x, s, log_mean, log_inv_disp, eps=1e-16):
  L = _nb_mix_llik(x, s, log_mean, log_inv_disp)
  m, _ = L.max(dim=1, keepdim=True)
  return -(m + torch.log(z * torch.exp(L - m) + eps)).mean()

def ebpm_gam_mix(x, s, k, y=None, lr=1e-2, batch_size=100, num_epochs=100, max_em_iters=10, tol=1e-3, shuffle=False, num_workers=0, log_dir=None):
  data, n, p, _, _, log_dir = mpebpm.sgd._check_args(x, s, None, None, None, lr, batch_size, num_epochs, log_dir)
  collate_fn = getattr(data, 'collate_fn', td.dataloader.default_collate)
  data = td.DataLoader(data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                       # Important: only CPU memory can be pinned
                       pin_memory=not torch.cuda.is_available(),
                       collate_fn=collate_fn)

  if torch.cuda.is_available():
    device = 'cuda:0'
  else:
    device = 'cpu:0'
  z = torch.rand([n, k], dtype=torch.float, requires_grad=False, device=device)
  log_mean = torch.zeros([k, p], dtype=torch.float, requires_grad=True, device=device)
  log_inv_disp = torch.zeros([k, p], dtype=torch.float, requires_grad=True, device=device)
  params = [log_mean, log_inv_disp]

  if log_dir is not None:
    writer = tb.SummaryWriter(log_dir=log_dir)
  opt = torch.optim.RMSprop(params, lr=lr)
  global_step = 0
  for t in range(max_em_iters):
    for _ in range(num_epochs):
      for x, s in data:
        opt.zero_grad()
        loss = _nb_mix_loss(z, x, s, log_mean, log_inv_disp)
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
      update = _nb_mix_loss(z, x, s, log_mean, log_inv_disp)
      if y is not None:
        l = torch.nn.functional.binary_cross_entropy(z, y)
    if writer is not None and y is not None:
      writer.add_scalar(f'loss/cross_entropy', l, t)
    if update > loss:
      raise RuntimeError('loss increased after E step')
    elif loss - update < tol:
      break
  else:
    raise RuntimeError(f'failed to converge in max_em_iters ({max_em_iters}) iterations')
  params.append(z)
  if torch.cuda.is_available:
    result = [p.detach().cpu().numpy() for p in params]
  else:
    result = [p.detach().numpy() for p in params]
  del data
  del params[:]
  return result
