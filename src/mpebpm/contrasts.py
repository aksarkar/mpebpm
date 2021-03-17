"""sklearn-style estimator for Gamma expression model with contrasts

"""
import numpy as np
import scipy.stats as st
import torch
import torch.utils.tensorboard as tb

class EBPMGammaContrasts(torch.nn.Module):
  def __init__(self, k, p, prior_log_mean, prior_log_inv_disp, prior_diff_scale=1.):
    assert prior_diff_scale > 0
    assert prior_log_mean.loc.shape == torch.Size([1, p])
    assert prior_log_inv_disp.loc.shape == torch.Size([1, p])
    super().__init__()
    self.prior_log_mean = prior_log_mean
    self.prior_log_inv_disp = prior_log_inv_disp
    self.prior_diff = torch.distributions.Normal(loc=0., scale=prior_scale)

    # TODO: scale is actually ln(2)
    self.q_log_mean_mean = torch.nn.Parameter(self.prior_log_mean.loc)
    self.q_log_mean_raw_scale = torch.nn.Parameter(torch.zeros([1, p]))
    self.q_log_inv_disp_mean = torch.nn.Parameter(self.prior_log_inv_disp.loc)
    self.q_log_inv_disp_raw_scale = torch.nn.Parameter(torch.zeros([1, p]))

    self.q_diff_mean_mean = torch.nn.Parameter(torch.zeros([k, p]))
    self.q_diff_mean_raw_scale = torch.nn.Parameter(torch.zeros([k, p]))
    self.q_diff_inv_disp_mean = torch.nn.Parameter(torch.zeros([k, p]))
    self.q_diff_inv_disp_raw_scale = torch.nn.Parameter(torch.zeros([k, p]))

  def forward(self, x, s, z, n_samples, writer=None, global_step=None):
    softplus = torch.nn.functional.softplus
    _kl = torch.distributions.kl.kl_divergence

    # We need to keep these around to draw samples
    q_log_mean = torch.distributions.Normal(loc=self.q_log_mean_mean, scale=softplus(self.q_log_mean_raw_scale))
    q_log_inv_disp = torch.distributions.Normal(loc=self.q_log_inv_disp_mean, scale=softplus(self.q_log_inv_disp_raw_scale))
    q_diff_mean = torch.distributions.Normal(loc=self.q_diff_mean_mean, scale=softplus(self.q_diff_mean_raw_scale))
    q_diff_inv_disp = torch.distributions.Normal(loc=self.q_diff_inv_disp_mean, scale=softplus(self.q_diff_inv_disp_raw_scale))

    # These can be computed without sampling
    kl_mean = _kl(q_log_mean, self.prior_log_mean).sum()
    kl_inv_disp = _kl(q_log_inv_disp, self.prior_log_inv_disp).sum()
    kl_diff_mean = _kl(q_diff_mean, self.prior_diff).sum()
    kl_diff_inv_disp = _kl(q_diff_inv_disp, self.prior_diff).sum()

    # [n_samples, 1, p]
    log_mean = q_log_mean.rsample([n_samples])
    log_inv_disp = q_log_inv_disp.rsample([n_samples])
    diff_mean = q_diff_mean.rsample([n_samples])
    diff_inv_disp = q_diff_inv_disp.rsample([n_samples])
    # [n_samples, k, p]
    a = torch.exp(log_inv_disp.T - z @ diff_inv_disp.T)
    b = torch.exp(-log_mean.T + log_inv_disp.T - z @ (diff_mean + diff_inv_disp).T)
    assert torch.isfinite(a).all()
    assert torch.isfinite(b).all()
    err = (x * torch.log(s / b)
           - x * torch.log(1 + s / b)
           - a * torch.log(1 + s / b)
           + torch.lgamma(x + a)
           - torch.lgamma(a)
           - torch.lgamma(x + 1)).mean(dim=0).sum()
    assert err <= 0
    elbo = err - kl_mean - kl_inv_disp - kl_diff_mean - kl_diff_inv_disp
    if writer is not None:
      writer.add_scalar('loss/err', err, global_step)
      writer.add_scalar('loss/kl_mean', kl_mean, global_step)
      writer.add_scalar('loss/kl_inv_disp', kl_inv_disp, global_step)
      writer.add_scalar('loss/kl_diff_mean', kl_diff_mean, global_step)
      writer.add_scalar('loss/kl_diff_inv_disp', kl_diff_inv_disp, global_step)
    return -elbo

  def fit(self, data, n_epochs, n_samples=1, log_dir=None, **kwargs):
    # TODO: include design matrix
    if log_dir is not None:
      writer = tb.SummaryWriter(log_dir)
    else:
      writer = None
    global_step = 0
    opt = torch.optim.RMSprop(self.parameters(), **kwargs)
    for epoch in range(n_epochs):
      for x, s, z in data:
        opt.zero_grad()
        loss = self.forward(x, s, z, n_samples, writer, global_step)
        if torch.isnan(loss):
          raise RuntimeError('nan loss')
        loss.backward()
        opt.step()
        global_step += 1
    return self

  @property
  def log_mean(self):
    return self.q_log_mean_mean.detach().numpy(), np.log1p(np.exp(self.q_log_mean_raw_scale.detach().numpy()))

  @property
  def diff_mean(self):
    return self.q_diff_mean_mean.detach().numpy(), np.log1p(np.exp(self.q_diff_mean_raw_scale.detach().numpy()))

  @property
  def lfsr_diff_mean(self):
    loc, scale = self.diff_mean
    return min(st.norm(loc, scale).cdf(0), st.norm(loc, scale).sf(0))

  @property
  def log_inv_disp(self):
    return self.q_log_inv_disp_mean.detach().numpy(), np.log1p(np.exp(self.q_log_inv_disp_raw_scale.detach().numpy()))

  @property
  def diff_inv_disp(self):
    return self.q_diff_inv_disp_mean.detach().numpy(), np.log1p(np.exp(self.q_diff_inv_disp_raw_scale.detach().numpy()))

  @property
  def lfsr_diff_inv_disp(self):
    loc, scale = self.diff_inv_disp
    return min(st.norm(loc, scale).cdf(0), st.norm(loc, scale).sf(0))
