r"""Noisy topic model

We propose a noisy topic model combining a Poisson measurement model
[Sarkar2020]_ with the low rank expression model

  .. math::

    \lambda_{ij} \sim g_{ij}(\cdot) = \operatorname{Gamma}(\exp(-(\mathbf{L}\mathbf{H}')_{ij}), \exp(-(\mathbf{L}(\mathbf{F} + \mathbf{H})')_{ij})),

where the Gamma distribution is parameterized by shape and rate,
:math:`\mathbf{L}` is an :math:`n \times K` matrix, :math:`l_{ik} \geq 0`,
:math:`\sum_k l_{ik} = 1`, :math:`\mathbf{F}` is a :math:`K \times p` matrix of
log mean parameters, and :math:`\mathbf{H}` is a :math:`K \times p` matrix of
log dispersion parameters. Marginally, this yields an NB observation model,
where the means and dispersions for each cell at each gene are convex
combinations of the topic means and dispersions. The model is fit by maximizing
the marginal likelihood via stochastic gradient descent, using an inference
network to amortize inference over observed data.

This observation model is a linearization of the DCA observation model
[Eraslan2019]_ (where the decoder is linear rather than non-linear), and is a
generalization of the LDVAE observation model [Svensson2020]_ (where dispersion
parameters at each gene are not assumed to be constant over observations).

"""
import mpebpm.sgd
import torch
import torch.utils.data as td
import torch.utils.tensorboard as tb

def _nb_llik(x, s, L, F, H):
  mean = s * torch.exp(L @ F)
  inv_disp = torch.exp(L @ H)
  return (x * torch.log(mean / inv_disp)
          - x * torch.log(1 + mean / inv_disp)
          - inv_disp * torch.log(1 + mean / inv_disp)
          + torch.lgamma(x + inv_disp)
          - torch.lgamma(inv_disp)
          - torch.lgamma(x + 1))

class Log1p(torch.nn.Module):
  """Apply the log1p transform elementwise"""
  def __init__(self):
    super().__init__()

  def forward(self, x):
    """Apply the log1p transform elementwise"""
    return torch.log1p(x)

class Encoder(torch.nn.Module):
  """Inference network for noisy topic model"""
  def __init__(self, input_dim, hidden_dim, latent_dim):
    """Initialize the inference network

    Args:

      input_dim (`int`): number of genes
      hidden_dim (`int`): hidden layer dimension (inference network)
      latent_dim (`int`): number of topics

    """
    super().__init__()
    self.net = torch.nn.Sequential(
      Log1p(),
      torch.nn.Linear(input_dim, hidden_dim),
      torch.nn.ReLU(),
      torch.nn.Linear(hidden_dim, hidden_dim),
      torch.nn.ReLU(),
      torch.nn.Linear(hidden_dim, latent_dim),
      torch.nn.Softmax(dim=1)
    )

  def forward(self, x):
    """Return the latent loadings for observations `x`."""
    return self.net(x)

class NoisyTopicModel(torch.nn.Module):
  def __init__(self, input_dim, latent_dim, hidden_dim=128):
    """Initialize the noisy topic model

    Args:

      input_dim (`int`): number of genes
      hidden_dim (`int`): hidden layer dimension (inference network)
      latent_dim (`int`): number of topics

    Attributes:

      encoder (`mpebpm.topics.Encoder`): inference network
      F (`torch.tensor`, ``[latent_dim, input_dim]``): log mean parameters
      H (`torch.tensor`, ``[latent_dim, input_dim]``): log inverse dispersion parameters

    """
    super().__init__()
    self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
    self.F = torch.nn.Parameter(0.01 * torch.randn([latent_dim, input_dim]))
    self.H = torch.nn.Parameter(0.01 * torch.randn([latent_dim, input_dim]))

  def forward(self, x, s):
    """Return the negative log likelihood of the observed counts `x` and size
    factors `s`.

    """
    l = self.encoder.forward(x)
    return -_nb_llik(x, s, l, self.F, self.H).sum()

  def fit(self, x, s=None, lr=1e-2, batch_size=100, num_epochs=100, shuffle=False, log_dir=None):
    r"""Fit the model to observed counts ``x``.

    Args:

      x (array-like ``[n, p]``): observed counts
      s (array-like ``[n, 1]``, optional): size factors 
      lr (`float`): learning rate
      batch_size (`int`): number of data points for minibatch SGD
      num_epochs (`int`): number of passes through the data
      shuffle (`bool`): randomly sample data points in each minibatch
      log_dir (`str`): output directory for TensorBoard

    Returns:

      `self`

    """    
    data, n, p, k, m, log_dir = mpebpm.sgd._check_args(x, s, None, None, None, lr, batch_size, num_epochs, log_dir)
    if torch.cuda.is_available():
      self.cuda()
    if log_dir is not None:
      writer = tb.SummaryWriter(log_dir)
    global_step = 0
    collate_fn = getattr(data, 'collate_fn', td.dataloader.default_collate)
    data = td.DataLoader(data, batch_size=batch_size, shuffle=shuffle,
                         pin_memory=not torch.cuda.is_available(),
                         collate_fn=collate_fn)
    global_step = 0
    opt = torch.optim.RMSprop(self.parameters(), lr=lr)
    for epoch in range(num_epochs):
      for x, s in data:
        opt.zero_grad()
        loss = self.forward(x, s)
        if torch.isnan(loss):
          raise RuntimeError('nan loss')
        if log_dir is not None:
          writer.add_scalar('loss', loss, global_step)
        global_step += 1
        loss.backward()
        opt.step()
    # Clean up GPU memory
    del data
    return self
