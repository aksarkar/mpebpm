import numpy as np
import pytest
import scipy.sparse as ss
import scipy.special as sp
import scipy.stats as st
import mpebpm.sgd
import torch
import torch.utils.data as td

from fixtures import *

def test__pois_llik(simulate_point):
  x, s, log_mu, oracle_loss = simulate_point
  llik = mpebpm.sgd._pois_llik(torch.tensor(x, dtype=torch.float),
                               torch.tensor(s, dtype=torch.float),
                               torch.tensor(log_mu, dtype=torch.float),
                               beta=torch.zeros([1, x.shape[1]]),
                               design=torch.zeros([x.shape[0], 1])).sum()
  assert np.isclose(llik, -oracle_loss)

def test__nb_llik(simulate_gamma):
  x, s, log_mu, log_phi, oracle_loss = simulate_gamma
  llik = mpebpm.sgd._nb_llik(torch.tensor(x, dtype=torch.float),
                             torch.tensor(s, dtype=torch.float),
                             torch.tensor(log_mu, dtype=torch.float),
                             torch.tensor(-log_phi, dtype=torch.float)).sum()
  assert np.isclose(llik, -oracle_loss)

def test__zinb_llik(simulate_gamma):
  x, s, log_mu, log_phi, oracle_loss = simulate_gamma
  llik = mpebpm.sgd._zinb_llik(torch.tensor(x, dtype=torch.float),
                               torch.tensor(s, dtype=torch.float),
                               torch.tensor(log_mu, dtype=torch.float),
                               torch.tensor(-log_phi, dtype=torch.float),
                               torch.tensor(-100, dtype=torch.float)).sum()
  assert np.isclose(llik, -oracle_loss)

def test__zinb_llik_zinb_data(simulate_point_gamma):
  x, s, log_mu, log_phi, logodds, oracle_loss = simulate_point_gamma
  n, p = x.shape
  llik = mpebpm.sgd._zinb_llik(torch.tensor(x, dtype=torch.float),
                               torch.tensor(s, dtype=torch.float),
                               torch.tensor(log_mu, dtype=torch.float),
                               torch.tensor(-log_phi, dtype=torch.float),
                               torch.tensor(logodds, dtype=torch.float)).sum()
  assert np.isclose(llik, -oracle_loss)

def test_ebpm_point_analytic(simulate_point):
  x, s, log_mu, l0 = simulate_point
  n, p = x.shape
  log_mu_hat = mpebpm.sgd.ebpm_point(x, s)
  l1 = -st.poisson(mu=s * np.exp(log_mu_hat)).logpmf(x).sum()
  assert np.isfinite(log_mu_hat).all()
  assert log_mu_hat.shape == (1, p)
  assert l1 < l0

def test_ebpm_point_analytic_onehot(simulate_point):
  x, s, log_mu, l0 = simulate_point
  n, p = x.shape
  onehot = np.ones((x.shape[0], 1))
  log_mu_hat = mpebpm.sgd.ebpm_point(x, s, onehot=onehot)
  l1 = -st.poisson(mu=s * np.exp(log_mu_hat)).logpmf(x).sum()
  assert np.isfinite(log_mu_hat).all()
  assert log_mu_hat.shape == (1, p)
  assert l1 < l0
  
def test_ebpm_point_sgd(simulate_point):
  x, s, log_mu, l0 = simulate_point
  n, p = x.shape
  design = np.zeros((n, 1))
  log_mu_hat, beta_hat = mpebpm.sgd.ebpm_point(x, s, design=design, batch_size=1, num_epochs=20)
  l1 = -st.poisson(mu=s * np.exp(log_mu_hat)).logpmf(x).sum()
  assert np.isfinite(log_mu_hat).all()
  assert np.isfinite(beta_hat).all()
  assert np.isclose(beta_hat, 0).all()
  assert log_mu_hat.shape == (1, p)
  assert beta_hat.shape == (1, p)
  # TODO: this doesn't beat oracle?

def test_ebpm_gamma_batch(simulate_gamma):
  x, s, log_mu, log_phi, l0 = simulate_gamma
  n, p = x.shape
  log_mu_hat, neg_log_phi_hat = mpebpm.sgd.ebpm_gamma(x, s, batch_size=n, num_epochs=2000)
  l1 = -st.nbinom(n=np.exp(neg_log_phi_hat), p=1 / (1 + s * np.exp(log_mu_hat - neg_log_phi_hat))).logpmf(x).sum()
  assert np.isfinite(log_mu_hat).all()
  assert np.isfinite(neg_log_phi_hat).all()
  assert log_mu_hat.shape == (1, p)
  assert neg_log_phi_hat.shape == (1, p)
  assert l1 < l0

def test_ebpm_gamma_minibatch(simulate_gamma):
  x, s, log_mu, log_phi, l0 = simulate_gamma
  n, p = x.shape
  log_mu_hat, neg_log_phi_hat = mpebpm.sgd.ebpm_gamma(x, s, batch_size=100, num_epochs=500, lr=1e-2)
  l1 = -st.nbinom(n=np.exp(neg_log_phi_hat), p=1 / (1 + s * np.exp(log_mu_hat - neg_log_phi_hat))).logpmf(x).sum()
  assert np.isfinite(log_mu_hat).all()
  assert np.isfinite(neg_log_phi_hat).all()
  assert log_mu_hat.shape == (1, p)
  assert neg_log_phi_hat.shape == (1, p)
  assert l1 < l0

def test_ebpm_gamma_minibatch_shuffle(simulate_gamma):
  x, s, log_mu, log_phi, l0 = simulate_gamma
  n, p = x.shape
  log_mu_hat, neg_log_phi_hat = mpebpm.sgd.ebpm_gamma(x, s, batch_size=100, num_epochs=500, lr=1e-2, shuffle=True)
  l1 = -st.nbinom(n=np.exp(neg_log_phi_hat), p=1 / (1 + s * np.exp(log_mu_hat - neg_log_phi_hat))).logpmf(x).sum()
  assert np.isfinite(log_mu_hat).all()
  assert np.isfinite(neg_log_phi_hat).all()
  assert log_mu_hat.shape == (1, p)
  assert neg_log_phi_hat.shape == (1, p)
  assert l1 < l0

def test_ebpm_gamma_sgd(simulate_gamma):
  x, s, log_mu, log_phi, l0 = simulate_gamma
  n, p = x.shape
  # Important: learning rate has to lowered to compensate for increased
  # variance in gradient estimator
  log_mu_hat, neg_log_phi_hat = mpebpm.sgd.ebpm_gamma(x, s, batch_size=1, num_epochs=10, lr=5e-3)
  l1 = -st.nbinom(n=np.exp(neg_log_phi_hat), p=1 / (1 + s * np.exp(log_mu_hat - neg_log_phi_hat))).logpmf(x).sum()
  assert np.isfinite(log_mu_hat).all()
  assert np.isfinite(neg_log_phi_hat).all()
  assert log_mu_hat.shape == (1, p)
  assert neg_log_phi_hat.shape == (1, p)
  assert l1 < l0

def test_ebpm_gamma_batch_onehot(simulate_gamma):
  x, s, log_mu, log_phi, l0 = simulate_gamma
  n, p = x.shape
  log_mu_hat, neg_log_phi_hat = mpebpm.sgd.ebpm_gamma(x, s, onehot=np.ones((n, 1)), batch_size=n, num_epochs=2000)
  l1 = -st.nbinom(n=np.exp(neg_log_phi_hat), p=1 / (1 + s * np.exp(log_mu_hat - neg_log_phi_hat))).logpmf(x).sum()
  assert np.isfinite(log_mu_hat).all()
  assert np.isfinite(neg_log_phi_hat).all()
  assert log_mu_hat.shape == (1, p)
  assert neg_log_phi_hat.shape == (1, p)
  assert l1 < l0

def test_ebpm_gamma_batch_onehot_2(simulate_gamma):
  x, s, log_mu, log_phi, l0 = simulate_gamma
  n, p = x.shape
  z = np.random.uniform(size=n) < 0.5
  onehot = np.vstack((z, ~z)).T.astype(float)
  log_mu_hat, neg_log_phi_hat = mpebpm.sgd.ebpm_gamma(x, s, onehot=onehot, batch_size=1, num_epochs=10)
  l1 = -st.nbinom(n=onehot @ np.exp(neg_log_phi_hat), p=1 / (1 + s * onehot @ np.exp(log_mu_hat - neg_log_phi_hat))).logpmf(x).sum()
  assert np.isfinite(log_mu_hat).all()
  assert np.isfinite(neg_log_phi_hat).all()
  assert log_mu_hat.shape == (2, p)
  assert neg_log_phi_hat.shape == (2, p)
  # TODO: this doesn't beat oracle log likelihood?

def test_ebpm_gamma_batch_design(simulate_gamma):
  x, s, log_mu, log_phi, l0 = simulate_gamma
  n, p = x.shape
  log_mu_hat, neg_log_phi_hat, beta_hat = mpebpm.sgd.ebpm_gamma(x, s, design=np.zeros((n, 1)), batch_size=1, num_epochs=20)
  l1 = -st.nbinom(n=np.exp(neg_log_phi_hat), p=1 / (1 + s * np.exp(log_mu_hat - neg_log_phi_hat))).logpmf(x).sum()
  assert np.isfinite(log_mu_hat).all()
  assert np.isfinite(neg_log_phi_hat).all()
  assert np.isfinite(beta_hat).all()
  assert log_mu_hat.shape == (1, p)
  assert neg_log_phi_hat.shape == (1, p)
  assert beta_hat.shape == (1, p)
  # TODO: this doesn't beat oracle log likelihood?

def test_ebpm_gamma_batch_onehot_design(simulate_gamma):
  x, s, log_mu, log_phi, l0 = simulate_gamma
  n, p = x.shape
  log_mu_hat, neg_log_phi_hat, beta_hat = mpebpm.sgd.ebpm_gamma(x, s, onehot=np.ones((n, 1)), design=np.zeros((n, 1)), batch_size=1, num_epochs=20)
  l1 = -st.nbinom(n=np.exp(neg_log_phi_hat), p=1 / (1 + s * np.exp(log_mu_hat - neg_log_phi_hat))).logpmf(x).sum()
  assert np.isfinite(log_mu_hat).all()
  assert np.isfinite(neg_log_phi_hat).all()
  assert np.isfinite(beta_hat).all()
  assert log_mu_hat.shape == (1, p)
  assert neg_log_phi_hat.shape == (1, p)
  assert beta_hat.shape == (1, p)
  # TODO: this doesn't beat oracle log likelihood?

def test_ebpm_point_gamma_oracle_init(simulate_point_gamma):
  x, s, log_mu, log_phi, logodds, l0 = simulate_point_gamma
  n, p = x.shape
  log_mu_hat, neg_log_phi_hat, logodds_hat = mpebpm.sgd.ebpm_point_gamma(x, s, init=(log_mu, -log_phi), batch_size=n, num_epochs=2000)
  pi0hat = sp.expit(logodds_hat)
  Fhat = st.nbinom(n=np.exp(neg_log_phi_hat), p=1 / (1 + s * np.exp(log_mu_hat - neg_log_phi_hat)))
  llik_nonzero = np.log(1 - pi0hat) + Fhat.logpmf(x)
  l1 = -np.where(x < 1, np.log(pi0hat + np.exp(llik_nonzero)), llik_nonzero).sum()
  assert np.isfinite(log_mu_hat).all()
  assert np.isfinite(neg_log_phi_hat).all()
  assert np.isfinite(logodds_hat).all()
  assert log_mu_hat.shape == (1, p)
  assert neg_log_phi_hat.shape == (1, p)
  assert logodds.shape == (1, p)
  assert l1 < l0

def test_ebpm_point_gamma_batch(simulate_point_gamma):
  x, s, log_mu, log_phi, logodds, l0 = simulate_point_gamma
  n, p = x.shape
  log_mu_hat, neg_log_phi_hat, logodds_hat = mpebpm.sgd.ebpm_point_gamma(x, s, batch_size=n, num_epochs=2000)
  pi0hat = sp.expit(logodds_hat)
  Fhat = st.nbinom(n=np.exp(neg_log_phi_hat), p=1 / (1 + s * np.exp(log_mu_hat - neg_log_phi_hat)))
  llik_nonzero = np.log(1 - pi0hat) + Fhat.logpmf(x)
  l1 = -np.where(x < 1, np.log(pi0hat + np.exp(llik_nonzero)), llik_nonzero).sum()
  assert np.isfinite(log_mu_hat).all()
  assert np.isfinite(neg_log_phi_hat).all()
  assert np.isfinite(logodds_hat).all()
  assert log_mu_hat.shape == (1, p)
  assert neg_log_phi_hat.shape == (1, p)
  assert logodds.shape == (1, p)
  assert l1 < l0

def test_ebpm_point_gamma_minibatch(simulate_point_gamma):
  x, s, log_mu, log_phi, logodds, l0 = simulate_point_gamma
  n, p = x.shape
  log_mu_hat, neg_log_phi_hat, logodds_hat = mpebpm.sgd.ebpm_point_gamma(x, s, batch_size=100, num_epochs=500)
  pi0hat = sp.expit(logodds_hat)
  Fhat = st.nbinom(n=np.exp(neg_log_phi_hat), p=1 / (1 + s * np.exp(log_mu_hat - neg_log_phi_hat)))
  llik_nonzero = np.log(1 - pi0hat) + Fhat.logpmf(x)
  l1 = -np.where(x < 1, np.log(pi0hat + np.exp(llik_nonzero)), llik_nonzero).sum()
  assert np.isfinite(log_mu_hat).all()
  assert np.isfinite(neg_log_phi_hat).all()
  assert np.isfinite(logodds_hat).all()
  assert log_mu_hat.shape == (1, p)
  assert neg_log_phi_hat.shape == (1, p)
  assert logodds.shape == (1, p)
  assert l1 < l0

def test_ebpm_point_gamma_minibatch_shuffle(simulate_point_gamma):
  x, s, log_mu, log_phi, logodds, l0 = simulate_point_gamma
  n, p = x.shape
  log_mu_hat, neg_log_phi_hat, logodds_hat = mpebpm.sgd.ebpm_point_gamma(x, s, batch_size=100, num_epochs=500, shuffle=True)
  pi0hat = sp.expit(logodds_hat)
  Fhat = st.nbinom(n=np.exp(neg_log_phi_hat), p=1 / (1 + s * np.exp(log_mu_hat - neg_log_phi_hat)))
  llik_nonzero = np.log(1 - pi0hat) + Fhat.logpmf(x)
  l1 = -np.where(x < 1, np.log(pi0hat + np.exp(llik_nonzero)), llik_nonzero).sum()
  assert np.isfinite(log_mu_hat).all()
  assert np.isfinite(neg_log_phi_hat).all()
  assert np.isfinite(logodds_hat).all()
  assert log_mu_hat.shape == (1, p)
  assert neg_log_phi_hat.shape == (1, p)
  assert logodds.shape == (1, p)
  assert l1 < l0

def test_ebpm_point_gamma_sparse(simulate_point_gamma):
  x, s, log_mu, log_phi, logodds, l0 = simulate_point_gamma
  y = ss.csr_matrix(x)
  n, p = x.shape
  log_mu_hat, neg_log_phi_hat, logodds_hat = mpebpm.sgd.ebpm_point_gamma(y, s, batch_size=100, num_epochs=500)
  pi0hat = sp.expit(logodds_hat)
  Fhat = st.nbinom(n=np.exp(neg_log_phi_hat), p=1 / (1 + s * np.exp(log_mu_hat - neg_log_phi_hat)))
  llik_nonzero = np.log(1 - pi0hat) + Fhat.logpmf(x)
  l1 = -np.where(x < 1, np.log(pi0hat + np.exp(llik_nonzero)), llik_nonzero).sum()
  assert np.isfinite(log_mu_hat).all()
  assert np.isfinite(neg_log_phi_hat).all()
  assert np.isfinite(logodds_hat).all()
  assert log_mu_hat.shape == (1, p)
  assert neg_log_phi_hat.shape == (1, p)
  assert logodds.shape == (1, p)
  assert l1 < l0

def test_ebpm_point_gamma_sgd(simulate_point_gamma):
  x, s, log_mu, log_phi, logodds, l0 = simulate_point_gamma
  n, p = x.shape
  log_mu_hat, neg_log_phi_hat, logodds_hat = mpebpm.sgd.ebpm_point_gamma(x, s, batch_size=1, num_epochs=30, lr=5e-3)
  pi0hat = sp.expit(logodds_hat)
  Fhat = st.nbinom(n=np.exp(neg_log_phi_hat), p=1 / (1 + s * np.exp(log_mu_hat - neg_log_phi_hat)))
  llik_nonzero = np.log(1 - pi0hat) + Fhat.logpmf(x)
  l1 = -np.where(x < 1, np.log(pi0hat + np.exp(llik_nonzero)), llik_nonzero).sum()
  assert np.isfinite(log_mu_hat).all()
  assert np.isfinite(neg_log_phi_hat).all()
  assert np.isfinite(logodds_hat).all()
  assert log_mu_hat.shape == (1, p)
  assert neg_log_phi_hat.shape == (1, p)
  assert logodds.shape == (1, p)
  assert l1 < l0
