import numpy as np
import pytest
import scipy.sparse as ss
import scipy.special as sp
import scipy.stats as st
import mpebpm.sgd
import torch
import torch.utils.data as td

from fixtures import *

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

def test_ebpm_gamma_batch(simulate_gamma):
  x, s, log_mu, log_phi, l0 = simulate_gamma
  n, p = x.shape
  log_mu_hat, neg_log_phi_hat, l1 = mpebpm.sgd.ebpm_gamma(x, s, batch_size=n, max_epochs=2000)
  assert np.isfinite(log_mu_hat).all()
  assert np.isfinite(neg_log_phi_hat).all()
  assert log_mu_hat.shape == (1, p)
  assert neg_log_phi_hat.shape == (1, p)
  assert l1 < l0

def test_ebpm_gamma_minibatch(simulate_gamma):
  x, s, log_mu, log_phi, l0 = simulate_gamma
  n, p = x.shape
  log_mu_hat, neg_log_phi_hat, _ = mpebpm.sgd.ebpm_gamma(x, s, batch_size=100, max_epochs=500, lr=1e-2)
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
  log_mu_hat, neg_log_phi_hat, _ = mpebpm.sgd.ebpm_gamma(x, s, batch_size=1, max_epochs=10, lr=5e-3)
  l1 = -st.nbinom(n=np.exp(neg_log_phi_hat), p=1 / (1 + s * np.exp(log_mu_hat - neg_log_phi_hat))).logpmf(x).sum()
  assert np.isfinite(log_mu_hat).all()
  assert np.isfinite(neg_log_phi_hat).all()
  assert log_mu_hat.shape == (1, p)
  assert neg_log_phi_hat.shape == (1, p)
  assert l1 < l0

def test_ebpm_gamma_trace(simulate_gamma):
  x, s, log_mu, log_phi, l0 = simulate_gamma
  n, p = x.shape
  max_epochs = 5
  log_mu_hat, neg_log_phi_hat, l1, trace = mpebpm.sgd.ebpm_gamma(x[:,0].reshape(-1, 1), s, batch_size=n, max_epochs=max_epochs, trace=True)
  assert len(trace) == max_epochs

def test_ebpm_point_gamma_oracle_init(simulate_point_gamma):
  x, s, log_mu, log_phi, logodds, l0 = simulate_point_gamma
  n, p = x.shape
  log_mu_hat, neg_log_phi_hat, logodds_hat, l1 = mpebpm.sgd.ebpm_point_gamma(x, s, init=(log_mu, -log_phi), batch_size=n, max_epochs=2000)
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
  log_mu_hat, neg_log_phi_hat, logodds_hat, _ = mpebpm.sgd.ebpm_point_gamma(x, s, batch_size=n, max_epochs=2000)
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
  log_mu_hat, neg_log_phi_hat, logodds_hat, l1 = mpebpm.sgd.ebpm_point_gamma(x, s, batch_size=100, max_epochs=500)
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
  log_mu_hat, neg_log_phi_hat, logodds_hat, _ = mpebpm.sgd.ebpm_point_gamma(y, s, batch_size=100, max_epochs=500)
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
  log_mu_hat, neg_log_phi_hat, logodds_hat, l1 = mpebpm.sgd.ebpm_point_gamma(x, s, batch_size=1, max_epochs=30, lr=5e-3)
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

def test_ebpm_point_gamma_trace(simulate_point_gamma):
  x, s, log_mu, log_phi, logodds, l0 = simulate_point_gamma
  n, p = x.shape
  max_epochs = 2000
  log_mu_hat, neg_log_phi_hat, logodds_hat, l1, trace = mpebpm.sgd.ebpm_point_gamma(x[:,0].reshape(-1, 1), s, batch_size=n, max_epochs=max_epochs, trace=True)
  assert len(trace) == max_epochs
