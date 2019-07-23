# Copyright 2017 The 'Scalable Private Learning with PATE' Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Core functions for RDP analysis in PATE framework.

This library comprises the core functions for doing differentially private
analysis of the PATE architecture and its various Noisy Max and other
mechanisms.
"""

import math
import numpy as np
import scipy.stats


def _logaddexp(x):
  """Addition in the log space. Analogue of numpy.logaddexp for a list."""
  m = max(x)
  return m + math.log(sum(np.exp(x - m)))


def _log1mexp(x):
  """Numerically stable computation of log(1-exp(x))."""
  if x < -1:
    return math.log1p(-math.exp(x))
  elif x < 0:
    return math.log(-math.expm1(x))
  elif x == 0:
    return -np.inf
  else:
    raise ValueError("Argument must be non-positive.")


def compute_eps_from_delta(orders, rdp, delta):
  """Translates between RDP and (eps, delta)-DP.
  Args:
    orders: A list (or a scalar) of orders.
    rdp: A list of RDP guarantees (of the same length as orders).
    delta: Target delta.
  Returns:
    Pair of (eps, optimal_order).
  Raises:
    ValueError: If input is malformed.
  """
  if len(orders) != len(rdp):
    raise ValueError("Input lists must have the same length.")
  eps = np.array(rdp) - math.log(delta) / (np.array(orders) - 1)
  idx_opt = np.argmin(eps)
  return eps[idx_opt], orders[idx_opt]


#####################
# RDP FOR THE GNMAX #
#####################


def compute_logq_gaussian(counts, sigma):
  """Returns an upper bound on ln Pr[outcome != argmax] for GNMax. Implementation of Proposition 7.
  Args:
    counts: A numpy array of scores.
    sigma: The standard deviation of the Gaussian noise in the GNMax mechanism.
  Returns:
    logq: Natural log of the probability that outcome is different from argmax.
  """
  n = len(counts)
  variance = sigma**2
  idx_max = np.argmax(counts)
  counts_normalized = counts[idx_max] - counts
  counts_rest = counts_normalized[np.arange(n) != idx_max]  # exclude one index
  # Upper bound q via a union bound rather than a more precise calculation.
  logq = _logaddexp(
      scipy.stats.norm.logsf(counts_rest, scale=math.sqrt(2 * variance)))  # sqrt(2) in scale * sqrt(2) in logsf = 2

  return min(logq, math.log(1 - (1 / n)))  # (n-1)/n is upper bound when distribution is uniform


def rdp_data_independent_gaussian(sigma, orders):
  """Computes a data-independent RDP curve for GNMax. Implementation of Proposition 8.
  Args:
    sigma: Standard deviation of Gaussian noise.
    orders: An array_like list of Renyi orders.
  Returns:
    Upper bound on RPD for all orders. A scalar if orders is a scalar.
  Raises:
    ValueError: If the input is malformed.
  """
  if sigma < 0 or np.any(orders <= 1):  # not defined for alpha=1
    raise ValueError("Inputs are malformed.")

  variance = sigma**2
  if np.isscalar(orders):
    return orders / variance
  else:
    return np.atleast_1d(orders) / variance


def rdp_gaussian(logq, sigma, orders):
  """Bounds RDP from above of GNMax given an upper bound on q (Theorem 6).
  Args:
    logq: Natural logarithm of the probability of a non-argmax outcome.
    sigma: Standard deviation of Gaussian noise.
    orders: An array_like list of Renyi orders.
  Returns:
    Upper bound on RPD for all orders. A scalar if orders is a scalar.
  Raises:
    ValueError: If the input is malformed.
  """
  if logq > 0 or sigma < 0 or np.any(orders <= 1):  # not defined for alpha=1
    raise ValueError("Inputs are malformed.")

  if np.isneginf(logq):  # If the mechanism's output is fixed, it has 0-DP.
    if np.isscalar(orders):
      return 0.
    else:
      return np.full_like(orders, 0., dtype=np.float)

  variance = sigma**2

  # Use two different higher orders: mu_hi1 and mu_hi2 computed according to
  # Proposition 10.
  mu_hi2 = math.sqrt(variance * -logq)
  mu_hi1 = mu_hi2 + 1

  orders_vec = np.atleast_1d(orders)

  ret = orders_vec / variance  # baseline: data-independent bound

  # Filter out entries where data-dependent bound does not apply.
  mask = np.logical_and(mu_hi1 > orders_vec, mu_hi2 > 1)

  rdp_hi1 = mu_hi1 / variance
  rdp_hi2 = mu_hi2 / variance

  log_a2 = (mu_hi2 - 1) * rdp_hi2

  # Make sure q is in the increasing wrt q range and A is positive.
  comp_val = log_a2 - mu_hi2 * (math.log(1 + 1 / (mu_hi1 - 1)) + math.log(1 + 1 / (mu_hi2 - 1)))
  if np.any(mask) and logq <= comp_val and -logq > rdp_hi2:

    # Use log1p(x) = log(1 + x) to avoid catastrophic cancellations when x ~ 0.
    log1q = _log1mexp(logq)  # log1q = log(1-q)
    log_a = (orders - 1) * (log1q - _log1mexp((logq + rdp_hi2) * (1 - 1 / mu_hi2)))
    log_b = (orders - 1) * (rdp_hi1 - logq / (mu_hi1 - 1))

    # Use logaddexp(x, y) = log(e^x + e^y) to avoid overflow for large x, y.
    log_s = np.logaddexp(log1q + log_a, logq + log_b)
    ret[mask] = np.minimum(ret, log_s / (orders - 1))[mask]

  assert np.all(ret >= 0)

  if np.isscalar(orders):
    return ret.item()
  else:
    return ret


###################################
# RDP FOR THE THRESHOLD MECHANISM #
###################################


def compute_logpr_answered(t, sigma, counts):
  """Computes log of the probability that a noisy threshold is crossed.
  Args:
    t: The threshold.
    sigma: The stdev of the Gaussian noise added to the threshold.
    counts: An array of votes.
  Returns:
    Natural log of the probability that max is larger than a noisy threshold.
  """
  # Compared to the paper, max(counts) is rounded to the nearest integer. This
  # is done to facilitate computation of smooth sensitivity for the case of
  # the interactive mechanism, where votes are not necessarily integer.
  return scipy.stats.norm.logsf(t - round(max(counts)), scale=sigma)


def compute_rdp_data_independent_threshold(sigma, orders):
  # The input to the threshold mechanism has stability 1, compared to
  # GNMax, which has stability = 2. Hence the sqrt(2) factor below.
  return rdp_data_independent_gaussian(2**.5 * sigma, orders)


def compute_rdp_threshold(log_pr_answered, sigma, orders):
  logq = min(log_pr_answered, _log1mexp(log_pr_answered))
  # The input to the threshold mechanism has stability 1, compared to
  # GNMax, which has stability = 2. Hence the sqrt(2) factor below.
  return rdp_gaussian(logq, 2**.5 * sigma, orders)
