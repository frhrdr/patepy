from pate_core import compute_logq_gaussian, rdp_data_independent_gaussian, rdp_gaussian, _log1mexp, \
  compute_eps_from_delta, compute_logpr_answered
from smooth_sensitivity import compute_local_sensitivity_bounds_gnmax, compute_local_sensitivity_bounds_threshold, \
  compute_rdp_of_smooth_sensitivity_gaussian, compute_discounted_max


# define aliases of functions from pate_core and smooth_sensitivity in order to simplify things a little bit.

def get_log_q(votes, sigma):
  """

  :param votes:
  :param sigma:
  :return:
  """
  return compute_logq_gaussian(votes, sigma)


def rdp_max_vote(sigma, orders, log_q=None):
  """

  :param sigma:
  :param orders:
  :param log_q:
  :return:
  """
  if log_q is None:
    return rdp_data_independent_gaussian(sigma, orders)
  else:
    return rdp_gaussian(log_q, sigma, orders)


def rdp_threshold(sigma, orders, threshold, votes):
  """

  :param sigma:
  :param orders:
  :param threshold:
  :param votes:
  :return:
  """
  log_pr_answered = compute_logpr_answered(threshold, sigma, votes)
  logq = min(log_pr_answered, _log1mexp(log_pr_answered)) if log_pr_answered is not None else None
  return rdp_max_vote(2**.5 * sigma, orders, logq)  # sqrt(2) due to lower sensitivity


def rdp_to_dp(orders, rdp, delta):
  """

  :param orders:
  :param rdp:
  :param delta:
  :return:
  """
  return compute_eps_from_delta(orders, rdp, delta)


def local_sensitivity(votes, num_teachers, sigma, order, thresh=None):
  """

  :param votes:
  :param num_teachers:
  :param sigma:
  :param order:
  :param thresh:
  :return:
  """
  if thresh is not None:
    return compute_local_sensitivity_bounds_threshold(votes, num_teachers, thresh, sigma, order)
  else:
    return compute_local_sensitivity_bounds_gnmax(votes, num_teachers, sigma, order)


def local_to_smooth_sens(beta, ls_by_dist):
  """

  :param beta:
  :param ls_by_dist:
  :return:
  """
  return compute_discounted_max(beta, ls_by_dist)


def rdp_eps_release(beta, sigma, order):
  """

  :param beta:
  :param sigma:
  :param order:
  :return:
  """
  return compute_rdp_of_smooth_sensitivity_gaussian(beta, sigma, order)
