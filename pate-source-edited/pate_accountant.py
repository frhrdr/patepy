import numpy as np
import torch as pt
from pate_core import compute_logq_gaussian, rdp_data_independent_gaussian, rdp_gaussian, _log1mexp, \
  compute_eps_from_delta, compute_logpr_answered
from smooth_sensitivity import compute_local_sensitivity_bounds_gnmax, compute_local_sensitivity_bounds_threshold, \
  compute_rdp_of_smooth_sensitivity_gaussian, compute_discounted_max

# DEFINE ALIASES


def get_log_q(counts, sigma):
  return compute_logq_gaussian(counts, sigma)


def rdp_max_vote(sigma, orders, log_q=None):
  if log_q is None:
    return rdp_data_independent_gaussian(sigma, orders)
  else:
    return rdp_gaussian(log_q, sigma, orders)


def rdp_threshold(sigma, orders, log_pr_answered=None):
  logq = min(log_pr_answered, _log1mexp(log_pr_answered)) if log_pr_answered is not None else None
  return rdp_max_vote(2**.5 * sigma, orders, logq)


def rdp_to_dp(orders, rdp, delta):
  return compute_eps_from_delta(orders, rdp, delta)


def local_sensitivity(counts, num_teachers, sigma, order, thresh=None):
  if thresh is not None:
    return compute_local_sensitivity_bounds_threshold(counts, num_teachers, thresh, sigma, order)
  else:
    return compute_local_sensitivity_bounds_gnmax(counts, num_teachers, sigma, order)


def local_to_smooth_sens(beta, ls_by_dist):
  return compute_discounted_max(beta, ls_by_dist)


def rdp_eps_release(beta, sigma, order):
  return compute_rdp_of_smooth_sensitivity_gaussian(beta, sigma, order)


# DEFINE CLASS TO TRACK PRIVACY LOSS AND EMPLOYS GNMAX GN-conf and CN-interactive training

class PATEPyTorch:
  """
  This implementation is per sample. Minibatched version coming later...
  class is meant to guide training of an arbitrary student model in the PATE framework. Its tasks are:
  privacy accounting is done in numpy, for vote perturbation, pytorch is assumed
  - track privacy loss during training
  - determine, which teacher responses are used for training following GNMax, GN-conf, GN-int or a mix of the latter two
  - terminate training when a predetermined privacy budget is exhausted
  - log  training details
  """

  def __init__(self, target_delta, sigma_vote, n_teachers, sigma_eps_release,
               threshold_mode=None, threshold_t=None, threshold_gamma=None, sigma_thresh=None,
               short_list_orders=True):
    super(PATEPyTorch, self).__init__()

    self.target_delta = target_delta
    self.budget_spent = 0.
    self.sigma_vote = sigma_vote
    self.sigma_thresh = sigma_thresh
    self.sigma_eps_release = sigma_eps_release
    self.threshold_mode = None
    self.set_threshold_mode(threshold_mode)
    self.threshold_T = threshold_t
    self.threshold_gamma = threshold_gamma
    self.orders = self.get_orders(short_list_orders)
    self.rdp_eps_by_order = np.zeros(len(self.orders))
    self.vote_log = []
    self.n_teachers = n_teachers
    self.selected_order = None
    self.data_dependent_eps = None
    # self.n_labels = None

  def set_threshold_mode(self, threshold_mode):
    assert threshold_mode in (None, 'conf', 'int')
    self.threshold_mode = threshold_mode

  @staticmethod
  def get_orders(short_list):
    if short_list:
      return np.round(np.concatenate((np.arange(2, 50 + 1, 1), np.logspace(np.log10(50), np.log10(1000), num=20))))
    else:
      return np.concatenate((np.arange(2, 100 + 1, .5), np.logspace(np.log10(100), np.log10(500), num=100)))

  def add_rdp_loss(self, counts, sigma, thresh=False):
    """
    calculates privacy loss from counts and the sigma that was used to perturb them
    :param counts:
    :param sigma:
    :param thresh:
    :return:
    """
    if thresh:
      log_pr_answered = compute_logpr_answered(self.threshold_T, sigma, counts)
      rdp_thresh = rdp_threshold(sigma, self.orders, log_pr_answered)
      self.rdp_eps_by_order += rdp_thresh
    else:
      log_q = get_log_q(counts, sigma)
      rdp_query = rdp_max_vote(sigma, self.orders, log_q)
      self.rdp_eps_by_order += rdp_query

  def rdp_max_vote(self, counts, preds):
    """
    depending on threshold_mode, return index of noisy teacher consensus, confident student label or None
    :param counts: teacher votes
    :param preds: student predictions (vector of class probabilities)
    :return:
    """
    release_votes = False
    data_intependent_ret = None

    if self.threshold_mode is None:
      release_votes = True

    elif self.threshold_mode is 'conf':
      self.add_rdp_loss(counts, self.sigma_thresh, thresh=True)
      if pt.max(counts) + pt.normal(0., self.sigma_thresh) < self.threshold_T:
        release_votes = True

    elif self.threshold_mode is 'int':
      self.add_rdp_loss(counts, self.sigma_thresh, thresh=True)  # TODO verify this is correct. very likely different.
      if pt.max(counts - self.n_teachers * preds) + pt.normal(0., self.sigma_thresh) >= self.threshold_T:
        release_votes = True
      elif pt.max(preds) > self.threshold_gamma:
        # self.check_budget()
        data_intependent_ret = pt.argmax(preds)

    self.vote_log.append((counts, release_votes))

    if release_votes:
      self.add_rdp_loss(counts, self.sigma_vote, thresh=False)
      # self.check_budget()
      return pt.argmax(counts + pt.normal(pt.ones_like(counts), self.sigma_thresh))
    else:
      return data_intependent_ret

  def data_dependent_delta_dp(self):
    if self.data_dependent_eps is None:
      eps, order = rdp_to_dp(self.orders, self.rdp_eps_by_order, self.target_delta)
      self.selected_order = order
      self.data_dependent_eps = eps
    return eps

  def check_budget(self):
    """
    since the data-dependent epsilon should probably not be used in the algorithm
    (as then the algorithm contains a secret), no stopping condition based on privacy budget is possible.
    """
    # eps, order = rdp_to_dp(self.orders, self.rdp_eps_by_order, self.budget_delta)
    # if eps > self.budget_eps:
    #   raise ValueError
    pass

  # def rdp_max_vote_batch(self, counts, preds):
  #   """
  #       depending on threshold_mode, return index of noisy teacher consensus, confident student label or None
  #       this one assumes batches of counts and preds
  #       :param counts: teacher votes (bs, n_labels)
  #       :param preds: student predictions (bs, n_labels)
  #       :return:
  #       """
  #   if self.threshold_mode is 'conf':
  #     self.add_rdp_loss(counts, self.sigma_thresh)
  #     if pt.max(counts) + pt.normal(0., self.sigma_thresh) >= self.threshold_T:
  #       return None
  #
  #   elif self.threshold_mode is 'int':
  #     self.add_rdp_loss(counts, self.sigma_thresh)  # TODO verify this is correct. very likely different.
  #     if pt.max(counts - self.n_teachers * preds) + pt.normal(0., self.sigma_thresh) >= self.threshold_T:
  #       pass
  #     elif pt.max(preds) > self.threshold_gamma:
  #       # self.check_budget()
  #       return pt.argmax(preds)
  #     else:
  #       return None
  #
  #   self.add_rdp_loss(counts, self.sigma_vote)
  #   # self.check_budget()
  #   return pt.argmax(counts + pt.normal(pt.ones_like(counts), self.sigma_thresh))

  def choose_beta(self):
    """
    this hyperparameter may warrant some search. for now, we just use 0.4/order as recommended in the paper
    :return: a single beta (may be extended to a list of canididate betas later, with some other changes)
    """
    return 0.4/self.selected_order

  def release_epsilon_fixed_order(self):
    """
    goes through the vote_log and computes the smooth sensitivity of the data-dependent epsilon
    depends on the chosen threshold mode.
    As in the Papernot script (smooth_sensitivity_table.py), the best order from the data-dependent RDP cycle is used.
    searching over orders in both this and the data-dependent privacy analysis is very costly,
    but may be implemented in a separate function later on.
    :return: parameters of the private epsilon distribution along with a sinlge draw for release.
    """
    order = self.selected_order
    data_dependent_eps = self.data_dependent_delta_dp()
    ls_by_dist_acc = np.zeros(self.n_teachers)

    for idx, (vote, released) in enumerate(self.vote_log):
      if self.threshold_mode is not None:
        # add threshold cost
        ls_by_dist = local_sensitivity(vote, self.n_teachers, self.sigma_thresh, order, self.threshold_T)
        ls_by_dist_acc += ls_by_dist

      if released:
        # add release cost
        ls_by_dist = local_sensitivity(vote, self.n_teachers, self.sigma_vote, order)
        ls_by_dist_acc += ls_by_dist

      # compute smooth sensitivity by distounting each
      beta = self.choose_beta()

      smooth_s = local_to_smooth_sens(beta, ls_by_dist_acc)
      eps_release_rdp = rdp_eps_release(beta, self.sigma_eps_release, order)

      release_mean = data_dependent_eps + eps_release_rdp
      release_sdev = smooth_s * self.sigma_eps_release

      release_sample = np.random.normal(release_mean, release_sdev)

      return release_sample, release_mean, release_sdev
