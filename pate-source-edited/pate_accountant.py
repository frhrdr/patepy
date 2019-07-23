import numpy as np
from aliases import get_log_q, rdp_max_vote, rdp_threshold, rdp_to_dp, \
  local_sensitivity, local_to_smooth_sens, rdp_eps_release


class PATEPyTorch:
  """
  This implementation is per sample. Minibatched version coming later...
  class is meant to guide training of an arbitrary student model in the PATE framework. Its tasks are:
  currently, everything is based on numpy
  - track privacy loss during training
  - determine, which teacher responses are used for training following GNMax, GN-conf, GN-int or a mix of the latter two
  (- terminate training when a predetermined privacy budget is exhausted) useful budget definition is hard to define
  - log  training details
  """

  def __init__(self, target_delta, sigma_votes, n_teachers, sigma_eps_release,
               threshold_mode='basic', threshold_t=None, threshold_gamma=None, sigma_thresh=None,
               order_specs=None):
    super(PATEPyTorch, self).__init__()

    self.n_teachers = n_teachers
    self.orders = self._get_orders(order_specs)
    self.target_delta = target_delta
    self.sigma_votes = sigma_votes
    self.sigma_thresh = sigma_thresh
    self.sigma_eps_release = sigma_eps_release

    self.threshold_mode = None
    self.basic_mode = 0
    self.confident_mode = 1
    self.interactive_mode = 2
    self.threshold_mode = self._get_threshold_mode(threshold_mode)

    self.threshold_T = threshold_t
    self.threshold_gamma = threshold_gamma
    self.rdp_eps_by_order = np.zeros(len(self.orders))  # stores the data-dependent privacy loss
    self.votes_log = []  # stores votes and whether the threshold was passsed for smooth sensitivity analysis
    self.selected_order = None
    self.data_dependent_ddp_eps = None
    self.data_dependent_rdp_eps = None

  @staticmethod
  def _get_orders(specs):
    if specs in ('short', 'shortlist', None):
      return np.round(np.concatenate((np.arange(2, 50 + 1, 1), np.logspace(np.log10(50), np.log10(1000), num=20))))
    elif specs in ('long', 'longlist'):
      return np.concatenate((np.arange(2, 100 + 1, .5), np.logspace(np.log10(100), np.log10(500), num=100)))
    elif isinstance(specs, list):
      return np.asarray(specs)
    else:
      raise ValueError

  def _get_threshold_mode(self, threshold_mode):
    modes = {'basic': self.basic_mode, 'confident': self.confident_mode, 'interactive': self.interactive_mode}
    assert threshold_mode in modes
    return modes[threshold_mode]

  def _add_rdp_loss(self, votes, sigma, thresh=False):
    """
    calculates privacy loss from votes and the sigma that was used to perturb them
    :param votes:
    :param sigma:
    :param thresh:
    :return:
    """
    if thresh:
      self.rdp_eps_by_order += rdp_threshold(sigma, self.orders, self.threshold_T, votes)
    else:
      self.rdp_eps_by_order += rdp_max_vote(sigma, self.orders, get_log_q(votes, sigma))

  def gn_max(self, votes, preds):
    """
    depending on threshold_mode, return index of noisy teacher consensus, confident student label or None
    :param votes: teacher votes
    :param preds: student predictions (vector of class probabilities)
    :return:
    """
    release_votes = False
    data_intependent_ret = None
    thresh_votes = votes

    # thresholding based on mode
    if self.threshold_mode == self.basic_mode:
      release_votes = True

    elif self.threshold_mode == self.confident_mode:
      self._add_rdp_loss(votes, self.sigma_thresh, thresh=True)
      if np.max(votes) + np.random.normal(0., self.sigma_thresh) >= self.threshold_T:
        release_votes = True

    elif self.threshold_mode == self.interactive_mode:
      thresh_votes = votes - self.n_teachers * preds
      self._add_rdp_loss(thresh_votes, self.sigma_thresh, thresh=True)
      if np.max(thresh_votes) + np.random.normal(0., self.sigma_thresh) >= self.threshold_T:
        release_votes = True
      elif np.max(preds) > self.threshold_gamma:
        data_intependent_ret = np.argmax(preds)

    self.votes_log.append((votes, thresh_votes, release_votes))

    # release max vote if threshold is passed
    if release_votes:
      self._add_rdp_loss(votes, self.sigma_votes, thresh=False)
      return np.argmax(votes + np.random.normal(np.ones_like(votes), self.sigma_votes))
    else:
      return data_intependent_ret

  def gn_max_batched(self, votes, preds):
    """
    cuts batch of votes and preds into samples and computes rdp_max_vote individually, then aggregates the results.
    Filters out none values
    more efficient version requires deeper changes and will be added later.
    :param votes: tensor of teacher votes (bs, n_labels)
    :param preds: student predictions (bs, n_labels)
    :return:
    """
    bs_range = list(range(votes.size()[0]))
    ret_vals = [self.gn_max(votes[idx], preds[idx]) for idx in bs_range]
    indices = [idx for idx in bs_range if ret_vals[idx] is not None]
    released_vals = ret_vals[indices]
    return released_vals, indices

  def _data_dependent_dp(self, rdp=True):
    """
    computes and saves the data-dependent epsilon
    :return:
    """
    if self.data_dependent_ddp_eps is None:
      eps, order = rdp_to_dp(self.orders, self.rdp_eps_by_order, self.target_delta)
      self.selected_order = order
      self.data_dependent_ddp_eps = eps
      self.data_dependent_rdp_eps = self.rdp_eps_by_order[np.argwhere(self.orders == order)]

    if rdp:
      return self.data_dependent_rdp_eps, self.selected_order
    else:
      return self.data_dependent_ddp_eps, self.target_delta

  def release_epsilon_fixed_order(self, custom_beta=None, analysis=False, verbose=False):
    """
    goes through the votes_log and computes the smooth sensitivity of the data-dependent epsilon
    depends on the chosen threshold mode.
    As in the Papernot script (smooth_sensitivity_table.py), the best order from the data-dependent RDP cycle is used.
    searching over orders in both this and the data-dependent privacy analysis is very costly,
    but may be implemented in a separate function later on.
    :return: parameters of the private epsilon distribution along with a sinlge draw for release.
    """
    data_dependent_rdp_eps, order = self._data_dependent_dp(rdp=True)

    ls_by_dist_acc = np.zeros(self.n_teachers)

    if verbose:
      print('accounting ls of votes:')

    for idx, (votes, thresh_votes, released) in enumerate(self.votes_log):
      if self.threshold_mode is not self.basic_mode:
        ls_by_dist_acc += local_sensitivity(thresh_votes, self.n_teachers, self.sigma_thresh, order, self.threshold_T)

      if released:
        ls_by_dist_acc += local_sensitivity(votes, self.n_teachers, self.sigma_votes, order)

      if verbose and idx % (len(self.votes_log) // 10) == 0 and idx > 0:
        print('{}%'.format(idx/len(self.votes_log)))

    beta = 0.4 / self.selected_order if custom_beta is None else custom_beta  # default recommended in the paper
    smooth_s = local_to_smooth_sens(beta, ls_by_dist_acc)
    eps_release_rdp = rdp_eps_release(beta, self.sigma_eps_release, order)

    release_rdp_mean = data_dependent_rdp_eps + eps_release_rdp
    release_rdp_sdev = smooth_s * self.sigma_eps_release
    release_rdp_sample = np.random.normal(release_rdp_mean, release_rdp_sdev)

    delta_dp_sample, _ = rdp_to_dp([self.selected_order], [release_rdp_sample], self.target_delta)
    delta_dp_mean, _ = rdp_to_dp([self.selected_order], [release_rdp_mean], self.target_delta)

    if not analysis:
      return release_rdp_sample, delta_dp_sample
    else:

      return release_rdp_sample, release_rdp_mean, release_rdp_sdev, eps_release_rdp, delta_dp_sample, delta_dp_mean
