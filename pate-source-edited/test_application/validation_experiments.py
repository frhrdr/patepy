import numpy as np
from pate_accountant import PATEPyTorch

# load vote files provided by the paper and use the stated hyperparameters.
# repeatedly let the conident gnmax algorithm answer random queries from the dataset up to the limit given in the paper.
# If the implementation is correct, the final data-dependent eps should match the values given in the paper.


def dataset_validation(patepy, dataset_name, custom_beta, n_to_answer, verbose=True):

  votes = np.load("pate2018_test_data/{}.npy".format(dataset_name))
  n_votes = votes.shape[0]
  votes = votes[np.random.permutation(n_votes)]
  assert np.sum(votes[0, :]) == patepy.n_teachers

  n_answered = 0
  n_unchanged = 0
  for idx in range(n_votes):

    votes_i = votes[idx, :]
    ret_val = patepy.gn_max(votes_i, preds=None)

    if ret_val is not None:
      n_answered += 1
      if ret_val == np.argmax(votes_i):
        n_unchanged += 1
      if n_answered >= n_to_answer:
        if verbose:
          print('got {} queries. answered {}. {} answers transmitted correctly'.format(idx+1, n_answered, n_unchanged))
        break

  data_dependent_rdp_eps, order = patepy._data_dependent_dp(rdp=True)
  data_dependent_ddp_eps, delta = patepy._data_dependent_dp(rdp=False)
  results = patepy.release_epsilon_fixed_order(custom_beta, analysis=True, verbose=verbose)
  rdp_sample, rdp_mean, rdp_sdev, gnss_rdp, ddp_sample, ddp_mean = results

  if verbose:
    print('data-depentent-analysis: ({}, {})-RDP & ({}, {})-delta-DP'.format(data_dependent_rdp_eps[0, 0], order,
                                                                             data_dependent_ddp_eps, delta))
    print('GNSS mechanism alone: ({}, {})-RDP'.format(gnss_rdp, order))
    print('GNSS distribution: mean={}, sdev={}, random draw={}'.format(rdp_mean[0, 0], rdp_sdev, rdp_sample[0, 0]))
    print('total epsilon in eps-delta-DP mean={}, random draw={}'.format(ddp_mean[0, 0], ddp_sample[0, 0]))

  return ddp_mean, ddp_sample, data_dependent_ddp_eps


def mnist_validation():
  patepy = PATEPyTorch(target_delta=1e-5,
                       sigma_votes=40.,
                       n_teachers=250,
                       sigma_eps_release=6.23,
                       threshold_mode='confident',
                       threshold_t=200.,
                       threshold_gamma=None,
                       sigma_thresh=150.,
                       order_specs=[14.])
  return dataset_validation(patepy, 'mnist_250_teachers', custom_beta=0.0329, n_to_answer=286, verbose=True)


def svhn_validation():
  patepy = PATEPyTorch(target_delta=1e-6,
                       sigma_votes=40.,
                       n_teachers=250,
                       sigma_eps_release=4.88,
                       threshold_mode='confident',
                       threshold_t=300.,
                       threshold_gamma=None,
                       sigma_thresh=200.,
                       order_specs=[7.5])
  return dataset_validation(patepy, 'svhn_250_teachers', custom_beta=0.0533, n_to_answer=3098, verbose=True)


def glyph_validation(n_to_answer=10762):
  if n_to_answer > 1000:
    print('caution! this will take a while...')
  patepy = PATEPyTorch(target_delta=1e-8,
                       sigma_votes=100.,
                       n_teachers=5000,
                       sigma_eps_release=11.9,
                       threshold_mode='confident',
                       threshold_t=1000.,
                       threshold_gamma=None,
                       sigma_thresh=500.,
                       order_specs=[20.5])
  return dataset_validation(patepy, 'glyph_5000_teachers', custom_beta=0.0205, n_to_answer=n_to_answer, verbose=True)


if __name__ == '__main__':
  mnist_validation()
  # svhn_validation()
  # glyph_validation()
