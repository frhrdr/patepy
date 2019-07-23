# patepy


This code provides a wrapper for the released code from the "Scalable Private Learning with PATE" paper by Papernot et al (https://arxiv.org/abs/1802.08908) with the goal of making more accessible and easier to use.



### Dependencies
    python 3.x
    numpy 
    scipy

### Repository structure

- `pate_core.py` and `smooth_sensitivity.py` contain the actual analysis code taken from https://github.com/tensorflow/privacy/tree/master/research
- `aliases.py` imports from above files and renames and comments functions for easier use.
- `pate_accountant.py` provides a class for easy use of the GNMax mechanisms (basic, confident and interactive) with PATE analysis.
    - Function gn_max() is used to release a lable for a given input and compute data-dependent privacy cost.
    - Function release_epsilon_fixed_order() is called after training to choose an order lambda and release the overall privacy cost via GNSS mechanism with smooth senistivity analysis.
- `mnist_cnn.py` contains and trains the reference CNN model used in the paper
- `test_application/validation_experiments.py` uses the vote data provided by the authors to verify that the pate_accountant outputs reasonable results.
    - This file also contains a small running example of how the PATE accountant may be applied to compute __confident GNMax__ vote release and privacy cost.
- 

### Disclaimer

This is a work in progress. It may well contain errors and the functionality and clarity of explanation will hopefully improve in the coming months. 


##### TODOs
After my vacation I will work on the following points

- implement efficient labeling function directly for pytorch
- Provide full working example code including trained models
- improve clarity in naming and documentation of code
- iework the write-up from a loose collection of thoughts into a comprehensive read

If you have any questions, comments or corrections, please don't hesitate to [contact me](https://ei.is.tuebingen.mpg.de/person/fharder).

