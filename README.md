# S^2G-MCMC
## Code for Stochastic Gradient MCMC with Stale Gradients

The system is adapted from: https://www.cs.rochester.edu/u/yli/caffe-ps.html, which is based on Caffe and MPICH. The system is used in the NIPS 2016 paper "Stochastic Gradient MCMC with Stale Gradients".

Changyou Chen (cchangyou@gmail.com), 2.24.2016

## License

Please note that this code should be used at your own risk. There is no implied guarantee that it will not do anything stupid. Permission is granted to use and modify the code.

## Usage

To use the system, you have to install the right version of third-party libraries such as MPICH.

I added some SG-MCMC solvers such as SGHMC and PSGLD. I used an unstandard (somewhat stupid) implementation for these solvers, such that to use them, you need to define names of layers ended with **BDK** in the model definition file. For an example, please refer to examples/mnist/lenet_train_test_paral_BDK.prototxt.

The three datasets and running scripts used in the paper are given in the **example** folder.

The system only works in CPU mode.

## Citing S^2G-MCMC

Please cite our NIPS paper if it helps your research:

@inproceedings{S2GMCMC_NIPSS2016,
  Author = {C. Chen, N. Ding, C. Li, Y. Zhang, and L. Carin},
  Title = {Stochastic Gradient MCMC with Stale Gradients},
  booktitle={NIPS},
  Year  = {2016}
}

