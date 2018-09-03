## Tensorflow CurveBall

This code is an unofficial tensorflow reproduction of the optimizer implemented in the paper:

João F. Henriques, Sebastien Ehrhardt, Samuel Albanie, Andrea Vedaldi
**["Small steps and giant leaps: Minimal Newton solvers for Deep Learning"](https://arxiv.org/abs/1805.08095)**
arXiv preprint, 2018, with official [repository](https://github.com/jotaf98/curveball).

### Requirement

Tensorflow version > 1.3


### Training your own network

The main optimizer is implemented in `CurveBall.py`.

To train your own network using this code change directory to `cifar10/`. This folder is inspired of Tensorflow basic cifar10 [example](https://github.com/tensorflow/models/tree/master/tutorials/image/cifar10). To run the most basic example call `python cifar10_train.py --optimizer 'CurveBall' --lr 1`. In addition you can use ResNet-18 example from the original paper by running `python cifar10_train.py --optimizer 'CurveBall' --lr 1 --network 'ResNet'`. Note that GPU is enabled by default in this case and unlike the basic network activations and weight distribution are not recorded.

Results of running the two aforementioned commands can be found in `docs/` Curveball curves are in red while the blue curves are standard Adam trained with learning rate of 0.001.

### Misc

The forward mode autodiff is based on this [issue](https://github.com/renmengye/tensorflow-forward-ad/issues/2).

All the job is done in the `compute_gradients` method. Mometum and variables are all updated in `apply_gradient` follwing the momentum update largely taken from Tesnorflow's `MomentumOptimizer`. From this code it should be straightforward to implement your own `minimize` function.

In addition, `tests/` contains a test for forward mode autodiff. The solver is able to perform curveball method with true hessian as an option .

### Known error
It seems that for some users 'BatchNormGradGrad' is stuck in an infinite loop. If so it is likely you will get stuck before initializing your session.
