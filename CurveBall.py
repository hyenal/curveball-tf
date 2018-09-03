"""Curveball for TensorFlow."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework.ops import Tensor
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.training import optimizer
from tensorflow.python.training import training_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.gradients_impl import _hessian_vector_product

def fmad_prod(ys, xs, d_xs):
  """Forward-mode pushforward analogous to the pullback defined by tf.gradients.
  With tf.gradients, grad_ys is the vector being pulled back, and here d_xs is
  the vector being pushed forward.
  Args:
    ys: A Tensor representing the logits.
    xs: A list of Tensors which contains the
      variable to compute the gradient w.r.to
    d_xs: A list of Tensors the same size as xs.
    This is representing the vector v for which
      we want to compute J^T*v.

  """
  # Sanity check
  for i in range(len(xs)):
    assert xs[i].shape.as_list()==d_xs[i].shape.as_list(), 'xs and d_xs\
      have different shapes'

  v = tf.zeros_like(ys)  # dummy variable
  g = tf.gradients(ys, xs, grad_ys=v)
  return tf.gradients(g, v, grad_ys=d_xs)[0]

class CurveBallOptimizer(optimizer.Optimizer):
  """Optimizer that implements the gradient descent algorithm.
  """

  def __init__(self, pre_loss, learning_rate=1, last_loss=None, loss_name='logistic',
                lambd=1, true_hessian=False, beta = 0.05, momentum = 0.9,
                autoparam=True, autoparam_reg=0,
                autolambda=False, autolambda_step=5, autolambda_w1=0.999, autolambda_thresh=0.5,
                use_locking=False, name="CurveBall"):
    """Construct a new CurveBall optimizer.
    Args:
      pre_loss: A Tensor or list of tensor of variables prior
       the loss computation.
      learning_rate: A Tensor or a floating point value.  The learning
        rate to use.
      loss_name: A python string to indicate which loss is used to
        compute loss hessian.
      lambd: Trust region value.
      true_hessian: Whether to use true hessian or not.
      beta: Learning rate for linear system GD step.
      momentum: Forget factor for linear system solution.
      autoparam: Whether to adapt beta and momentum automatically.
      autoparam_reg: Regularizer for 2x2 autoparam linear system.
      use_locking: If True use locks for update operations.
      name: Optional name prefix for the operations created when applying
        gradients. Defaults to "CurveBall".
    @compatibility(eager)
    Not compatible
    @end_compatibility
    """
    super(CurveBallOptimizer, self).__init__(use_locking, name)
    # Make the learning_rate = 1, apply_gradient will just change the
    # value of the network weights all the work will be done in the
    # compute gradient function
    self._learning_rate  = 1.
    self._lr             = learning_rate
    self._pre_loss       = pre_loss
    self._loss_name      = loss_name
    self._hessian	     = true_hessian
    self._step           = tf.get_variable("%s/_global_step"%(name), shape=[], dtype=tf.int64,\
                                            initializer=tf.zeros_initializer(), trainable=False)
    self._last_params    = last_loss
    self._last_loss      = tf.get_variable("%s/_last_loss"%(name), shape=[],\
                                            initializer=tf.zeros_initializer(), trainable=False)

    self._autoparam      = autoparam
    self._autoparam_reg  = autoparam_reg

    self._lambda         = lambd
    self._beta           = beta
    self._momentum       = momentum

    self._autolambda     = autolambda

    # Sanity check
    if not isinstance(last_loss,Tensor) and autolambda:
      raise RuntimeError('If use autoparam please specify updated loss')
    self._auto_step      = autolambda_step
    self._autolambda_w1  = autolambda_w1
    self._autolambda_thresh = autolambda_thresh
    self._M_val             = tf.get_variable("%s/_M_val"%(name), shape=[],\
                                            initializer=tf.ones_initializer(), trainable=False)

  def compute_gradients(self, loss, var_list=None,
                        aggregation_method=None,
                        colocate_gradients_with_ops=False,
                        device='/cpu:0'):
    """Compute gradients of `loss` for the variables in `var_list`.
    This is the first part of `minimize()`.  It returns a list
    of (gradient, variable) pairs where "gradient" is the gradient
    for "variable".  Note that "gradient" can be a `Tensor`, an
    `IndexedSlices`, or `None` if there is no gradient for the
    given variable.
    Args:
      loss: A Tensor containing the value to minimize or a callable taking
        no arguments which returns the value to minimize. When eager execution
        is enabled it must be a callable.
      var_list: Optional list or tuple of `tf.Variable` to update to minimize
        `loss`.  Defaults to the list of variables collected in the graph
        under the key `GraphKeys.TRAINABLE_VARIABLES`.
      aggregation_method: Specifies the method used to combine gradient terms.
        Valid values are defined in the class `AggregationMethod`.
      colocate_gradients_with_ops: If True, try colocating gradients with
        the corresponding op.
      device: which device to compute the variables dot product on.
    Returns:
      A list of (gradient, variable) pairs. Variable is always present, but
      gradient can be `None`.
    Raises:
      TypeError: If `var_list` contains anything else than `Variable` objects.
      ValueError: If some arguments are invalid.
      NotImplementedError: If called with eager execution enabled. or with
        unknown loss name
    @compatibility(eager)
    Not compatible.
    @end_compatibility
    """
    if callable(loss):
      raise NotImplementedError('Eager execution is not available yet')

    if self._autolambda:
      self._lambda = tf.reshape(tf.cond(tf.equal(tf.mod(self._step, self._auto_step),0),\
          self._autolam, lambda: self._lambda),[])
      self._loss = loss

    # Get trainable variables
    if var_list is None:
      var_list = (variables.trainable_variables() +
          ops.get_collection(ops.GraphKeys.TRAINABLE_RESOURCE_VARIABLES))
    else:
      var_list = nest.flatten(var_list)

    # pylint: disable=protected-access
    var_list += ops.get_collection(ops.GraphKeys._STREAMING_MODEL_PORTS)

    # Check if we have anything to optimize
    if not var_list:
      raise ValueError("No variables to optimize.")

    # TODO enable more variables mode maybe fix z device placement
    var_refs = var_list

    # Init momentum vector
    mu = 1.
    self._z = []
    self._zdic = {}
    for i in range(len(var_refs)):
      self._z.append(tf.get_variable("z%03d"%(i), shape=var_refs[i].get_shape(),
                  caching_device=var_refs[i].device, initializer=tf.zeros_initializer()))
      self._zdic[var_refs[i].name] = self._z[i]
    # do two GD steps. first update z (linear system state), then use the
    # result (whitened gradient estimate) to update the parameters w.
    #
    # zdelta = grad{||(mu * J' * Hl * J + lambda * I) * z - J' * Jl'||^2}
    #        = (mu * J' * Hl * J + lambda * I) * z - J' * Jl'
    #        =  mu * J' * Hl * J * z + lambda * z - J' * Jl'
    #
    # znew = momentum * z - beta * zdelta
    #
    # wnew = w - lr * znew
    #

    # Assert that pre_loss is a single tensorflow tensor for simplicity

    if not isinstance(self._pre_loss,Tensor):
      raise NotImplementedError('Optimizer not yet working with vector of logits')

    delta_z = []
    if not self._hessian:
      Jz  = fmad_prod(self._pre_loss, var_refs, self._z)

      # Evaluate Hessian loss and gradient
      Jz_ = self._hessian_grad_loss(self._loss_name, self._pre_loss, loss, Jz)
      Jl  = tf.gradients(loss, self._pre_loss)[0]
      Jz_ = mu*Jz_

      # Backpropagate Jz_ - Jl
      h_term = tf.gradients(self._pre_loss, var_refs, Jz_+Jl)
      for i in range(len(var_refs)):
        delta_z.append(h_term[i] + self._lambda*self._z[i])
    else:
	  # Compute gradient w.r.t the loss
      grad = tf.gradients(loss, var_refs)

	  # Tensorflow build-in function, compute hessian vector products
      h_term = _hessian_vector_product(loss, var_refs, self._z)
      for i in range(len(var_refs)):
        delta_z.append(h_term[i] + self._lambda*self._z[i] + grad[i])


    # Autoparam
    if self._autoparam:
      if not self._hessian:
      	Jdz = fmad_prod(self._pre_loss, var_refs, delta_z)

      	Jdz_ = self._hessian_grad_loss(self._loss_name, self._pre_loss, loss, Jdz)

      	with tf.device(device):
      	  A11 = mu*tf.matmul(tf.reshape(Jdz,[1,-1]), tf.reshape(Jdz_,[-1,1]))
      	  A12 = mu*tf.matmul(tf.reshape(Jz,[1,-1]), tf.reshape(Jdz_,[-1,1]))
      	  A22 = mu*tf.matmul(tf.reshape(Jz,[1,-1]), tf.reshape(Jz_,[-1,1]))

      	  b1 = tf.matmul(tf.reshape(Jl,[1,-1]), tf.reshape(Jdz,[-1,1]))
      	  b2 = tf.matmul(tf.reshape(Jl,[1,-1]), tf.reshape(Jz,[-1,1]))

      	  for i in range(len(var_refs)):
      	    # compute the system we want to invert
      	    z_vec  = tf.reshape(self._z[i], [1,-1])
      	    dz_vec = tf.reshape(delta_z[i], [1,-1])

      	    A11 = A11 + tf.matmul(dz_vec, dz_vec, transpose_b=True) * self._lambda
      	    A12 = A12 + tf.matmul(dz_vec, z_vec, transpose_b=True) * self._lambda
      	    A22 = A22 + tf.matmul(z_vec, z_vec, transpose_b=True) * self._lambda
      else:
	    # Tensorflow build-in function, compute hessian vector products
        h_term_dz = _hessian_vector_product(loss, var_refs, delta_z)

        with tf.device(device):
      	  A11, A12, A22 = 0, 0, 0
      	  b1, b2 = 0, 0

      	  for i in range(len(var_refs)):
      	    # compute the system we want to invert
      	    z_vec  = tf.reshape(self._z[i], [1,-1])
      	    dz_vec = tf.reshape(delta_z[i], [1,-1])

      	    hz_vec  = tf.reshape(h_term[i], [1,-1])
      	    hdz_vec = tf.reshape(h_term_dz[i], [1,-1])

            A11 = A11 + tf.matmul(hdz_vec, dz_vec, transpose_b=True) + tf.matmul(dz_vec, dz_vec, transpose_b=True) * self._lambda
            A12 = A12 + tf.matmul(hdz_vec, z_vec, transpose_b=True) + tf.matmul(dz_vec, z_vec, transpose_b=True) * self._lambda
            A22 = A22 + tf.matmul(hz_vec, z_vec, transpose_b=True) + tf.matmul(z_vec, z_vec, transpose_b=True) * self._lambda

      	    b1 = b1 + tf.matmul(tf.reshape(grad[i],[1,-1]), tf.reshape(dz_vec,[-1,1]))
      	    b2 = b2 + tf.matmul(tf.reshape(grad[i],[1,-1]), tf.reshape(z_vec,[-1,1]))

      # compute beta and momentum coefficient
      A = tf.concat([tf.concat([A11, A12], 0), tf.concat([A12, A22], 0)],1)
      b = tf.concat([b1, b2], 0)

      # Solve linear system
      m_b     = tf.matrix_solve_ls(A, b, l2_regularizer=self._autoparam_reg, fast=False)
      self._M = - 0.5 * tf.reduce_sum(m_b*b)

      m_b = tf.unstack(m_b, axis = 0)

      beta           = -tf.to_float(m_b[0])
      self._momentum = -tf.to_float(m_b[1])
    else:
      beta           = -self._beta

    # Update gradient
    for i in range(len(var_refs)):
      # delta_z handle the momentum update
      delta_z[i] = beta*delta_z[i]


    grads_and_vars = list(zip(delta_z, var_list))
    self._assert_valid_dtypes(
        [v for g, v in grads_and_vars
         if g is not None and v.dtype != dtypes.resource])
    return grads_and_vars

  def _hessian_grad_loss(self, loss_name, pre_loss, loss, x):
    # computes the loss value, its gradient, and the hessian (multiplied by a
    # vector x).
    batch_size = pre_loss.get_shape().as_list()[0]
    # get value of last var (assumes only one output)
    pred = tf.reshape(pre_loss, [batch_size, -1])  # reshape to 4D tensor for vl_nnloss

    # switch loss
    if loss_name == 'ls':  # least-squares loss.
      # compute Hl * x. Hl = 2 / batch_size * I.
      Hlx = 2 / batch_size * x
      return Hlx

    if loss_name == 'logistic': # logistic loss.
    # compute Hl * x. for a single sample, Hl = diag(p) - p * p', where p
    # is a column-vector. for many samples, p has one column per sample,
    # and Hl is a block-diag with each block as above (so one independent
    # matrix-vec product per sample). first compute p' * x for all samples

      p = tf.nn.softmax(pred);  # softmaxed probabilities
      px = tf.reduce_sum(p * x, 1, keep_dims=True);
      # now finish computing Hl * x = diag(p) * x - p * p' * x
      Hlx = p * x - p * px;
      Hlx = Hlx / batch_size;
      return Hlx
    else:
      raise NotImplementedError('Unknown loss.');

  def _autolam(self):
    ''' Update lambda automatically'''
    # # ratio between true curvature and quadratic fit curvature
    # ratio = (h_new - last_loss) / self._M
    ratio = (self._last_params - self._last_loss)/self._M_val

    # increase or decrease lambda based on ratio
    w1     = self._autolambda_w1;
    thresh = self._autolambda_thresh
    if not type(thresh) is list:
      thresh = [thresh, 2 - thresh]
    assert thresh[1] - thresh[0] >= 0, 'Difference between threshold values < 0'

    lambd = tf.to_float(tf.less(ratio,thresh[0]))*self._lambda / w1 +\
      tf.to_float(tf.greater(ratio,thresh[1]))*self._lambda * w1

    return lambd

  ####################################
  # Apply gradients and few updates
  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    """Apply gradients to variables.
    This is the second part of `minimize()`. It returns an `Operation` that
    applies gradients.
    Args:
      grads_and_vars: List of (gradient, variable) pairs as returned by
        `compute_gradients()`.
      global_step: Optional `Variable` to increment by one after the
        variables have been updated.
      name: Optional name for the returned operation.  Default to the
        name passed to the `Optimizer` constructor.
    Returns:
      An `Operation` that applies the specified gradients. If `global_step`
      was not None, that operation also increments `global_step`.
    Raises:
      TypeError: If `grads_and_vars` is malformed.
      ValueError: If none of the variables have gradients.
      RuntimeError: If you should use `_distributed_apply()` instead.
    """
    apply_gradients_updates = super(CurveBallOptimizer, self).apply_gradients(grads_and_vars, global_step, name)
    if self._autolambda:
      with tf.control_dependencies([self._lambda]):
        assign_last_loss   = tf.assign(self._last_loss, self._loss)
        assign_M_value     = tf.assign(self._M_val, self._M)
        increm_global_step = tf.assign_add(self._step, 1)
      apply_gradients_updates = tf.group(apply_gradients_updates, assign_last_loss, increm_global_step, assign_M_value)
    return apply_gradients_updates


  ####################################
  # Taken from MomentumOptimizer
  def _prepare(self):
    learning_rate = -self._learning_rate
    if callable(learning_rate):
      learning_rate = learning_rate()
    self._learning_rate_tensor = ops.convert_to_tensor(learning_rate,
                                                       name="learning_rate")
    momentum = self._momentum
    if callable(momentum):
      momentum = momentum()
    self._momentum_tensor = ops.convert_to_tensor(momentum, name="momentum")
    self._momentum_tensor = tf.reshape(self._momentum_tensor,[])

  def _apply_dense(self, grad, var):
    mom = self._zdic[var.name]
    return training_ops.apply_momentum(
        var, mom,
        math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype),
        grad,
        math_ops.cast(self._momentum_tensor, var.dtype.base_dtype),
        use_locking=self._use_locking,
        use_nesterov=False).op

  def _resource_apply_dense(self, grad, var):
    mom = self._zdic[var.name]
    return training_ops.resource_apply_momentum(
        var.handle, mom.handle,
        math_ops.cast(self._learning_rate_tensor, grad.dtype.base_dtype),
        grad,
        math_ops.cast(self._momentum_tensor, grad.dtype.base_dtype),
        use_locking=self._use_locking,
        use_nesterov=False)

  def _apply_sparse(self, grad, var):
    mom = self._zdic[var.name]
    return training_ops.sparse_apply_momentum(
        var, mom,
        math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype),
        grad.values, grad.indices,
        math_ops.cast(self._momentum_tensor, var.dtype.base_dtype),
        use_locking=self._use_locking,
        use_nesterov=False).op

  def _resource_apply_sparse(self, grad, var, indices):
    mom = self._zdic[var.name]
    return training_ops.resource_sparse_apply_momentum(
        var.handle, mom.handle,
        math_ops.cast(self._learning_rate_tensor, grad.dtype),
        grad, indices,
        math_ops.cast(self._momentum_tensor, grad.dtype),
        use_locking=self._use_locking,
        use_nesterov=False)
