"""
NOTE: a policy as is defined here includes the possibility of a location exchanging with itself. Since this sort of exchange is higly penalised, it should be automatically optimized away.
"""




import functools
import numpy as np
import operator
from scipy import linalg
from scipy import optimize
from scipy import sparse

from distribution_solver_abstract import *


class DistributionSolverLinear :
  """
  Create an instance of Distribution then call the solver with ``optimal_policy = my_solver.solve()``.

  The constructor has the following all-keyword arguments:
    * ``nb_countries``(``int``): number of countries/regional entities involved in the exchange
    * ``nb_time_steps``(``int``): number of time steps(days) for which the predictions of number of cases are available.
    * `` edge_transit_durations``(``array`` of ``float``): for each pair of countries, the estimated transit time of supply from the first to the second country(this is not assumed symmetric). The shape should be in the form ``(nb_countries, nb_countries)``, first index being the source country and second index the destination. The diagonal of the matrix is ignored.
    * ``exchange_cost_factor``(``float``): constant factor for the linear cost penality incurred on each exchange.
    * ``supply_predictions``(``array`` of ``float``): predicted available supply for each location at each time step. Shape is ``(nb_coutnries, nb_time_steps)``.
    * ``supply_buffer``(``array`` of ``float``): amount of supply to keep at each location to serve as a safety buffer. Shape is ``(nb_countries,)``. If ``None`` will be initialized to zero.
    * ``future_discount_factor``(``float``): multiplicator applied at each time step. If less than 1, future predictions will have less weight in the optimization process. Incompatible with ``future_discount_coefficients``.
    * ``future_discount_coefficients``(``array`` of ``float``): discount coefficients for each time step. Expected shape is ``(nb_time_steps,)``. Incompatible with ``future_discount_factor``.
    * ``tol``(``float``): optional tolerance for floating point tests(default ``1.e-5``).
    * ``do_use_sparse``(``bool``):(optional) if True, the solver will use a sparse matrix(default is ``False``).
    
  The solver has the following all-keyword arguments:
    * ``predicted_supply``(``array`` of ``float``): if not ``None``, will be used to update the value given to the initializer with this new set of predictions.
    * ``method_scipy_name``(``string``): an optional method name to use. The name should be the same as used by Scipy. The default is ``'interior-point'``.
    * ``optimizer_options``(dictionary): options to pass to the optimizer(compare ``scipy.optimize.linprog`` documentation). Default is the following dictionary:
      ``{ maxiter : 1000, disp : False, autoscale : False }``.
    
  ``solve`` returns a optimal policy and the ``scipy.optimize.OptimzeResult`` object returned by ``scipy.optimize.linprog``. The optimal policy comes in the form of an array of shape ``(nb_countries, nb_countries, nb_time_steps)``. The first index of the policy represent the giving country and the second the country which is the destination of the exchange, and this is for each time step.
  """

  def __init__(self,
                nb_countries = None,
                nb_time_steps = 21,
                edge_transit_durations = None,
                exchange_cost_factor = 0.,
                tol = 1.e-5,
                supply_predictions = None,
                supply_buffer = None,
                future_discount_factor = None,
                future_discount_coefficients = None,
                do_use_sparse = False):
    self.nb_countries = nb_countries
    self.nb_time_steps = nb_time_steps
    self.policy_shape =(self.nb_countries, self.nb_countries, self.nb_time_steps)
    self.system_dimension = functools.reduce(operator.mul, self.policy_shape)
    self.supply_by_country_shape =(self.nb_countries, self.nb_time_steps)
    self.exchange_cost_factor = exchange_cost_factor
    self.basically_zero = tol
    if(supply_buffer is None):
      self.supply_buffer = np.zeros((nb_countries, ), dtype = float)
    else :
      self.supply_buffer = np.array(supply_buffer)
    check_shape(self.supply_buffer,(self.nb_countries, ))
    self.supply_predictions = np.array(supply_predictions)
    check_shape(self.supply_predictions,(self.nb_countries, self.nb_time_steps))
    self.edge_transit_durations = np.array(edge_transit_durations)
    if(not issubclass(self.edge_transit_durations.dtype.type, np.integer)):
      raise Exception("DistributionSolver init, wrong type")
    check_shape(self.edge_transit_durations,(self.nb_countries, self.nb_countries))
    self.future_discount_coefficients = prepare_future_discount_coefficients(self.nb_time_steps, future_discount_factor, future_discount_coefficients)
    check_shape(self.future_discount_coefficients,(self.nb_time_steps, ))
    self.do_use_sparse = do_use_sparse
    self.is_A_ub_computed = False
    self.is_b_ub_computed = False
    self.is_c_computed = False


  def compute_b_ub(self):
    if(self.is_b_ub_computed):
      return
    B = np.tile(self.supply_buffer,(self.nb_time_steps, 1)).transpose().flatten()
    self.b_ub = self.supply_predictions.flatten() - B
    self.is_b_ub_computed = True
    
  def _compute_giver_matrix_block(self):
    small_block = np.tri(self.nb_time_steps, dtype = float)
    return np.block(self.nb_countries *[ small_block, ])

  def _compute_giver_matrix(self):
    giver_matrix_block = self._compute_giver_matrix_block()
    if(self.do_use_sparse):
      G = sparse.block_diag(self.nb_countries *(giver_matrix_block, ))
    else :
      # thatd be great if linalg.block_diag had the same syntax as sparse.block_diag
      G = linalg.block_diag(* self.nb_countries *(giver_matrix_block, ))
    return G
    
  def _compute_receiver_small_matrix_block(self, giver_index, receiver_index):
    return np.tri(self.nb_time_steps, k = - self.edge_transit_durations[giver_index, receiver_index], dtype = float)
    
  def _compute_receiver_big_matrix_block(self, receiver_index):
    small_blocks =[ self._compute_receiver_small_matrix_block(giver_index, receiver_index) for giver_index in range(self.nb_countries) ]
    if(self.do_use_sparse):
      big_block = sparse.block_diag(small_blocks)
    else :
      big_block = linalg.block_diag(* small_blocks)
    return big_block
    
  def _compute_receiver_matrix(self):
    blocks =[ [ self._compute_receiver_big_matrix_block(i) for i in range(self.nb_countries) ] ]
    if(self.do_use_sparse):
      R = sparse.bmat(blocks)
    else :
      R = np.block(blocks)
    return R
    
   
  def compute_A_ub(self):
    if(self.is_A_ub_computed):
      return
    G = self._compute_giver_matrix()
    R = self._compute_receiver_matrix()
    self.A_ub = G - R
    self.is_A_ub_computed = True


  def compute_c(self):
    if(self.is_c_computed):
      return
    if(not self.is_A_ub_computed):
      raise Exception("???")
    CS = - np.tile(self.future_discount_coefficients, self.nb_countries)
    Cpi = self.exchange_cost_factor * np.tile(self.future_discount_coefficients, self.nb_countries * self.nb_countries)
    self.c = - self.A_ub.transpose() @ CS + Cpi
    self.is_c_computed = True

  def update_predictions(self, new_supply_predictions):
    self.supply_predictions = new_supply_predictions
    self.is_b_ub_computed = False

  def assemble_system(self):
    self.compute_b_ub()
    self.compute_A_ub()
    self.compute_c()
    
  def solve(self,
             predicted_supply = None,
             method_scipy_name = 'interior-point',
             optimizer_options = { 'maxiter' : 1000, 'disp' : False, 'autoscale' : False }):
    if(predicted_supply is not None):
      self.update_predictions(predicted_supply)
    self.assemble_system()
    optimization_results = optimize.linprog(self.c,
                                               A_ub = self.A_ub,
                                               b_ub = self.b_ub,
                                               bounds =((0, None), ),
                                               method = method_scipy_name,
                                               options = optimizer_options)
    solution = optimization_results.x.reshape(self.policy_shape)
    return(solution, optimization_results)



























