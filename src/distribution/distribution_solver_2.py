
"""
TODO:
- make an abstract class DistributionSolver and derive from it
- use networkx
- optimize computations using numpy
- todo: comput all local supply might be called too many times, should i write my own optimizer?
- optimizer
- compute the gradient(!)

DONE:
- make a class, global variables(such as edge time length) becomes members
- make tests
"""


import collections
import numpy as np
from scipy import optimize


def check_shape(array_to_check, target_shape):
  if(array_to_check.shape != target_shape):
    raise Exception("init DistributionSolver params")



class DistributionSolver :
  """
  Create an instance of Distribution then call the solver with ``optimal_policy = my_solver.solve()``.

  The constructor has the following all-keyword arguments:
    * ``nb_countries``(``int``): number of countries/regional entities involved in the exchange
    * ``nb_time_steps``(``int``): number of time steps(days) for which the predictions of number of cases are available.
    * `` edge_transit_durations``(``array`` of ``float``): for each pair of countries, the estimated transit time of supply from the first to the second country(this is not assumed symmetric). The shape should be in the form ``(nb_countries, nb_countries)``, first index being the source country and second index the destination. The diagonal of the matrix is ignored.
    * ``exchange_cost``(``float``): constant cost penality to incur on each exchange.
    * ``supply_initial``(``array`` of ``float``): initial supply for each location. Shape is ``(nb_coutnries,)``.
    * ``supply_buffer``(``array`` of ``float``): amount of supply to keep at each location to serve as a safety buffer. Shape is ``(nb_countries,)``. If ``None`` will be initialized to zero.
    * ``nb_predicted_cases``(``array`` of ``float``): predicted number of cases for each location and each time step. Shape is ``(nb_countries, nb_time_steps)``.
    * ``tol``(``float``): optional tolenrance for floating point tests(default ``1.e-5``).
    
  The solver has the following all-keyword arguments:
    * ``policy_initial_guess``(``array`` of ``float``): an optional policy to serve as an initialization for the solver. Shape is ``(nb_countries, nb_countries, nb_time_steps)``. If ``None``, a zero vector will be used.
    * ``optimization_algorithm``(``string``): an optional method name to use. The name should be the same as used by Scipy. The default is ``'trust-constr'``.
    
  ``solve`` returns a optimal policy in the form of an array of shape ``(nb_countries, nb_countries, nb_time_steps)``. The first index represent the giving country and the second the country which is the destination of the exchange, and this is for each time step.
  """

  def __init__(self,
                nb_countries = None,
                nb_time_steps = 21,
                edge_transit_durations = None,
                exchange_cost = None,
                tol = 1.e-5,
                supply_initial = None,
                supply_buffer = None,
                nb_predicted_cases = None):
    self.nb_countries = nb_countries
    self.nb_time_steps = nb_time_steps
    self.policy_shape =(self.nb_countries, self.nb_countries, self.nb_time_steps)
    self.supply_by_country_shape =(self.nb_countries, self.nb_time_steps)
    self.exchange_cost = exchange_cost
    self.basically_zero = tol
    self.supply_initial = np.array(supply_initial)
    check_shape(self.supply_initial,(self.nb_countries, ))
    if(supply_buffer is None):
      self.supply_buffer = np.zeros((nb_countries, ), dtype = float)
    else :
      self.supply_buffer = np.array(supply_buffer)
    check_shape(self.supply_buffer,(self.nb_countries, ))
    self.nb_cases = np.array(nb_predicted_cases)
    check_shape(self.nb_cases,(self.nb_countries, self.nb_time_steps))
    self.edge_transit_durations = np.array(edge_transit_durations)
    if(not issubclass(self.edge_transit_durations.dtype.type, np.integer)):
      raise Exception("DistributionSolver init, wrong type")
    check_shape(self.edge_transit_durations,(self.nb_countries, self.nb_countries))
    self.local_supply_by_country = np.zeros(self.supply_by_country_shape, dtype = float)    


  """
  graph/network implemetation dependent part
  """
  def _check_if_is_edge_location(self, index):
    return isinstance(index, collections.Sequence)

  def _get_source_and_destination_indexes(self, edge_index):
    return edge_index
  """
  end network implementation dependent part
  """
  
  def _compute_slice_for_transiting_supplies(self, time, edge_index):
    source_country, destination_country = self._get_source_and_destination_indexes(edge_index)
    start_time = max(0, time - self.edge_transit_durations[edge_index] + 1)
    end_time = time + 1
    return source_country, destination_country, start_time, end_time

  def compute_local_supply_on_edge(self, policy, time, edge_index):
    source_country, destination_country, start_time, end_time = self._compute_slice_for_transiting_supplies(time, edge_index)
    supply = np.sum(policy[source_country, destination_country, start_time : end_time])
    return supply
  
  def _compute_selector_for_policy_receiver(self, time, country_index):
    return[ [[(t <= time - self.edge_transit_durations[j, country_index]) and(i == country_index)
                       for t in range(self.nb_time_steps) ]
                     for i in range(self.nb_countries) ]
                    for j in range(self.nb_countries) ]


  def compute_local_supply_in_country(self, policy, time, country_index):
    supply = self.supply_initial[country_index] - self.nb_cases[ country_index, time ]
    supply += - np.sum(policy[ country_index, :, 0 : time + 1 ])
    selectors = np.array(self._compute_selector_for_policy_receiver(time, country_index))
    supply += np.sum(policy[ selectors ])
    return supply

  def compute_local_supply(self, policy, time, location_index):
    if(self._check_if_is_edge_location(location_index)):
      local_supply = self.compute_local_supply_on_edge(policy, time, location_index)
    else :
      local_supply = self.compute_local_supply_in_country(policy, time, localtion_index)
    return local_supply

  def compute_all_local_supplies(self, policy, time):
    self.local_supply_by_country[ :, time ] =[ self.compute_local_supply_in_country(policy, time, country)
                                                   for country in range(self.nb_countries) ]

  def compute_all_local_supplies_for_all_times(self, policy):
    for time in range(self.nb_time_steps):
      self.compute_all_local_supplies(policy, time)

  def compute_objective_function(self, policy):
    self.compute_all_local_supplies_for_all_times(policy)
    objective_function_value = - sum( sum([ self.local_supply_by_country[ country, time ]
                                            for country in range(self.nb_countries) if self.local_supply_by_country[ country, time ] < 0. ])
                                       for time in range(self.nb_time_steps) )
    objective_function_value += sum(  sum( sum( self.exchange_cost for j in range(self.nb_countries) if abs(policy[ i, j, time ]) > self.basically_zero )
                                           for i in range(self.nb_countries) )
                                    for time in range(self.nb_time_steps))
    return objective_function_value

  def compute_constraints_on_edge(self, policy, time, location_index):
    return 0.

  def  compute_constraints_in_country(self, policy, time, country_index):
    return self.local_supply_by_country[ country_index, time ] - self.supply_buffer[ country_index ]

  def compute_local_constraints(self, policy, time, location_index):
    if(self._check_if_is_edge_location(location_index)):
      constraint = self.compute_constraints_on_edge(policy, time, location_index)
    else :
      constraint = self.compute_constraints_in_country(policy, time, location_index)
    return constraint

  def compute_constraints(self, policy):
    self.compute_all_local_supplies_for_all_times(policy)
    constraints = np.array([[ self.compute_local_constraints(policy, time, country) for country in range(self.nb_countries) ] for time in range(self.nb_time_steps) ])
    return constraints

  def compute_objective_function_flat_input(self, policy_flat):
    policy = np.reshape(policy_flat, self.policy_shape)
    return self.compute_objective_function(policy)
    
  def compute_constraints_flat_input(self, policy_flat):
    policy = np.reshape(policy_flat, self.policy_shape)
    return self.compute_constraints(policy).flatten()

  def solve(self, policy_initial_guess = None, optimization_algorithm = 'trust-constr'):
    if(policy_initial_guess is None):
      policy_initial_guess = np.zeros(self.policy_shape, dtype = float)
    solution = optimize.minimize(self.compute_objective_function_flat_input,
                                    policy_initial_guess.flatten(),
                                    method = optimization_algorithm,
                                    constraints = optimize.NonlinearConstraint(self.compute_constraints_flat_input, 0., np.inf),
                                    options = { 'verbose' : 1 })
    return solution


























