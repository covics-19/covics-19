

from distribution_solver_2 import *

import unittest

import numpy as np

def create_a_base_solver(nb_countries = 4,
                          nb_time_steps = 3,
                          edge_transit_durations = None):
    if(edge_transit_durations is None):
      edge_transit_durations = np.ones((nb_countries, nb_countries), dtype = int)
    supply_initial = 10. * np.ones((nb_countries,), dtype = float)
    supply_buffer = np.ones((nb_countries,), dtype = float)
    policy_initial_guess = np.zeros((nb_countries, nb_countries, nb_time_steps), dtype = float)
    nb_predicted_cases = 5. * np.ones((nb_countries, nb_time_steps), dtype = float)
    solver = DistributionSolver(nb_countries = nb_countries,
                                 nb_time_steps = nb_time_steps,
                                 edge_transit_durations = edge_transit_durations,
                                 exchange_cost = 1.22,
                                 supply_initial = supply_initial,
                                 supply_buffer = supply_buffer,
                                 policy_initial_guess = policy_initial_guess,
                                 nb_predicted_cases = nb_predicted_cases)
    return solver



class TestDistributionSolver(unittest.TestCase):

  def test_type_check_transit_durations(self):
    edge_transit_durations = np.ones((4,4), dtype = float)
    supply_initial = 10. * np.ones((4,), dtype = float)
    supply_buffer = np.ones((4,), dtype = float)
    policy_initial_guess = np.zeros((4, 4, 3), dtype = float)
    nb_predicted_cases = 5. * np.ones((4, 3), dtype = float)
    # i dont understand how assertRaises work
    try :
      DistributionSolver(nb_countries = 4,
                                 nb_time_steps = 3,
                                 edge_transit_durations = edge_transit_durations,
                                 exchange_cost = 1.22,
                                 supply_initial = supply_initial,
                                 supply_buffer = supply_buffer,
                                 policy_initial_guess = policy_initial_guess,
                                 nb_predicted_cases = nb_predicted_cases)
    except(BaseException):
      success = True
    else :
      success = False
    self.assertTrue(success)

  def test_check_if_is_edge_location(self):
    solver = create_a_base_solver()
    self.assertTrue(solver._check_if_is_edge_location((0, 0)))
    self.assertFalse(solver._check_if_is_edge_location(0))

  def test_get_source_and_destination_indexes(self):
    solver = create_a_base_solver()
    source, dest = solver._get_source_and_destination_indexes((1, 3))
    self.assertEqual(source, 1)
    self.assertEqual(dest, 3)


  def test_compute_selector_for_policy_receiver(self):
    solver = create_a_base_solver(nb_countries = 2, nb_time_steps = 3, edge_transit_durations = np.ones((2, 2), dtype = int))
    selector = solver._compute_selector_for_policy_receiver(2, 1)
    self.assertSequenceEqual(selector,[ [  [ False, False, False ],[ True, True, False ] ],[ [ False, False, False ],[ True, True, False ] ] ])
    #self.assertSequenceEqual(selector,[ [ 0, 0, 1, 1 ],[ 1, 1, 1, 1 ],[ 0, 1, 0, 1  ] ])
    
  def _subcheck_test_compute_slice_for_transiting_supplies(self, solver, time, edge, source, dest, start, end):
    s, d, ts, te = solver._compute_slice_for_transiting_supplies(time, edge)
    self.assertEqual(s, source)
    self.assertEqual(d, dest)
    self.assertEqual(ts, start)
    self.assertEqual(te, end)
  
  def test_compute_slice_for_transiting_supplies(self):
    solver = create_a_base_solver(nb_countries = 2, nb_time_steps = 3, edge_transit_durations = np.ones((2, 2), dtype = int))
    self._subcheck_test_compute_slice_for_transiting_supplies(solver, 0,(0, 0), 0, 0, 0, 1)
    self._subcheck_test_compute_slice_for_transiting_supplies(solver, 0,(0, 1), 0, 1, 0, 1)
    self._subcheck_test_compute_slice_for_transiting_supplies(solver, 0,(1, 0), 1, 0, 0, 1)
    self._subcheck_test_compute_slice_for_transiting_supplies(solver, 0,(1, 1), 1, 1, 0, 1)
    self._subcheck_test_compute_slice_for_transiting_supplies(solver, 1,(0, 0), 0, 0, 1, 2)
    self._subcheck_test_compute_slice_for_transiting_supplies(solver, 1,(0, 1), 0, 1, 1, 2)
    self._subcheck_test_compute_slice_for_transiting_supplies(solver, 1,(1, 0), 1, 0, 1, 2)
    self._subcheck_test_compute_slice_for_transiting_supplies(solver, 1,(1, 1), 1, 1, 1, 2)
    self._subcheck_test_compute_slice_for_transiting_supplies(solver, 2,(0, 0), 0, 0, 2, 3)
    self._subcheck_test_compute_slice_for_transiting_supplies(solver, 2,(0, 1), 0, 1, 2, 3)
    self._subcheck_test_compute_slice_for_transiting_supplies(solver, 2,(1, 0), 1, 0, 2, 3)
    self._subcheck_test_compute_slice_for_transiting_supplies(solver, 2,(1, 1), 1, 1, 2, 3)


  def test_compute_local_supply_on_edge(self):
    edge_transit_durations = np.ones((4,4), dtype = int)
    supply_initial = 10. * np.ones((4,), dtype = float)
    supply_buffer = np.ones((4,), dtype = float)
    policy_initial_guess = np.zeros((4, 4, 3), dtype = float)
    nb_predicted_cases = 5. * np.ones((4, 3), dtype = float)
    solver = DistributionSolver(nb_countries = 4,
                                 nb_time_steps = 3,
                                 edge_transit_durations = edge_transit_durations,
                                 exchange_cost = 1.22,
                                 supply_initial = supply_initial,
                                 supply_buffer = supply_buffer,
                                 policy_initial_guess = policy_initial_guess,
                                 nb_predicted_cases = nb_predicted_cases)
    policy = np.zeros((4,4,3))
    policy[ 2, 1, 1 ] = 1.
    self.assertAlmostEqual(solver.compute_local_supply_on_edge(policy, 0,(2, 1)), 0.)
    self.assertAlmostEqual(solver.compute_local_supply_on_edge(policy, 1,(2, 1)), 1.)
    self.assertAlmostEqual(solver.compute_local_supply_on_edge(policy, 2,(2, 1)), 0.)
    self.assertAlmostEqual(solver.compute_local_supply_on_edge(policy, 0,(0, 0)), 0.)
    self.assertAlmostEqual(solver.compute_local_supply_on_edge(policy, 1,(0, 0)), 0.)
    self.assertAlmostEqual(solver.compute_local_supply_on_edge(policy, 2,(0, 0)), 0.)
    self.assertAlmostEqual(solver.compute_local_supply_on_edge(policy, 0,(0, 1)), 0.)
    self.assertAlmostEqual(solver.compute_local_supply_on_edge(policy, 1,(0, 1)), 0.)
    self.assertAlmostEqual(solver.compute_local_supply_on_edge(policy, 2,(0, 1)), 0.)
    self.assertAlmostEqual(solver.compute_local_supply_on_edge(policy, 0,(0, 2)), 0.)
    self.assertAlmostEqual(solver.compute_local_supply_on_edge(policy, 1,(0, 2)), 0.)
    self.assertAlmostEqual(solver.compute_local_supply_on_edge(policy, 2,(0, 2)), 0.)
    self.assertAlmostEqual(solver.compute_local_supply_on_edge(policy, 0,(0, 3)), 0.)
    self.assertAlmostEqual(solver.compute_local_supply_on_edge(policy, 1,(0, 3)), 0.)
    self.assertAlmostEqual(solver.compute_local_supply_on_edge(policy, 2,(0, 3)), 0.)
    self.assertAlmostEqual(solver.compute_local_supply_on_edge(policy, 0,(1, 0)), 0.)
    self.assertAlmostEqual(solver.compute_local_supply_on_edge(policy, 1,(1, 0)), 0.)
    self.assertAlmostEqual(solver.compute_local_supply_on_edge(policy, 2,(1, 0)), 0.)
    self.assertAlmostEqual(solver.compute_local_supply_on_edge(policy, 0,(1, 1)), 0.)
    self.assertAlmostEqual(solver.compute_local_supply_on_edge(policy, 1,(1, 1)), 0.)
    self.assertAlmostEqual(solver.compute_local_supply_on_edge(policy, 2,(1, 1)), 0.)
    self.assertAlmostEqual(solver.compute_local_supply_on_edge(policy, 0,(1, 2)), 0.)
    self.assertAlmostEqual(solver.compute_local_supply_on_edge(policy, 1,(1, 2)), 0.)
    self.assertAlmostEqual(solver.compute_local_supply_on_edge(policy, 2,(1, 2)), 0.)
    self.assertAlmostEqual(solver.compute_local_supply_on_edge(policy, 0,(1, 3)), 0.)
    self.assertAlmostEqual(solver.compute_local_supply_on_edge(policy, 1,(1, 3)), 0.)
    self.assertAlmostEqual(solver.compute_local_supply_on_edge(policy, 2,(1, 3)), 0.)
    self.assertAlmostEqual(solver.compute_local_supply_on_edge(policy, 0,(2, 0)), 0.)
    self.assertAlmostEqual(solver.compute_local_supply_on_edge(policy, 1,(2, 0)), 0.)
    self.assertAlmostEqual(solver.compute_local_supply_on_edge(policy, 2,(2, 0)), 0.)
    self.assertAlmostEqual(solver.compute_local_supply_on_edge(policy, 0,(2, 2)), 0.)
    self.assertAlmostEqual(solver.compute_local_supply_on_edge(policy, 1,(2, 2)), 0.)
    self.assertAlmostEqual(solver.compute_local_supply_on_edge(policy, 2,(2, 2)), 0.)
    self.assertAlmostEqual(solver.compute_local_supply_on_edge(policy, 0,(2, 3)), 0.)
    self.assertAlmostEqual(solver.compute_local_supply_on_edge(policy, 1,(2, 3)), 0.)
    self.assertAlmostEqual(solver.compute_local_supply_on_edge(policy, 2,(2, 3)), 0.)
    self.assertAlmostEqual(solver.compute_local_supply_on_edge(policy, 0,(3, 0)), 0.)
    self.assertAlmostEqual(solver.compute_local_supply_on_edge(policy, 1,(3, 0)), 0.)
    self.assertAlmostEqual(solver.compute_local_supply_on_edge(policy, 2,(3, 0)), 0.)
    self.assertAlmostEqual(solver.compute_local_supply_on_edge(policy, 0,(3, 1)), 0.)
    self.assertAlmostEqual(solver.compute_local_supply_on_edge(policy, 1,(3, 1)), 0.)
    self.assertAlmostEqual(solver.compute_local_supply_on_edge(policy, 2,(3, 1)), 0.)
    self.assertAlmostEqual(solver.compute_local_supply_on_edge(policy, 0,(3, 2)), 0.)
    self.assertAlmostEqual(solver.compute_local_supply_on_edge(policy, 1,(3, 2)), 0.)
    self.assertAlmostEqual(solver.compute_local_supply_on_edge(policy, 2,(3, 2)), 0.)
    self.assertAlmostEqual(solver.compute_local_supply_on_edge(policy, 0,(3, 3)), 0.)
    self.assertAlmostEqual(solver.compute_local_supply_on_edge(policy, 1,(3, 3)), 0.)
    self.assertAlmostEqual(solver.compute_local_supply_on_edge(policy, 2,(3, 3)), 0.)
    
  def test_compute_local_supply_in_country(self):
    edge_transit_durations = np.ones((4,4), dtype = int)
    supply_initial = 10. * np.ones((4,), dtype = float)
    supply_buffer = np.ones((4,), dtype = float)
    policy_initial_guess = np.zeros((4, 4, 3), dtype = float)
    nb_predicted_cases = 5. * np.ones((4, 3), dtype = float)
    solver = DistributionSolver(nb_countries = 4,
                                 nb_time_steps = 3,
                                 edge_transit_durations = edge_transit_durations,
                                 exchange_cost = 1.22,
                                 supply_initial = supply_initial,
                                 supply_buffer = supply_buffer,
                                 policy_initial_guess = policy_initial_guess,
                                 nb_predicted_cases = nb_predicted_cases)
    policy = np.zeros((4,4,3))
    policy[2,1,1] = 1.
    self.assertAlmostEqual(solver.compute_local_supply_in_country(policy, 0, 0), 5.)
    self.assertAlmostEqual(solver.compute_local_supply_in_country(policy, 0, 1), 5.)
    self.assertAlmostEqual(solver.compute_local_supply_in_country(policy, 0, 2), 5.)
    self.assertAlmostEqual(solver.compute_local_supply_in_country(policy, 0, 3), 5.)
    self.assertAlmostEqual(solver.compute_local_supply_in_country(policy, 1, 0), 5.)
    self.assertAlmostEqual(solver.compute_local_supply_in_country(policy, 1, 1), 5.)
    self.assertAlmostEqual(solver.compute_local_supply_in_country(policy, 1, 2), 4.)
    self.assertAlmostEqual(solver.compute_local_supply_in_country(policy, 1, 3), 5.)
    self.assertAlmostEqual(solver.compute_local_supply_in_country(policy, 2, 0), 5.)
    self.assertAlmostEqual(solver.compute_local_supply_in_country(policy, 2, 1), 6.)
    self.assertAlmostEqual(solver.compute_local_supply_in_country(policy, 2, 2), 4.)
    self.assertAlmostEqual(solver.compute_local_supply_in_country(policy, 2, 3), 5.)
    
    
  def test_compute_all_local_supplies_for_all_times(self):
    edge_transit_durations = np.ones((4,4), dtype = int)
    supply_initial = 10. * np.ones((4,), dtype = float)
    supply_buffer = np.ones((4,), dtype = float)
    policy_initial_guess = np.zeros((4, 4, 3), dtype = float)
    nb_predicted_cases = 5. * np.ones((4, 3), dtype = float)
    nb_predicted_cases[ 2, 2 ] = 8.
    solver = DistributionSolver(nb_countries = 4,
                                 nb_time_steps = 3,
                                 edge_transit_durations = edge_transit_durations,
                                 exchange_cost = 1.22,
                                 supply_initial = supply_initial,
                                 supply_buffer = supply_buffer,
                                 policy_initial_guess = policy_initial_guess,
                                 nb_predicted_cases = nb_predicted_cases)
    policy = np.zeros((4,4,3))
    policy[ 3, 0, 1 ] = 1.
    solver.compute_all_local_supplies_for_all_times(policy)
    self.assertAlmostEqual(solver.local_supply_by_country[ 0, 0 ], 5.)
    self.assertAlmostEqual(solver.local_supply_by_country[ 1, 0 ], 5.)
    self.assertAlmostEqual(solver.local_supply_by_country[ 2, 0 ], 5.)
    self.assertAlmostEqual(solver.local_supply_by_country[ 3, 0 ], 5.)
    self.assertAlmostEqual(solver.local_supply_by_country[ 0, 1 ], 5.)
    self.assertAlmostEqual(solver.local_supply_by_country[ 1, 1 ], 5.)
    self.assertAlmostEqual(solver.local_supply_by_country[ 2, 1 ], 5.)
    self.assertAlmostEqual(solver.local_supply_by_country[ 3, 1 ], 4.)
    self.assertAlmostEqual(solver.local_supply_by_country[ 0, 2 ], 6.)
    self.assertAlmostEqual(solver.local_supply_by_country[ 1, 2 ], 5.)
    self.assertAlmostEqual(solver.local_supply_by_country[ 2, 2 ], 2.)
    self.assertAlmostEqual(solver.local_supply_by_country[ 3, 2 ], 4.)

  def test_compute_all_local_supplies_for_all_times_with_a_negative_value(self):
    edge_transit_durations = np.ones((4,4), dtype = int)
    supply_initial = 10. * np.ones((4,), dtype = float)
    supply_buffer = np.ones((4,), dtype = float)
    policy_initial_guess = np.zeros((4, 4, 3), dtype = float)
    nb_predicted_cases = 5. * np.ones((4, 3), dtype = float)
    nb_predicted_cases[ 2, 2 ] = 8.
    solver = DistributionSolver(nb_countries = 4,
                                 nb_time_steps = 3,
                                 edge_transit_durations = edge_transit_durations,
                                 exchange_cost = 1.22,
                                 supply_initial = supply_initial,
                                 supply_buffer = supply_buffer,
                                 policy_initial_guess = policy_initial_guess,
                                 nb_predicted_cases = nb_predicted_cases)
    policy = np.zeros((4,4,3))
    policy[ 0, 1, 0 ] = 7.
    solver.compute_all_local_supplies_for_all_times(policy)
    self.assertAlmostEqual(solver.local_supply_by_country[ 0, 0 ], -2.)
    self.assertAlmostEqual(solver.local_supply_by_country[ 1, 0 ], 5.)
    self.assertAlmostEqual(solver.local_supply_by_country[ 2, 0 ], 5.)
    self.assertAlmostEqual(solver.local_supply_by_country[ 3, 0 ], 5.)
    self.assertAlmostEqual(solver.local_supply_by_country[ 0, 1 ], -2.)
    self.assertAlmostEqual(solver.local_supply_by_country[ 1, 1 ], 12.)
    self.assertAlmostEqual(solver.local_supply_by_country[ 2, 1 ], 5.)
    self.assertAlmostEqual(solver.local_supply_by_country[ 3, 1 ], 5.)
    self.assertAlmostEqual(solver.local_supply_by_country[ 0, 2 ], -2.)
    self.assertAlmostEqual(solver.local_supply_by_country[ 1, 2 ], 12.)
    self.assertAlmostEqual(solver.local_supply_by_country[ 2, 2 ], 2.)
    self.assertAlmostEqual(solver.local_supply_by_country[ 3, 2 ], 5.)


  def test_compute_objective_function(self):
    edge_transit_durations = np.ones((4,4), dtype = int)
    supply_initial = 10. * np.ones((4,), dtype = float)
    supply_buffer = np.ones((4,), dtype = float)
    policy_initial_guess = np.zeros((4, 4, 3), dtype = float)
    nb_predicted_cases = 5. * np.ones((4, 3), dtype = float)
    nb_predicted_cases[ 2, 2 ] = 8.
    solver = DistributionSolver(nb_countries = 4,
                                 nb_time_steps = 3,
                                 edge_transit_durations = edge_transit_durations,
                                 exchange_cost = 1.22,
                                 supply_initial = supply_initial,
                                 supply_buffer = supply_buffer,
                                 policy_initial_guess = policy_initial_guess,
                                 nb_predicted_cases = nb_predicted_cases)
    policy = np.zeros((4,4,3))
    policy[ 3, 0, 1 ] = 1.
    self.assertAlmostEqual(solver.compute_objective_function(policy), 1.22)
    policy = np.zeros((4,4,3))
    policy[ 0, 1, 0 ] = 7.
    self.assertAlmostEqual(solver.compute_objective_function(policy), 6. + 1.22)
    
    
  def test_compute_constraints(self):
    edge_transit_durations = np.ones((4,4), dtype = int)
    supply_initial = 10. * np.ones((4,), dtype = float)
    supply_buffer = np.ones((4,), dtype = float)
    policy_initial_guess = np.zeros((4, 4, 3), dtype = float)
    nb_predicted_cases = 5. * np.ones((4, 3), dtype = float)
    nb_predicted_cases[ 2, 2 ] = 8.
    solver = DistributionSolver(nb_countries = 4,
                                 nb_time_steps = 3,
                                 edge_transit_durations = edge_transit_durations,
                                 exchange_cost = 1.22,
                                 supply_initial = supply_initial,
                                 supply_buffer = supply_buffer,
                                 policy_initial_guess = policy_initial_guess,
                                 nb_predicted_cases = nb_predicted_cases)
    policy = np.zeros((4,4,3))
    """
    self.assertAlmostEqual(solver.local_supply_by_country[ 0, 0 ], 5.)
    self.assertAlmostEqual(solver.local_supply_by_country[ 1, 0 ], 5.)
    self.assertAlmostEqual(solver.local_supply_by_country[ 2, 0 ], 5.)
    self.assertAlmostEqual(solver.local_supply_by_country[ 3, 0 ], 5.)
    self.assertAlmostEqual(solver.local_supply_by_country[ 0, 1 ], 5.)
    self.assertAlmostEqual(solver.local_supply_by_country[ 1, 1 ], 5.)
    self.assertAlmostEqual(solver.local_supply_by_country[ 2, 1 ], 5.)
    self.assertAlmostEqual(solver.local_supply_by_country[ 3, 1 ], 5.)
    self.assertAlmostEqual(solver.local_supply_by_country[ 0, 2 ], 5.)
    self.assertAlmostEqual(solver.local_supply_by_country[ 1, 2 ], 5.)
    self.assertAlmostEqual(solver.local_supply_by_country[ 2, 2 ], 2.)
    self.assertAlmostEqual(solver.local_supply_by_country[ 3, 2 ], 5.)
    """
    constraints =[ [ 4., 4., 4., 4. ],[ 4., 4., 4., 4. ],[ 4., 4., 1., 4. ] ]
    self.assertTrue(np.allclose(solver.compute_constraints(policy), np.array(constraints)))
    policy[ 0, 1, 0 ] = 7.
    constraints =[ [ -3., 4., 4., 4. ],[ -3., 11., 4., 4. ],[ -3., 11., 1., 4. ] ]
    self.assertTrue(np.allclose(solver.compute_constraints(policy), np.array(constraints)))
    
  def test_solve(self):
    solver = create_a_base_solver()
    solution = solver.solve()
    print(solution)
    self.assertTrue(True)




if __name__ == '__main__' :
  unittest.main()
