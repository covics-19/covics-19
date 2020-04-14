
from distribution_solver_linear import *

import unittest

import numpy as np

def create_solver_with_default_values(nb_time_steps = 4,
                                       nb_countries = 3,
                                       edge_transit_durations = None,
                                       supply_predictions = None,
                                       supply_buffer = None,
                                       exchange_cost_factor = 0.1,
                                       future_discount_factor = 0.95,
                                       future_discount_coefficients = None,
                                       do_use_sparse = False):
  if(edge_transit_durations is None):
    edge_transit_durations = np.ones((nb_countries, nb_countries), dtype = int)
  if(supply_predictions is None):
    supply_predictions = np.ones((nb_countries, nb_time_steps), dtype = float)
  if(supply_buffer is None):
    supply_buffer = 0.1 * np.ones((nb_countries, ), dtype = float)
  
  solver = DistributionSolverLinear(nb_countries = nb_countries,
                nb_time_steps = nb_time_steps,
                edge_transit_durations = edge_transit_durations,
                exchange_cost_factor = exchange_cost_factor,
                supply_predictions = supply_predictions,
                supply_buffer = supply_buffer,
                future_discount_factor = future_discount_factor,
                future_discount_coefficients = future_discount_coefficients,
                do_use_sparse = do_use_sparse)
  return solver


def make_receiver_matrix(test_case = 0):
  if(test_case == 0):
    receiver_matrix = np.zeros((4*3,4*3*3), dtype = float)
    receiver_matrix[1, 0] = 1.
    receiver_matrix[2, 0] = 1.
    receiver_matrix[2, 1] = 1.
    receiver_matrix[3, 0] = 1.
    receiver_matrix[3, 1] = 1.
    receiver_matrix[3, 2] = 1.
    receiver_matrix[5, 4] = 1.
    receiver_matrix[6, 4] = 1.
    receiver_matrix[6, 5] = 1.
    receiver_matrix[7, 4] = 1.
    receiver_matrix[7, 5] = 1.
    receiver_matrix[7, 6] = 1.
    receiver_matrix[2, 12] = 1.
    receiver_matrix[3, 12] = 1.
    receiver_matrix[3, 13] = 1.
    receiver_matrix[5, 16] = 1.
    receiver_matrix[6, 16] = 1.
    receiver_matrix[6, 17] = 1.
    receiver_matrix[7, 16] = 1.
    receiver_matrix[7, 17] = 1.
    receiver_matrix[7, 18] = 1.
    receiver_matrix[3, 24] = 1.
    receiver_matrix[5, 28] = 1.
    receiver_matrix[6, 28] = 1.
    receiver_matrix[6, 29] = 1.
    receiver_matrix[7, 28] = 1.
    receiver_matrix[7, 29] = 1.
    receiver_matrix[7, 30] = 1.
    receiver_matrix[10, 32] = 1.
    receiver_matrix[11, 32] = 1.
    receiver_matrix[11, 33] = 1.
  return receiver_matrix

def make_giver_matrix(test_case = 0):
  if(test_case == 0):
    giver_matrix = np.zeros((4*3,4*3*3), dtype = float)
    block = np.tri(4, dtype = float)
    giver_matrix[0:4, 0:4] = block
    giver_matrix[0:4, 4:8] = block
    giver_matrix[0:4, 8:12] = block
    giver_matrix[4:8, 12:16] = block
    giver_matrix[4:8, 16:20] = block
    giver_matrix[4:8, 20:24] = block
    giver_matrix[8:12, 24:28] = block
    giver_matrix[8:12, 28:32] = block
    giver_matrix[8:12, 32:36] = block    
  return giver_matrix

def make_A_ub(test_case = 1):
  if(test_case == 0):
    R = make_receiver_matrix(test_case = 0)
    G = make_giver_matrix(test_case = 0)
    A_ub = G - R
  elif(test_case == 1):
    A_ub = np.zeros((12, 6), dtype = float)
    block = - np.tri(3, dtype = float).transpose()
    A_ub[0, 0] = -1
    A_ub[1, 1] = -1
    A_ub[2, 2] = -1
    A_ub[3:6, 0:3] = block
    A_ub[3, 4] = 1
    A_ub[3, 5] = 1
    A_ub[4, 5] = 1
    A_ub[6, 1] = 1
    A_ub[6, 2] = 1
    A_ub[7, 2] = 1
    A_ub[6:9, 3:6] = block
    A_ub[9, 3] = -1
    A_ub[9, 4] = -1
    A_ub[10, 4] = -1
    A_ub[10, 5] = -1
    A_ub[11, 5] = -1
    A_ub = - A_ub.transpose()
  return A_ub

def make_c(test_case = 1):
  if(test_case == 1):
    c = np.array(3 *[ 1 + 0.11, 0.98 *(1 + 0.11), 0.98 * 0.98 *(1. + 0.11) ]
                                  +[ 1 + 0.98 + 0.11, 0.98 *(1 + 0.98 + 0.11), 0.98 * 0.98 *(1. + 0.11) ],
                                 dtype = float)
  return c

def make_b_ub(test_case = 0):
  if(test_case == 0):
    b_ub =  np.array([[ 1 - 0.5, 2 - 0.5, 3 - 0.5, 4 -0.5 ],[ 1 - 0.1, 0 - 0.1, -1-0.1, 0.5-0.1 ],[ 1-2, 1-2, 1-2, 1-2 ] ], dtype = float).flatten()
  return b_ub

class TestDistributionSolverLinear(unittest.TestCase):

  def test_future_discount_coefficients_copy(self):
    future_discount_coefficients = np.arange(1., 5., dtype = float)
    solver = create_solver_with_default_values(nb_time_steps = 4,
                                                future_discount_factor = None,
                                                future_discount_coefficients = future_discount_coefficients)
    self.assertTrue(np.allclose(future_discount_coefficients, solver.future_discount_coefficients))
  
  
  def test_future_discount_coefficients_computation(self):
    future_discount_coefficients = np.array([ 1., 0.95, 0.95*0.95, 0.95*0.95*0.95 ])
    solver = create_solver_with_default_values(nb_time_steps = 4,
                                                future_discount_factor = 0.95,
                                                future_discount_coefficients = None)
    self.assertTrue(np.allclose(future_discount_coefficients, solver.future_discount_coefficients))
     
  def test_compute_b_ub(self):
    nb_time_steps = 4
    nb_countries = 3
    supply_buffer = np.array([0.5, 0.1, 2 ], dtype = float)
    supply_predictions = np.array([[ 1, 2, 3, 4 ],[ 1, 0, -1, 0.5 ],[ 1, 1, 1, 1 ] ], dtype = float)
    solver = create_solver_with_default_values(nb_time_steps = nb_time_steps,
                                                nb_countries = nb_countries,
                                                supply_buffer = supply_buffer,
                                                supply_predictions = supply_predictions)
    expected_b_ub = make_b_ub(test_case = 0)
    solver.compute_b_ub()
    self.assertTrue(np.allclose(expected_b_ub, solver.b_ub))
    
    
  def test_compute_giver_matrix_block(self):
    solver = create_solver_with_default_values(nb_time_steps = 4, nb_countries = 3, do_use_sparse = False)
    base_block =[[1,1,1,1],[0,1,1,1],[0,0,1,1],[0,0,0,1]]
    expected_block = np.array(3 * base_block, dtype = float).transpose()
    computed_block = solver._compute_giver_matrix_block()
    self.assertTrue(np.allclose(computed_block, expected_block))

  def test_compute_giver_matrix_block_sparse(self):
    # Note: right now sparseness is not involved in these computations
    solver = create_solver_with_default_values(nb_time_steps = 4, nb_countries = 3, do_use_sparse = True)
    base_block =[[1,1,1,1],[0,1,1,1],[0,0,1,1],[0,0,0,1]]
    expected_block = np.array(3 * base_block, dtype = float).transpose()
    try :
      computed_block = solver._compute_giver_matrix_block().toarray()
      self.assertTrue(np.allclose(computed_block, expected_block))
    except(AttributeError):
      self.assertTrue(True)
    else :
      self.assertTrue(False)
      
  def test_compute_giver_matrix(self):
    solver = create_solver_with_default_values(nb_time_steps = 4, nb_countries = 3, do_use_sparse = False)
    expected_matrix = make_giver_matrix(test_case = 0)
    self.assertTrue(np.allclose(expected_matrix, solver._compute_giver_matrix()))

  def test_compute_giver_matrix_sparse(self):
    solver = create_solver_with_default_values(nb_time_steps = 4, nb_countries = 3, do_use_sparse = True)
    expected_matrix = make_giver_matrix(test_case = 0)
    self.assertTrue(np.allclose(expected_matrix, solver._compute_giver_matrix().toarray()))

  def test_compute_receiver_small_matrix_block(self):
    edge_transit_durations = np.array([[1,2,3],[1,1,1],[6,5,2]], dtype = int)
    solver = create_solver_with_default_values(nb_time_steps = 4,
                                                nb_countries = 3,
                                                edge_transit_durations = edge_transit_durations,
                                                do_use_sparse = False)
    expected_block = np.array([[0,0,0,0],[1,0,0,0],[1,1,0,0],[1,1,1,0]],dtype=float)
    self.assertTrue(np.allclose(expected_block, solver._compute_receiver_small_matrix_block(0, 0)))
    expected_block = np.array([[0,0,0,0],[0,0,0,0],[1,0,0,0],[1,1,0,0]],dtype=float)
    self.assertTrue(np.allclose(expected_block, solver._compute_receiver_small_matrix_block(0, 1)))
    expected_block = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[1,0,0,0]],dtype=float)
    self.assertTrue(np.allclose(expected_block, solver._compute_receiver_small_matrix_block(0, 2)))
    expected_block = np.array([[0,0,0,0],[1,0,0,0],[1,1,0,0],[1,1,1,0]],dtype=float)
    self.assertTrue(np.allclose(expected_block, solver._compute_receiver_small_matrix_block(1, 0)))
    expected_block = np.array([[0,0,0,0],[1,0,0,0],[1,1,0,0],[1,1,1,0]],dtype=float)
    self.assertTrue(np.allclose(expected_block, solver._compute_receiver_small_matrix_block(1, 1)))
    expected_block = np.array([[0,0,0,0],[1,0,0,0],[1,1,0,0],[1,1,1,0]],dtype=float)
    self.assertTrue(np.allclose(expected_block, solver._compute_receiver_small_matrix_block(1, 2)))
    expected_block = np.zeros((4,4),dtype=float)
    self.assertTrue(np.allclose(expected_block, solver._compute_receiver_small_matrix_block(2, 0)))
    expected_block = np.zeros((4,4),dtype=float)
    self.assertTrue(np.allclose(expected_block, solver._compute_receiver_small_matrix_block(2, 1)))
    expected_block = np.array([[0,0,0,0],[0,0,0,0],[1,0,0,0],[1,1,0,0]],dtype=float)
    self.assertTrue(np.allclose(expected_block, solver._compute_receiver_small_matrix_block(2, 2)))

  def test_compute_receiver_small_matrix_block_sparse(self):
    # Note: right now sparseness is not involved in these computations
    edge_transit_durations = np.array([[1,2,3],[1,1,1],[6,5,2]], dtype = int)
    solver = create_solver_with_default_values(nb_time_steps = 4,
                                                nb_countries = 3,
                                                edge_transit_durations = edge_transit_durations,
                                                do_use_sparse = True)
    try :
      expected_block = np.array([[0,0,0,0],[1,0,0,0],[1,1,0,0],[1,1,1,0]],dtype=float)
      self.assertTrue(np.allclose(expected_block, solver._compute_receiver_small_matrix_block(0, 0).toarray()))
      expected_block = np.array([[0,0,0,0],[0,0,0,0],[1,0,0,0],[1,1,0,0]],dtype=float)
      self.assertTrue(np.allclose(expected_block, solver._compute_receiver_small_matrix_block(0, 1).toarray()))
      expected_block = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[1,0,0,0]],dtype=float)
      self.assertTrue(np.allclose(expected_block, solver._compute_receiver_small_matrix_block(0, 2).toarray()))
      expected_block = np.array([[0,0,0,0],[1,0,0,0],[1,1,0,0],[1,1,1,0]],dtype=float)
      self.assertTrue(np.allclose(expected_block, solver._compute_receiver_small_matrix_block(1, 0).toarray()))
      expected_block = np.array([[0,0,0,0],[1,0,0,0],[1,1,0,0],[1,1,1,0]],dtype=float)
      self.assertTrue(np.allclose(expected_block, solver._compute_receiver_small_matrix_block(1, 1).toarray()))
      expected_block = np.array([[0,0,0,0],[1,0,0,0],[1,1,0,0],[1,1,1,0]],dtype=float)
      self.assertTrue(np.allclose(expected_block, solver._compute_receiver_small_matrix_block(1, 2).toarray()))
      expected_block = np.zeros((4,4),dtype=float)
      self.assertTrue(np.allclose(expected_block, solver._compute_receiver_small_matrix_block(2, 0).toarray()))
      expected_block = np.zeros((4,4),dtype=float)
      self.assertTrue(np.allclose(expected_block, solver._compute_receiver_small_matrix_block(2, 1).toarray()))
      expected_block = np.array([[0,0,0,0],[0,0,0,0],[1,0,0,0],[1,1,0,0]],dtype=float)
      self.assertTrue(np.allclose(expected_block, solver._compute_receiver_small_matrix_block(2, 2).toarray()))
    except(AttributeError):
      self.assertTrue(True)
    else :
      self.assertTrue(False)


  def test_compute_receiver_big_matrix_block(self):
    edge_transit_durations = np.array([[1,2,3],[1,1,1],[6,5,2]], dtype = int)
    solver = create_solver_with_default_values(nb_time_steps = 4,
                                                nb_countries = 3,
                                                edge_transit_durations = edge_transit_durations,
                                                do_use_sparse = False)
    expected_block = np.zeros((12,12), dtype = float)
    expected_block[1, 0] = 1.
    expected_block[2, 0] = 1.
    expected_block[2, 1] = 1.
    expected_block[3, 0] = 1.
    expected_block[3, 1] = 1.
    expected_block[3, 2] = 1.
    expected_block[5, 4] = 1.
    expected_block[6, 4] = 1.
    expected_block[6, 5] = 1.
    expected_block[7, 4] = 1.
    expected_block[7, 5] = 1.
    expected_block[7, 6] = 1.
    self.assertTrue(np.allclose(expected_block, solver._compute_receiver_big_matrix_block(0)))
    expected_block = np.zeros((12,12), dtype = float)
    expected_block[2, 0] = 1.
    expected_block[3, 0] = 1.
    expected_block[3, 1] = 1.
    expected_block[5, 4] = 1.
    expected_block[6, 4] = 1.
    expected_block[6, 5] = 1.
    expected_block[7, 4] = 1.
    expected_block[7, 5] = 1.
    expected_block[7, 6] = 1.
    self.assertTrue(np.allclose(expected_block, solver._compute_receiver_big_matrix_block(1)))
    expected_block = np.zeros((12,12), dtype = float)
    expected_block[3, 0] = 1.
    expected_block[5, 4] = 1.
    expected_block[6, 4] = 1.
    expected_block[6, 5] = 1.
    expected_block[7, 4] = 1.
    expected_block[7, 5] = 1.
    expected_block[7, 6] = 1.
    expected_block[10, 8] = 1.
    expected_block[11, 8] = 1.
    expected_block[11, 9] = 1.
    self.assertTrue(np.allclose(expected_block, solver._compute_receiver_big_matrix_block(2)))

  def test_compute_receiver_big_matrix_block_sparse(self):
    edge_transit_durations = np.array([[1,2,3],[1,1,1],[6,5,2]], dtype = int)
    solver = create_solver_with_default_values(nb_time_steps = 4,
                                                nb_countries = 3,
                                                edge_transit_durations = edge_transit_durations,
                                                do_use_sparse = True)
    expected_block = np.zeros((12,12), dtype = float)
    expected_block[1, 0] = 1.
    expected_block[2, 0] = 1.
    expected_block[2, 1] = 1.
    expected_block[3, 0] = 1.
    expected_block[3, 1] = 1.
    expected_block[3, 2] = 1.
    expected_block[5, 4] = 1.
    expected_block[6, 4] = 1.
    expected_block[6, 5] = 1.
    expected_block[7, 4] = 1.
    expected_block[7, 5] = 1.
    expected_block[7, 6] = 1.
    self.assertTrue(np.allclose(expected_block, solver._compute_receiver_big_matrix_block(0).toarray()))
    expected_block = np.zeros((12,12), dtype = float)
    expected_block[2, 0] = 1.
    expected_block[3, 0] = 1.
    expected_block[3, 1] = 1.
    expected_block[5, 4] = 1.
    expected_block[6, 4] = 1.
    expected_block[6, 5] = 1.
    expected_block[7, 4] = 1.
    expected_block[7, 5] = 1.
    expected_block[7, 6] = 1.
    self.assertTrue(np.allclose(expected_block, solver._compute_receiver_big_matrix_block(1).toarray()))
    expected_block = np.zeros((12,12), dtype = float)
    expected_block[3, 0] = 1.
    expected_block[5, 4] = 1.
    expected_block[6, 4] = 1.
    expected_block[6, 5] = 1.
    expected_block[7, 4] = 1.
    expected_block[7, 5] = 1.
    expected_block[7, 6] = 1.
    expected_block[10, 8] = 1.
    expected_block[11, 8] = 1.
    expected_block[11, 9] = 1.
    self.assertTrue(np.allclose(expected_block, solver._compute_receiver_big_matrix_block(2).toarray()))
    
    
  def test_compute_receiver_matrix(self):
    edge_transit_durations = np.array([[1,2,3],[1,1,1],[6,5,2]], dtype = int)
    solver = create_solver_with_default_values(nb_time_steps = 4,
                                                nb_countries = 3,
                                                edge_transit_durations = edge_transit_durations,
                                                do_use_sparse = False)
    expected_matrix = make_receiver_matrix(test_case = 0)
    self.assertTrue(np.allclose(expected_matrix, solver._compute_receiver_matrix()))
    
  def test_compute_receiver_matrix_sparse(self):
    edge_transit_durations = np.array([[1,2,3],[1,1,1],[6,5,2]], dtype = int)
    solver = create_solver_with_default_values(nb_time_steps = 4,
                                                nb_countries = 3,
                                                edge_transit_durations = edge_transit_durations,
                                                do_use_sparse = True)
    expected_matrix = make_receiver_matrix(test_case = 0)
    self.assertTrue(np.allclose(expected_matrix, solver._compute_receiver_matrix().toarray()))
    
  def test_compute_A_ub(self):
    edge_transit_durations = np.array([[1,2,3],[1,1,1],[6,5,2]], dtype = int)
    solver = create_solver_with_default_values(nb_time_steps = 4,
                                                nb_countries = 3,
                                                edge_transit_durations = edge_transit_durations,
                                                do_use_sparse = False)
    expected_A_ub = make_A_ub(test_case = 0)
    solver.compute_A_ub()
    self.assertTrue(np.allclose(expected_A_ub, solver.A_ub))
    

  def test_compute_A_ub_sparse(self):
    edge_transit_durations = np.array([[1,2,3],[1,1,1],[6,5,2]], dtype = int)
    solver = create_solver_with_default_values(nb_time_steps = 4,
                                                nb_countries = 3,
                                                edge_transit_durations = edge_transit_durations,
                                                do_use_sparse = True)
    expected_A_ub = make_A_ub(test_case = 0)
    solver.compute_A_ub()
    self.assertTrue(np.allclose(expected_A_ub, solver.A_ub.toarray()))

  def test_compute_A_ub_test_2(self):
    edge_transit_durations = np.array([[ 1, 1 ],[ 1, 2 ] ], dtype = int)
    solver = create_solver_with_default_values(nb_time_steps = 3,
                                                nb_countries = 2,
                                                edge_transit_durations = edge_transit_durations,
                                                do_use_sparse = False,
                                                future_discount_factor = 0.98,
                                                exchange_cost_factor = 0.11)
    expected_A_ub = make_A_ub(test_case = 1)
    solver.compute_A_ub()
    self.assertTrue(np.allclose(expected_A_ub, solver.A_ub))

    
  def test_compute_c(self):
    edge_transit_durations = np.array([[1, 1],[1, 2] ], dtype = int)
    solver = create_solver_with_default_values(nb_time_steps = 3,
                                                nb_countries = 2,
                                                edge_transit_durations = edge_transit_durations,
                                                do_use_sparse = False,
                                                future_discount_factor = 0.98,
                                                exchange_cost_factor = 0.11)
    solver.compute_A_ub()
    solver.compute_c()
    expected_c = make_c(test_case = 1)
    self.assertTrue(np.allclose(solver.c, expected_c))
                                                
  def test_compute_c_sparse(self):
    edge_transit_durations = np.array([[1, 1],[1, 2] ], dtype = int)
    solver = create_solver_with_default_values(nb_time_steps = 3,
                                                nb_countries = 2,
                                                edge_transit_durations = edge_transit_durations,
                                                do_use_sparse = True,
                                                future_discount_factor = 0.98,
                                                exchange_cost_factor = 0.11)
    solver.compute_A_ub()
    solver.compute_c()
    expected_c = make_c(test_case = 1)
    self.assertTrue(np.allclose(solver.c, expected_c))
    
  def test_assemble_system(self):
    edge_transit_durations = np.array([[1,2,3],[1,1,1],[6,5,2]], dtype = int)
    supply_buffer = np.array([0.5, 0.1, 2. ], dtype = float)
    supply_predictions = np.array([[ 1., 2., 3., 4. ],[ 1., 0., -1., 0.5 ],[ 1., 1., 1., 1. ] ], dtype = float)
    solver = create_solver_with_default_values(nb_time_steps = 4,
                                                nb_countries = 3,
                                                supply_buffer = supply_buffer,
                                                supply_predictions = supply_predictions,
                                                edge_transit_durations = edge_transit_durations,
                                                do_use_sparse = False)
    self.assertTrue(not solver.is_b_ub_computed)
    self.assertTrue(not solver.is_A_ub_computed)
    solver.assemble_system()
    self.assertTrue(solver.is_b_ub_computed)
    self.assertTrue(solver.is_A_ub_computed)
    expected_A_ub = make_A_ub(test_case = 0)
    self.assertTrue(np.allclose(expected_A_ub, solver.A_ub))
    expected_b_ub = make_b_ub(test_case = 0)
    self.assertTrue(np.allclose(expected_b_ub, solver.b_ub))

  def test_assemble_system_2(self):
    edge_transit_durations = np.array([[1,1],[1,2]], dtype = int)
    supply_buffer = np.array([ 0.5, 0.1 ], dtype = float)
    supply_predictions = np.array([[ 1., 2., 3. ],[ 1., 0., -1. ] ], dtype = float)
    solver = create_solver_with_default_values(nb_time_steps = 3,
                                                nb_countries = 2,
                                                supply_buffer = supply_buffer,
                                                supply_predictions = supply_predictions,
                                                edge_transit_durations = edge_transit_durations,
                                                future_discount_factor = 0.98,
                                                exchange_cost_factor = 0.11,
                                                do_use_sparse = False)
    self.assertTrue(not solver.is_b_ub_computed)
    self.assertTrue(not solver.is_A_ub_computed)
    solver.assemble_system()
    self.assertTrue(solver.is_b_ub_computed)
    self.assertTrue(solver.is_A_ub_computed)
    expected_A_ub = make_A_ub(test_case = 1)
    self.assertTrue(np.allclose(expected_A_ub, solver.A_ub))
    expected_c = make_c(test_case = 1)
    self.assertTrue(np.allclose(expected_c, solver.c))

    
  def test_update_predictions(self):
    nb_time_steps = 4
    nb_countries = 3
    supply_buffer = np.array([0.5, 0.1, 2 ], dtype = float)
    supply_predictions = np.array([[ 1, 2, 3, 4 ],[ 1, 0, -1, 0.5 ],[ 1, 1, 1, 1 ] ], dtype = float)
    b_ub =  np.array([[ 1 - 0.5, 2 - 0.5, 3 - 0.5, 4 -0.5 ],[ 1 - 0.1, 0 - 0.1, -1-0.1, 0.5-0.1 ],[ 1-2, 1-2, 1-2, 1-2 ] ], dtype = float).flatten()
    solver = create_solver_with_default_values(nb_time_steps = nb_time_steps,
                                                nb_countries = nb_countries,
                                                supply_buffer = supply_buffer,
                                                supply_predictions = supply_predictions)
    solver.assemble_system()
    self.assertTrue(solver.is_b_ub_computed)
    self.assertTrue(solver.is_A_ub_computed)
    supply_predictions = np.array([[ 1, 2, 3, 4 ],[ 1, 0, -1, 0.5 ],[ 1, -1, 3., 1 ] ], dtype = float)
    solver.update_predictions(supply_predictions)
    self.assertTrue(not solver.is_b_ub_computed)
    self.assertTrue(solver.is_A_ub_computed)
    solver.assemble_system()
    expected_b_ub =  np.array([[ 1 - 0.5, 2 - 0.5, 3 - 0.5, 4 -0.5 ],[ 1 - 0.1, 0 - 0.1, -1-0.1, 0.5-0.1 ],[ 1-2, -1-2, 3-2, 1-2 ] ], dtype = float).flatten()
    self.assertTrue(np.allclose(expected_b_ub, solver.b_ub))
      
  def test_solve(self):
    solver = create_solver_with_default_values(nb_countries = 3, nb_time_steps = 4)
    solution, solver_results = solver.solve()
    self.assertSequenceEqual(solution.shape,(3, 3, 4))
    print(f'Solver returned: min={solver_results.fun}, status={solver_results.status}, nb_it={solver_results.nit}, message=\'{solver_results.message}\'')
  
  def test_solve_sparse(self):
    solver = create_solver_with_default_values(nb_countries = 3, nb_time_steps = 4, do_use_sparse = True)
    solution, solver_results = solver.solve()
    self.assertSequenceEqual(solution.shape,(3, 3, 4))
    print(f'Solver returned: min={solver_results.fun}, status={solver_results.status}, nb_it={solver_results.nit}, message=\'{solver_results.message}\'')







if __name__ == '__main__' :
  unittest.main()

