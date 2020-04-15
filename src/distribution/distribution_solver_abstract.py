import numpy as np

def check_shape(array_to_check, target_shape):
  if(array_to_check.shape != target_shape):
    raise Exception("Init DistributionSolver params(wrong shape)")

def prepare_future_discount_coefficients(nb_time_steps, discount_factor, discount_coefficients):
  if(discount_factor is None):
    if(discount_coefficients is None):
      discount_coefficients = np.ones((nb_time_steps, ), dtype = float)
  else :
    if(discount_coefficients is not None):
      raise Exception("Set either a discount factor or a list of discount coefficients but not both.")
    discount_coefficients = np.geomspace(1., discount_factor **(nb_time_steps - 1), nb_time_steps)
  return discount_coefficients
