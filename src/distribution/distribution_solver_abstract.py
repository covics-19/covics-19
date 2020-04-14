

def check_shape(array_to_check, target_shape):
  if(array_to_check.shape != target_shape):
    raise Exception("Init DistributionSolver params(wrong shape)")


