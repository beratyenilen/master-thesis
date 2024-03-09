# This file contains BQM related operations that can be used for the other scripts.
import dimod 
import numpy as np

def copy_BQM(bqm: dimod.BinaryQuadraticModel, gamma: float, num_copies:int=3, copy_type: str='triangle', invert_gamma: bool=True) -> dimod.BinaryQuadraticModel:
  '''
    Copies the BQM using a linear method of copying.
    Params:
      bqm (dimod.BinaryQuadraticModel): original BQM
      gamma (float)  : coupling strength between copies of the same logical qubit
      num_copies (int) : total number of copies to be made
      copy_type  (str) : It can be 'linear' or 'triangle'
      gamma_schedule (str) : If 'alternate', gamma alternates between +gamma and -gamma.
                             If 'straight', all couplings are +gamma
      invert_gamma (bool)  : If True gamma -> -gamma in the couplings.
    return
      copied BQM.
  '''
  allowed_copy_types = ['linear', 'triangle', 'unconnected']
  assert copy_type in allowed_copy_types, f"Allowed copy types are {allowed_copy_types}"
  
  if num_copies != 3 and copy_type == 'triangle':
    raise Exception("Hmmmm")
  
  if invert_gamma:
    gamma = -gamma
    
  copied_bqm = dimod.BinaryQuadraticModel(vartype=bqm.vartype)
  for copy in range(num_copies):
    for var in bqm.variables:
      # variable for the first copy
      copied_bqm.add_variable(f"{copy}_{var}")

  # We should add all the linear and quadratic terms as well
    for var, h in bqm.linear.items():
      copied_bqm.add_linear(f"{copy}_{var}",h)

    for var, j in bqm.quadratic.items():
      copied_bqm.add_quadratic(f"{copy}_{var[0]}", f"{copy}_{var[1]}", j)
      
  # finally we add the corresponding qubit interactions
  # so we have i_j <-> (i+1)_j for all variables j
  if copy_type == 'unconnected':
    pass
  else:
    for copy in range(num_copies-1):
      for var in bqm.variables:
        copied_bqm.add_quadratic(f"{copy}_{var}", f"{copy+1}_{var}", gamma)

    if copy_type == 'triangle':
      for var in bqm.variables:
        copied_bqm.add_quadratic(f"0_{var}", f"2_{var}", gamma)
  
  return copied_bqm


def tile_BQM(org_bqm: dimod.BinaryQuadraticModel, num_tiles:int, gamma: float=None, invert_gamma: bool=True, copy_bqm: bool=False) -> dimod.BinaryQuadraticModel:
  '''
    Given the original BQM, this function tiles the BQM with num_tiles many tiles. 
    If copy_bqm = True then the BQM is first copied in a triangle fashion and then tiled.
    Else, the tiling is performed on the original bqm. 

    Params:
      bqm (dimod.BinaryQuadraticModel): original BQM
      gamma (float)  : coupling strength between copies of the same logical qubit
      num_tiles (int) : total number of copies to be made
      invert_gamma (bool)  : If True gamma -> -gamma in the couplings.
      copy_bqm (bool) : If True then the given BQM is assumed to be the uncopied original BQM.
    return
      tiled BQM.
  '''
  assert (gamma == None and copy_bqm == False) or (gamma != None and copy_bqm == True), f"For copying a Gamma value should be provided"
    
  tiled_bqm = dimod.BinaryQuadraticModel(vartype=org_bqm.vartype)

  if not copy_bqm:
    for tile in range(num_tiles):
      for var in org_bqm.variables:
        # variable for the first copy
        tiled_bqm.add_variable(f"{tile}_0_{var}")

      # We should add all the linear and quadratic terms as well
      for var, h in org_bqm.linear.items():
        tiled_bqm.add_linear(f"{tile}_0_{var}",h)

      for var, j in org_bqm.quadratic.items():
        tiled_bqm.add_quadratic(f"{tile}_0_{var[0]}", f"{tile}_0_{var[1]}", j)

  else:
    # let's copy the BQM
    copied_bqm = copy_BQM(org_bqm, gamma=gamma, num_copies=3, copy_type='triangle', invert_gamma=invert_gamma)
    for tile in range(num_tiles):
        # we'll make triangle copies by default
      for copy_var in copied_bqm.variables:
        # add the variables
        tiled_bqm.add_variable(f"{tile}_{copy_var}")

      # We should add all the linear terms for each copy
      for copy_var,h  in copied_bqm.linear.items():
        tiled_bqm.add_linear(f"{tile}_{copy_var}", h)
      # Also add the quadratic terms 
      for copy_vars, J in copied_bqm.quadratic.items():
        copy_var1, copy_var2 = copy_vars
        tiled_bqm.add_quadratic(f"{tile}_{copy_var1}", f"{tile}_{copy_var2}", J)
      
  return tiled_bqm

def update_gamma(bqm: dimod.BinaryQuadraticModel, gamma: float, tiled:bool=False , invert_gamma:bool =True) -> dimod.BinaryQuadraticModel:
  '''
    Updates the Gamma, coupling strength between the corresponding qubits and returns the bqm.
    Assumes triangle copy.
  '''
  num_copies = 3
  if not tiled:
    # get the list of variables
    variables = set(v.split('_')[1] for v in bqm.variables)
    if invert_gamma:
      gamma = -gamma

    for var in variables:
      for copy in range(num_copies):  
        bqm.set_quadratic(f"{copy}_{var}", f"{(copy+1)%3}_{var}", gamma)
    return bqm
  else:
    # In this case the copies are simply trianguler
    variables = set(v.split('_')[-1] for v in bqm.variables)
    num_tiles = max(int(v.split('_')[0]) for v in bqm.variables) + 1
    
    if invert_gamma:
      gamma = -gamma

    for tile in range(num_tiles):
      for var in variables:
        for copy in range(num_copies):  
          bqm.set_quadratic(f"{tile}_{copy}_{var}", f"{tile}_{(copy+1)%3}_{var}", gamma)
    
    return bqm

def get_max_strength(bqm: dimod.BinaryQuadraticModel) -> float:
  return abs(max([max(bqm.linear.values(), key=abs), max(bqm.quadratic.values(), key=abs)],key=abs))

def introduce_precision_errors(bqm: dimod.BinaryQuadraticModel, error_model:str, precision:int=None, error_magnitude:float=None, verbose: bool=False) -> dimod.BinaryQuadraticModel:
  '''
    Changes h,J values of bqm according to the error_model and error_magnitude. 
    Param:
      bqm (dimod.BinaryQuadraticModel) : Binary quadratic model to introduce errors to
      error_model (str) : The following error schemes are implemented:
                          'quantization_error' : 
                          'random_error' : 
                          'analog_control_error' : 
      error_magnitude (float) : The magnitude of the introduced error.
      precision (int) : Precision used while generating the problem. If it is None, it will be derived from the BQM.
    Returns:
      bqm_with_precision_errors (dimod.BinaryQuadraticModel)
  '''
  assert precision != None, "precision=None is not implemented yet!"
  
  available_error_models = ['quantization_error', 'random_error', 'analog_control_error']

  assert error_model in available_error_models, f"Available error models are {available_error_models}"

  if error_model == 'quantization_error':
    pass
 

  bqm_w_errors = bqm.copy(deep=True)
  # introduce the errors on linear terms
  if error_model in ['quantization_error', 'random_error']:
    # in this case precision will denote the precision of the quantization
    h_intervals = np.round(np.linspace(-1, 1, 2**precision+1), 3)
    J_intervals = np.round(np.linspace(-1, 1, 2**precision+1), 3)

    # it may be the case that the minimum and maximum values for h are not -1, 1
    # but something smaller / greater. In that case, we will extend the h intervals
    # to include the minimum and maximum values.
    h_min = min( [min(bqm.linear.values()), -1] )
    h_max = max( [max(bqm.linear.values()), 1] )
    h_step_size = abs(h_intervals[1] - h_intervals[0])
    if h_min != -1:
      # extend the h intervals to include the minimum value
      current_val = -1 - h_step_size
      h_intervals = np.append(current_val, h_intervals)
      while current_val > h_min:
        current_val -= h_step_size
        h_intervals = np.append(current_val, h_intervals)
    
    if h_max != 1:
      # extend the h intervals to include the maximum value
      current_val = 1 + h_step_size
      h_intervals = np.append(h_intervals, current_val)
      while current_val < h_max:
        current_val += h_step_size
        h_intervals = np.append(h_intervals, current_val)
    
    if verbose:
      print("h_interval", h_intervals)
      print("J intervals", J_intervals)
    
    def get_closest(K, lst):
      #lst = np.asarray(lst)
      idx = (np.abs(lst - K)).argmin()
      return lst[idx]
    
    if error_model == 'quantization_error':
      # round the h terms
      for i, h_i in bqm.linear.items():
        bqm_w_errors.set_linear(i, get_closest(h_i, h_intervals))
      # round the J terms
      for ij, J_ij in bqm.quadratic.items():
        bqm_w_errors.set_quadratic(ij[0], ij[1], get_closest(J_ij, J_intervals))
    else:
      # then the error model is random_error
      # round the h terms
      for i, h_i in bqm.linear.items():
        # get the closest h value
        h_closest = get_closest(h_i, h_intervals)
        # get a random value between -2^-p, 2^p 
        h_error = round(np.random.uniform(-2**(-precision), 2**(-precision)), 3)
        if verbose:
          if h_error < 0:
            print(f"h={h_i}, h_w_error={h_closest}  {h_error} = {h_closest + h_error}")
          else:
            print(f"h={h_i}, h_w_error={h_closest} + {h_error} = {h_closest + h_error}")
        bqm_w_errors.set_linear(i, round(h_closest + h_error,3) )
      # round the J terms
      for ij, J_ij in bqm.quadratic.items():
        # get the closest J
        J_closest = get_closest(J_ij, J_intervals)
        # get a random value between -2^-p, 2^p
        J_error = round(np.random.uniform(-2**(-precision), 2**(-precision)), 3)
        if verbose:
          if J_error < 0:
            print(f"J{ij}={J_ij}, Jij_w_error={J_closest}  {J_error} = {J_closest + J_error}")
          else:
            print(f"J{ij}={J_ij}, Jij_w_error={J_closest} + {J_error} = {J_closest + J_error}")
            
        bqm_w_errors.set_quadratic(ij[0], ij[1], round(J_closest + J_error,3) )

  elif error_model == 'analog_control_error':
    if verbose:
      print("STDEV=", error_magnitude)
    for i, h_i in bqm.linear.items():
        # get the error term for the h value
        h_error = round(np.random.normal(loc=0, scale=error_magnitude), 3)
        if verbose:
          if h_error < 0:
            print(f"h={h_i}, h_w_error={h_i}  {h_error} = {h_i + h_error}")
          else:
            print(f"h={h_i}, h_w_error={h_i} + {h_error} = {h_i + h_error}")
        bqm_w_errors.set_linear(i, round(h_i + h_error,3) )
      # round the J terms
    for ij, J_ij in bqm.quadratic.items():
      # get the error value for the J
      J_error = round(np.random.normal(loc=0, scale=error_magnitude), 3)
      if verbose:
        if J_error < 0:
          print(f"J{ij}={J_ij}, Jij_w_error={J_ij}  {J_error} = {J_ij + J_error}")
        else:
          print(f"J{ij}={J_ij}, Jij_w_error={J_ij} + {J_error} = {J_ij + J_error}")
          
      bqm_w_errors.set_quadratic(ij[0], ij[1], round(J_ij + J_error,3) )

  if verbose:
    print("error_magnitude=",error_magnitude)
    print("h values Old value  -> New value")
    for h_old, h_new in zip(bqm.linear.values(), bqm_w_errors.linear.values()):
      print(h_old, "->", h_new)
    print("J values Old value  -> New value")  
    for j_old, j_new in zip(bqm.quadratic.values(), bqm_w_errors.quadratic.values()):
      print(j_old, "->", j_new)
    
  return bqm_w_errors