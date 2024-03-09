# Imports
import pickle
import os
import dimod
import numpy as np
import gurobipy
from dimod import ExactSolver



# This script generates 1D Ising Spin chain problem instances for a given set of parameters.

def generate_ising_chain(N: int, seed:int =10, sigma:float=1, precision: int=3, return_info: bool=False, use_gurobi: bool=False) -> tuple[dimod.BinaryQuadraticModel, dict, float]:
  '''
    Generates a 1D Ising chain, that is $J_{jk} \neq$ only when k=j+1. 
    h and J values are drawn independently from a Normal distribution with mean 0 and standard deviation sigma.

    Param:
      N (int)        : chain length
      seed (int)     : Seed for the RNG
      sigma (float)  : stdev for the Normal distribution
      precision (int): Number of decimal points for h,J values
      return_info (bool) : If True, ground state and ground state energy are also returned.
      use_gurobi (bool) : If True, Gurobi package is used to calculate the ground state and ground state energy. return_info is then not used!

    Returns:
      dimod.BinaryQuadraticModel for the problem [and if return_info = True] ground state and the ground state energy.

  '''
  bqm = dimod.BinaryQuadraticModel(vartype=dimod.SPIN)
  # let's seed the RNGs to get reproducable results
  np.random.seed(seed)
  
  # h,J range for Advantage machines
  h_min, h_max = -4,4
  J_min, J_max = -1,1

  if use_gurobi:
    # In this case variable names should start with a string value
    for i in range(N):
      bqm.add_variable(f"v{i}")
  
    # add the couplings for the neighbors
    for i in range(N-1):
      j = round(np.random.normal(scale=sigma), precision)
      if j < J_min:
        j = J_min
      if j > J_max:
        j = J_max
      bqm.add_quadratic(f"v{i}", f"v{i+1}", j)
    
    # add the local fields
    for i in range(N):
      h = round(np.random.normal(scale=sigma), precision)
      if h < h_min:
        h = h_min
      if h > h_max:
        h = h_max
      bqm.add_linear(f"v{i}", h)
    
    # transform BQM -> CQM
    cqm = dimod.ConstrainedQuadraticModel.from_quadratic_model(bqm)
    # .LP files don't accept Spin variables. Transform CQM variables to binary.
    cqm_binary = cqm.spin_to_binary()
    # We should Reset the objective offset for Gurobi
    cqm_binary.objective.offset = 0
    # Write the .LP file
    with open("./temp.lp", "w") as f:
      dimod.lp.dump(cqm_binary, f)

    # this will be one ground state solution, though the gs is not really important 
    # due to possible degeneracies.
    state = {}
    # We need to use such a flow to suppress the output
    with gurobipy.Env(empty=True) as env:
      env.setParam("OutputFlag", 0)
      env.start()
      # construct Gurobi model
      with gurobipy.read("temp.lp", env) as model:
        # solve the problem
        model.optimize()
        # Get the variables and their values
        vars = model.getVars()
        for var in vars:
          state[int(var.varName[1:])] = -1 + 2* var.X

          
    # Relabel BQM variables
    bqm.relabel_variables({f"v{i}": i for i in range(N)})
    # let's get the ground state energy as well
    energy = bqm.energy(state)
    
    return bqm, state, energy

  else:
    # first add the variables
    for i in range(N):
      bqm.add_variable(i)
    
    # add the couplings for the neighbors
    for i in range(N-1):
      j = round(np.random.normal(scale=sigma), precision)
      if j < J_min:
        j = J_min
      if j > J_max:
        j = J_max
      bqm.add_quadratic(i, i+1, j)
    
    # add the local fields
    for i in range(N):
      h = round(np.random.normal(scale=sigma), precision)
      if h < h_min:
        h = h_min
      if h > h_max:
        h = h_max
      bqm.add_linear(i, h)

    if return_info:
      # let's now calculate the ground state energy
      res = ExactSolver().sample(bqm).lowest()
      state = list(res.samples())[0]
      energy = res.record['energy'][0]

      return bqm, state, energy
    else:
      return bqm
    

if __name__ == "__main__":
  # Main directory to store the generated problems
  main_directory = "./1D_Ising_Chain/problems"
  # Number of instances to generate for each L
  num_instances = 100
  # Seeds to use in problem generation
  seeds = range(num_instances)
  # Lengths of the 1D Ising chains
  #Ls = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
  Ls = [3,4]
  # h and J values are drawn randomly from a Normal distribution with mean 0 and stdev sigma.
  sigma = 1
  # Generated floating point numbers will be rounded to the following decimal points
  precision = 3
  # If True Gurobi is used to calculate the ground states and ground state energies
  # Else dimod.ExactSolver is used. However after L=25, ExactSolver becomes really really slow.
  use_gurobi = True
  print(f"Generating 1D Ising Chain problems. They will be stored in {main_directory}.")
  print(f"Example filename={main_directory}/L-{5}_seed-{0}.pickle")
  print(f"Ls={Ls}\nseeds={seeds}\nnum_instances={num_instances}, sigma={sigma}, use_gurobi={use_gurobi}")

  # create the subdirectories if they don't exist
  for L in Ls:
    if not os.path.isdir(main_directory):
      os.makedirs(main_directory)

  for L in Ls:
    print(f"L={L}")
    for seed in seeds:
      # The function will return dimod.BinaryQuadraticModel, ground state (dict), ground state energy (float)
      res = generate_ising_chain(L, seed=seed, sigma=sigma, use_gurobi=True, return_info=True, precision=precision)
      with open(f"{main_directory}/L-{L}_seed-{seed}.pickle", "wb") as f:
        pickle.dump(res, f)


