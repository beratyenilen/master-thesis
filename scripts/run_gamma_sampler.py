# Imports
from dwave.system import DWaveSampler, EmbeddingComposite, FixedEmbeddingComposite
from embedding_operations import load_embedding
from bqm_operations import *
import numpy as np
from util import *
import struct


# This script can be used to generate experimental data.
# Take care that this will consume a lot of QPU and CPU time.
def seed_numpy_with_urandom():
    # Get 4 bytes of random data
    random_bytes = os.urandom(4)
    # Convert bytes to an unsigned integer
    random_seed = struct.unpack('I', random_bytes)[0]
    # Seed NumPy's random number generator
    np.random.seed(random_seed)

if __name__ == '__main__':
  # main directory to use
  main_directory = '/Users/beratyenilen/Desktop/Thesis/CODE/scripts/1D_Ising_Chain'
  # samplers to use 
  #sampler_ids = ["Advantage_system5.4", "Advantage_system4.1", "Advantage_system6.3"]
  #regions = ['eu-central-1', 'na-west-1', 'na-west-1']
  sampler_ids = ["Advantage_system5.4"]
  regions = ['eu-central-1']
  # Annealing time to use
  AT = 5
  # number of reads for each problem instance
  num_reads = 400
  # number of problem instances to use
  seeds = range(10)
  #seeds = [0]
  # the number of times each problem will be run
  # this controls the randomness of gamma errors! runseeds=1 or None for 'quantization_error'
  runseeds = 5
  #runseeds = None
  num_instances = len(seeds)
  # precision values to use for quantization
  precisions = [3,4,5]
  quantization_error, random_error, analog_control_error = 'quantization_error', 'random_error', 'analog_control_error'
  # Error model to use. Available models are 'quantization_error', 'random_error', 'analog_control_error'
  error_model = analog_control_error
  # this is only relevant for Analog errors. It controls the standard deviation of errors
  error_magnitude = None
  # Lengths of the 1D Ising spin chains
  Ls = [60]
  # if True, direct problems will run
  run_direct_problems = False
  # if True triangle copied problems will run
  run_triangle_copies = True
  # gamma parameter refers to the coupling between the same logical qubits
  # in different copies. They set the range of values to sample from.
  gamma_lower_limit, gamma_upper_limit, max_num_gamma_points = -0.3, 0.3, 10
  # Run the problems and save 
  print("Sampling results for:")
  print(f"Ls={Ls}")
  print(f"sampler_ids = {sampler_ids}")
  print(f"precisions = {precisions}, error_model={error_model}, run_direct_problems={run_direct_problems}, run_triangle_copies={run_triangle_copies}")
  print(f"num_reads={num_reads}, seeds={seeds}, num_instances={num_instances}, Gamma range=[{gamma_lower_limit, gamma_upper_limit}]")

  while runseeds > 0 or runseeds == None:
    if runseeds != None:
      seed_numpy_with_urandom()
      
      print()
      print(f"{'-'*30} runseeds = {runseeds} {'-'*30}")
      runseeds -= 1
    else:
      # this means we'll run the code only once
      runseeds = 0

    for sampler_id, region in zip(sampler_ids, regions):
      # First get the sampler
      try:
        sampler = DWaveSampler(solver=sampler_id, region=region)
      except Exception as e:
        print(f"Coudln't access the sampler. The following Exception occurred.")
        print(e)
      # If the chip_id is not equal to solver_id then we should abort
      if sampler.properties['chip_id'] != sampler_id:
        print(sampler.properties['chip_id'], sampler_id)
        raise Exception(f"chip_id and sampler_id don't match. chip_id={sampler.properties['chip_id']}, sampler_id={sampler_id}")
      
      print(f"{'-'*30} Working with sampler_id = {sampler_id} {'-'*30}")
      for L in Ls:
        print(f"{' '*10} L={L}") 
        # load the embeddings
        if run_direct_problems:
          direct_emb = load_embedding(L, solver_id=sampler_id, emb_type="direct", main_directory=main_directory,tiled=False)
          direct_sampler = FixedEmbeddingComposite(sampler, direct_emb)
        if run_triangle_copies:
          tri_emb = load_embedding(L, solver_id=sampler_id, emb_type="triangle_direct", main_directory=main_directory, tiled=False)
          tri_sampler = FixedEmbeddingComposite(sampler, tri_emb)
    
        for precision in precisions:     
          print(f"\t\tprecision={precision}")
      
          if error_model in  ['quantization_error', 'random_error']:
            # might as well introduce quantization errors directly to the available
            # gamma values. This way I don't need to introduce errors everytime after 
            # update gamma
            if precision <= 5:
              available_j_values = np.round(np.linspace(-1, 1, 2**precision+1), 3)
              gamma_range = [x for x in available_j_values if x <= gamma_upper_limit and x >= gamma_lower_limit]
            else:
              # to save computational resources we'll trim the data points since
              available_j_values = np.round(np.linspace(-1, 1, 2**5+1), 3)
              gamma_range = [x for x in available_j_values if x <= gamma_upper_limit and x >= gamma_lower_limit]

          elif error_model == 'analog_control_error':
            error_magnitude = get_sigma_from_precision(precision)
            gamma_range = np.round(np.linspace(gamma_lower_limit, gamma_upper_limit, max_num_gamma_points), 3)
            # we should have 0 in the gamma_range
            if 0.0 not in gamma_range:
              gamma_range = np.append(gamma_range, 0.0)
              gamma_range.sort()

          # load the problem instances
          problem_instances, _ = load_problems(L=L, seeds=seeds, main_directory=main_directory)
          # run the problems
          for seed, problem_instance in zip(seeds,problem_instances):
            org_bqm, gs, gs_energy = problem_instance
            if run_direct_problems:
              # tile the original bqm
              org_bqm_w_error = introduce_precision_errors(org_bqm,
                                                            error_model=error_model,
                                                            error_magnitude=error_magnitude,
                                                            precision=precision,
                                                            verbose=False)
              
              label = f"QAC:direct:{error_model}:L= {L}:precision={precision}:AT={AT}"
              direct_results = direct_sampler.sample(org_bqm_w_error, annealing_time=AT, num_reads=num_reads)
              fname = get_fname_for_results(L=L, seed=seed, error_model=error_model, gamma=None,
                                              precision=precision, problem_type='direct', solver_id=sampler_id,
                                              AT=AT, main_directory=main_directory, tiled=False)
                
              save_results_w_fname(direct_results, fname, as_pandas=True)
      
            if run_triangle_copies:
              # construct the tiled and copied BQM
              triangle_bqm = copy_BQM(bqm=org_bqm,  gamma=0, num_copies=3, copy_type='triangle')

              triangle_bqm_w_error = introduce_precision_errors(bqm=triangle_bqm,
                                                        error_model=error_model,
                                                        error_magnitude=error_magnitude,
                                                        precision=precision,
                                                        verbose=False)
              tri_results = []
              for gamma in gamma_range:
                if error_model == 'quantization_error':
                  gamma_error = 0
                elif error_model == 'random_error':
                  # We should add a perturbation to the gamma value from a uniform distribution
                  gamma_error = round(np.random.uniform(-2**(-precision), 2**(-precision) ), 3)
                  gamma += gamma_error
                elif error_model == 'analog_control_error':
                  gamma_error = round(np.random.normal(0, scale=error_magnitude), 3)
                  # we should add a perturbation to the gamma value from a normal distribution
                  gamma += gamma_error
                
                # sample and savedirect triangle embedding results
                label = f"QAC:Triangle:{error_model}:L={L}:seed={seed}:gamma={round(gamma,5)}:AT={AT}"
                # update the gamma value
                triangle_bqm_w_error = update_gamma(triangle_bqm_w_error, gamma)
                # sample results
                res = tri_sampler.sample(triangle_bqm_w_error, annealing_time=AT, num_reads=num_reads, label=label)
                fname = get_fname_for_results(L=L, seed=seed, error_model=error_model, 
                                              precision=precision, problem_type='triangle_direct', solver_id=sampler_id,
                                              AT=AT, gamma=(gamma-gamma_error), main_directory=main_directory, tiled=False)
                
                save_results_w_fname(res, fname, as_pandas=True)
    