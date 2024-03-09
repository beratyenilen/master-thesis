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
  num_reads = 500
  # number of problem instances to use
  seeds = range(100)
  #seeds = range(50,100)
  # the number of times each problem will be run
  # this controls the randomness of gamma errors! runseeds=1 or None for 'quantization_error'
  runseeds = 1
  #runseeds = None
  num_instances = len(seeds)
  error_model = 'no_error'
  # Lengths of the 1D Ising spin chains
  #Ls = [3,4,5,10,15,20,25,30,40,50]
  Ls = [60,70,80,90,100,110,120,130,140,150]
  # if True triangle copied problems will run
  run_triangle_copies = True
  # gamma parameter refers to the coupling between the same logical qubits
  # in different copies. They set the range of values to sample from.
  gamma_lower_limit, gamma_upper_limit, max_num_gamma_points = -0.3, 0.3, 11
  # Run the problems and save 
  print("Sampling results for:")
  print(f"Ls={Ls}")
  print(f"sampler_ids = {sampler_ids}")
  print(f"error_model={error_model}, run_triangle_copies={run_triangle_copies}")
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
        if run_triangle_copies:
          tri_emb = load_embedding(L, solver_id=sampler_id, emb_type="triangle_direct", main_directory=main_directory, tiled=False)
          tri_sampler = FixedEmbeddingComposite(sampler, tri_emb)
    

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
           
            if run_triangle_copies:
              # construct the tiled and copied BQM
              triangle_bqm = copy_BQM(bqm=org_bqm,  gamma=0, num_copies=3, copy_type='triangle')

              tri_results = []
              for gamma in gamma_range:
                # sample and savedirect triangle embedding results
                label = f"QAC:Triangle:{error_model}:L={L}:seed={seed}:gamma={round(gamma,5)}:AT={AT}"
                # update the gamma value
                triangle_bqm = update_gamma(triangle_bqm, gamma)
                # sample results
                res = tri_sampler.sample(triangle_bqm, annealing_time=AT, num_reads=num_reads, label=label)
                fname = get_fname_for_results(L=L, seed=seed, error_model=error_model, 
                                              precision=None, problem_type='triangle_direct', solver_id=sampler_id,
                                              AT=AT, gamma=gamma, main_directory=main_directory, tiled=False)

                save_results_w_fname(res, fname, as_pandas=True)
    