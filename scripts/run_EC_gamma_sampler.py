# Imports
from dwave.system import DWaveSampler, EmbeddingComposite, FixedEmbeddingComposite
from embedding_operations import load_embedding
from bqm_operations import *
import numpy as np
from util import *
import struct
from embedding_operations import *






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
  main_directory = "/Users/beratyenilen/Desktop/Thesis/CODE/exact-cover-generator-main/Exact_Cover"
  # samplers to use 
  #sampler_ids = ["Advantage_system5.4", "Advantage_system4.1", "Advantage_system6.3"]
  #regions = ['eu-central-1', 'na-west-1', 'na-west-1']
  sampler_id = "Advantage_system5.4"
  region = 'eu-central-1'
  # Annealing time to use
  AT = 2
  # number of reads for each problem instance
  num_reads = 5000
  # number of problem instances to use
  seeds = range(51,100)
  #seeds = [1]
  #NTSs = [6,12,18,24,30]  # number of total sets
  #NSSs = [2,2,3,4,5]      # number of solution sets
  NTSs = [24]
  NSSs = [4]

  num_instances = len(seeds)

  # if True, direct problems will run
  run_direct_problems = True
  # if True triangle copied problems will run
  run_triangle_copies = True
  # gamma parameter refers to the coupling between the same logical qubits
  # in different copies. They set the range of values to sample from.
  gamma_range = [-0.5, -0.4, -0.3, -0.2, -0.1, 0. , 0.1,  0.2,  0.3,  0.4,  0.5]
  #gamma_range = [0]
  # Run the problems and save 
  print("Sampling results for:")
  print(f"NTSs,NSSs={NTSs, NSSs} AT={AT}")
  print(f"sampler_id = {sampler_id}")
  print(f"run_direct_problems={run_direct_problems}, run_triangle_copies={run_triangle_copies}")
  print(f"num_reads={num_reads}, seeds={seeds}, num_instances={num_instances}, Gamma range={gamma_range}")

  # get the sampler
  try:
      sampler = DWaveSampler(solver=sampler_id, region=region)
  except Exception as e:
    print(f"Coudln't access the sampler. The following Exception occurred.")
    print(e)
  # If the chip_id is not equal to solver_id then we should abort
  if sampler.properties['chip_id'] != sampler_id:
    print(sampler.properties['chip_id'], sampler_id)
    raise Exception(f"chip_id and sampler_id don't match. chip_id={sampler.properties['chip_id']}, sampler_id={sampler_id}")
  
  sampler_graph = sampler.to_networkx_graph()
  
  print("--------------- LOADING BQMs and embeddings --------------------")
  BQMs = {}
  gs_energies = {}
  # tiled direct embeddings
  direct_embs = {}
  tri_embs = {}

  for NTS, NSS in zip(NTSs,NSSs):
    BQMs[(NTS,NSS)] = {}
    gs_energies[(NTS,NSS)] = {}
    direct_embs[(NTS,NSS)] = {}
    tri_embs[(NTS,NSS)] = {}

    for seed in seeds:
      # get the problems
      problem_path = f"{main_directory}/problems/NTS|{NTS}_NSS|{NSS}_seed|{seed}.txt"
      bqm, gs, gs_energy = create_bqm_from_qubo(problem_path)
      # get direct embedding
      tiled_direct_emb_path = f"{main_directory}/embeddings/{sampler_id}/NTS|{NTS}_NSS|{NSS}_seed|{seed}_direct-tiled.json"
      with open(tiled_direct_emb_path, "r") as f:
        tiled_direct_emb = json.load(f)

      tiled_bqm = copy_BQM(bqm, gamma=0, copy_type='unconnected')
      if check_embedding(tiled_bqm, tiled_direct_emb, sampler_graph):
        direct_embs[(NTS,NSS)][seed] = tiled_direct_emb
      else:
        print(f"Invalid tiled embedding for NTS,NSS={NTS,NSS} seed={seed}")

      tri_emb_path =  f"{main_directory}/embeddings/{sampler_id}/NTS|{NTS}_NSS|{NSS}_seed|{seed}_triangle.json"
      with open(tri_emb_path, "r") as f:
        tri_emb = json.load(f)

      tri_bqm = copy_BQM(bqm, gamma=0)
      if check_embedding(tri_bqm, tri_emb, sampler_graph ):
        tri_embs[(NTS,NSS)][seed] = tri_emb
      else:
        print(f"Invalid triangle embedding for NTS,NSS={NTS,NSS} seed={seed}")

      BQMs[(NTS,NSS)][seed] = bqm
      gs_energies[(NTS,NSS)][seed] = gs_energy

  print(" ----------- BQMS and embedding loading finished ------------")
  print(" ----------- RUnning the problems -------------")
  seed_numpy_with_urandom()
  # let's run the problems and save them
  for NTS,NSS in zip(NTSs, NSSs):
    print(f"NTS,NSS = {NTS, NSS}")
    for seed in seeds:
      bqm = BQMs[(NTS,NSS)][seed]
      gs_energy = gs_energies[(NTS,NSS)][seed]
      if run_direct_problems:
        # get tiled direct embedding
        direct_emb = direct_embs[(NTS,NSS)][seed]
        # copy bqm in unconnected manner
        tiled_bqm = copy_BQM(bqm, gamma=0, num_copies=3, copy_type='unconnected')
        try:
          emb_sampler = FixedEmbeddingComposite(sampler, direct_emb)
        except : 
          print('PROBLEM')
          print(f"NTS={NTS}, NSS={NSS}, seed={seed}")
          # load the direct embedding
          direct_emb_path = f"{main_directory}/embeddings/{sampler_id}/NTS|{NTS}_NSS|{NSS}_seed|{seed}_direct.json"
          with open(direct_emb_path, "r") as f:
            direct_emb = json.load(f)
          # tile it
          tiled_emb = tile_minor_embedding(bqm, direct_emb, sampler_graph, max_num_tiles=3, sampler=sampler)
          tiled_direct_emb_path = f"{main_directory}/embeddings/{sampler_id}/NTS|{NTS}_NSS|{NSS}_seed|{seed}_direct-tiled.json"
          with open(tiled_direct_emb_path, "w") as f:
            json.dump(tiled_emb, f)
            
          emb_sampler = FixedEmbeddingComposite(sampler, direct_emb)

        direct_res = emb_sampler.sample(tiled_bqm, 
                                        annealing_time=AT,
                                        num_reads=num_reads)
        direct_fname = get_fname_for_results_EC(NTS, NSS, seed, problem_type='direct',
                                                solver_id=sampler_id, AT=AT, gamma=None,
                                                main_directory=main_directory)

        save_results_w_fname(direct_res, direct_fname)
      if run_triangle_copies:
        # get triangle embedding
        tri_emb = tri_embs[(NTS,NSS)][seed]
        emb_sampler = FixedEmbeddingComposite(sampler, tri_emb)

        max_strength = get_max_strength(bqm)
        tri_bqm = copy_BQM(bqm, gamma=0, num_copies=3, copy_type='triangle')

        for gamma in gamma_range:
          tri_bqm = update_gamma(tri_bqm, gamma=gamma*max_strength)
          tri_res = emb_sampler.sample(tri_bqm, 
                                      annealing_time=AT,
                                      num_reads=num_reads)
          
          tri_fname = get_fname_for_results_EC(NTS, NSS, seed, problem_type='triangle',
                                            solver_id=sampler_id, AT=AT, gamma=gamma,
                                            main_directory=main_directory)

          save_results_w_fname(tri_res, tri_fname)

    
