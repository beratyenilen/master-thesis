# Imports
from embedding_operations import *
from generate_problem_instances import generate_ising_chain
from bqm_operations import copy_BQM
from minorminer import find_embedding
import json
import os

# This script is used to generate direct embeddings for 1D Ising chain problems

if __name__ == '__main__':
  # main directory to store the embeddings
  main_directory = "./1D_Ising_Chain/embeddings"
  # Solvers to use. Regions should correspond to the solver_ids.
  sampler_ids = ["Advantage_system4.1", "Advantage_system6.3", "Advantage_system5.4"]
  regions = ["na-west-1", "na-west-1", "eu-central-1"]
  # Lengths of the spin chains
  #Ls =  [5,10,15,20,25,30,40,50,60,70,80,90,100,150,175,200]
  Ls = [3,4]
  # if True finds direct embeddings for 1D Ising chains
  find_direct_embeddings = False
  # if True finds direct embeddings for triangle copied 1D Ising chains
  find_triangle_direct_embeddings = True

  for L in Ls:
    print(f"L={L}")

    for region, sampler_id in zip(regions, sampler_ids):
      # First get the sampler we will be working with
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
      # this the graph we'll find an embedding for
      sampler_graph = sampler.to_networkx_graph()

    #Create the relevant directories
    for L in Ls:
      if not os.path.isdir(f"{main_directory}/{sampler_id}"):
        os.makedirs(f"{main_directory}/{sampler_id}")
        
    
      # We need an example problem instance and its graph
      bqm = generate_ising_chain(L)
      bqm_graph = dimod.to_networkx_graph(bqm)
      # we should copy it in a triangle fashion for testing the embedding algorithm
      copied_bqm = copy_BQM(bqm, gamma=0.3, num_copies=3, copy_type='triangle')
      copied_bqm_graph = dimod.to_networkx_graph(copied_bqm)
      
      if find_direct_embeddings:
        print("\tLooking for a direct embedding. Depending on the sampler and the problem size this may take a looooong time")
        while True:
          direct_emb = find_embedding(bqm_graph, sampler_graph)
          if isDirectEmbedding(direct_emb):
            with open(f"{main_directory}/{sampler_id}/direct_L-{L}.json", 'w') as f:
              json.dump(direct_emb, f)
            break

      if find_triangle_direct_embeddings:
        print("\tLooking for a direct embedding for triangle copies. !! Depending on the sampler and the problem size this may NOT be possible !!")
        # copied_triangle_direct    
        emb = triangle_embedding(bqm, target_graph=sampler_graph, return_graph=False)
        print("\t\t is it a valid embedding?", check_embedding(copied_bqm, emb))

        with open(f"{main_directory}/{sampler_id}/triangle_direct_L-{L}.json", 'w') as f:
          json.dump(emb, f)
        