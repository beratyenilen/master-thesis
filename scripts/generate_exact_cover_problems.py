# Imports
from util import *
from bqm_operations import *
from embedding_operations import *
import dimod
import pandas as pd
import matplotlib.pyplot as plt
import dwave_networkx as dnx
import networkx as nx
from minorminer import find_embedding
import scipy
import csv
import subprocess
import os
import math

def get_good_embedding(bqm_graph:nx.Graph, sampler_graph: nx.Graph, sampler, num_tries:int=10)->dict:
  '''
    generates num_tries many embeddings and returns the one with the minimum number of qubits and m
  '''
  
  embs = []
  for _ in range(num_tries):
    emb = find_embedding(bqm_graph, sampler_graph)
    try: 
        temp = FixedEmbeddingComposite(sampler, emb)
        # if this works out then it is a valid embedding
        embs.append(emb)
    except:
        # if that's the case this wasn't a valid embedding
        continue
  
  # calculate the normalization factors for each parameter
  max_num_qubits_used = 0 
  max_average_chain_length = 0
  max_std_chain_length = 0
  for emb in embs:
    chain_lengths = [len(chain) for chain in emb.values()]
    total_num_qubits = np.sum(chain_lengths)

    if total_num_qubits > max_num_qubits_used:
      max_num_qubits_used = total_num_qubits

    average_chain_length = np.mean(chain_lengths)
    if average_chain_length > max_average_chain_length:
      max_average_chain_length = average_chain_length

    std_chain_length = np.std(chain_lengths)
    if std_chain_length > max_std_chain_length:
      max_std_chain_length = std_chain_length
    
  # let's calculate each embeddings score now
  emb_scores = []
  for emb in embs:
    chain_lengths = [len(chain) for chain in emb.values()]
    total_num_qubits = np.sum(chain_lengths) / max_num_qubits_used
    average_chain_length = np.mean(chain_lengths) / max_average_chain_length
    std_chain_length = np.std(chain_lengths) / max_std_chain_length
    emb_scores.append(total_num_qubits+average_chain_length+std_chain_length)
  
  best_emb_index = np.argmin(emb_scores)
  #print(emb_scores[best_emb_index])
  return embs[best_emb_index]
  #return np.min(emb_scores)

def parse_qubo_file(file_path):
    """Parse the QUBO file and return the QUBO instance details."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    num_qubits = 0
    S = ""
    C = 0
    qubo_entries = []
    qubo_started = False
    for line in lines:
        if line.startswith("QUBO"):
            qubo_started = True
            continue
        if qubo_started:
            if line.startswith("QUBITS"):
                num_qubits = int(line.split(" ")[1].strip())
            elif line.startswith("S"):
                S = line.split(" ")[1].strip()
            elif line.startswith("C"):
                C = -float(line.split(" ")[1].strip())
            elif line.strip() and line[0].isdigit():
                parts = line.split()
                i, j, J_ij = int(parts[0]), int(parts[1]), float(parts[2])
                qubo_entries.append((i, j, J_ij))
        
    return num_qubits, S, C, qubo_entries

def create_bqm_from_qubo(file_path):
    num_qubits, S_str, C, qubo_entries = parse_qubo_file(file_path)
    bqm = dimod.BinaryQuadraticModel('BINARY')

    for i, j, J_ij in qubo_entries:
        if i == j:
            bqm.add_linear(i, J_ij)
        else:
            bqm.add_quadratic(i, j, J_ij)
    S_str = S_str[-1::-1]
    S = {var: int(S_str[var]) for var in range(num_qubits)}

    if bqm.energy(S) != C:
        print(f"Discrepancy in energy calculation: expected {C}, got {bqm.energy(S)}")

    return bqm, S, C

# Example usage:
# bqm, S, C = create_bqm_from_qubo("path_to_your_file.txt")
# print("BQM created, solution state S and ground state energy C extracted.")

def run_EC_generator(maincpp_directory: str, NUM_TOTAL_SETS: int, NUM_SOLUTION_SETS: int, seed:int=0, PROB_MIN: float = 0.2, PROB_MAX: float = 0.5, NUM_ELEMENTS:int=63, USE_CUSTOM_PROB_SET_BIT:bool =False):
    # Construct the path to main.cpp based on the provided directory
    main_cpp_path = os.path.join(maincpp_directory, "main.cpp")
    
    # Read the current content of main.cpp
    with open(main_cpp_path, "r") as file:
        lines = file.readlines()
    
    # Modify the lines where the variables are defined
    new_lines = []
    for line in lines:
        if "constexpr ULL SEED" in line:
            new_lines.append(f"constexpr ULL SEED = {seed};\n")
        elif "constexpr UL NUM_TOTAL_SETS" in line:
            new_lines.append(f"constexpr UL NUM_TOTAL_SETS = {NUM_TOTAL_SETS};\n")
        elif "constexpr UL NUM_SOLUTION_SETS" in line:
            new_lines.append(f"constexpr UL NUM_SOLUTION_SETS = {NUM_SOLUTION_SETS};\n")
        elif "constexpr UL NUM_ELEMENTS" in line:
            new_lines.append(f"constexpr UL NUM_ELEMENTS = {NUM_ELEMENTS};\n")
        elif "constexpr bool USE_CUSTOM_PROB_SET_BIT" in line:
            if USE_CUSTOM_PROB_SET_BIT:
                new_lines.append(f"constexpr bool USE_CUSTOM_PROB_SET_BIT = true;\n")
            else:
                new_lines.append(f"constexpr bool USE_CUSTOM_PROB_SET_BIT = false;\n")
        elif "constexpr double PROB_MIN" in line:
            new_lines.append(f"constexpr double PROB_MIN = {PROB_MIN};\n")
        elif "constexpr double PROB_MAX" in line:
            new_lines.append(f"constexpr double PROB_MAX  = {PROB_MAX};\n")
        else:
            new_lines.append(line)
    
    # Write the modified content back to main.cpp
    with open(main_cpp_path, "w") as file:
        file.writelines(new_lines)
    
    # Change to the directory where main.cpp is located
    os.chdir(maincpp_directory)
    
    # Define the output file path based on function parameters
    if USE_CUSTOM_PROB_SET_BIT:
        output_file_path = f"./Exact_Cover/NTS|{NUM_TOTAL_SETS}_NSS|{NUM_SOLUTION_SETS}_seed|{seed}_{PROB_MIN, PROB_MAX}.txt"
    else:
        output_file_path = f"./Exact_Cover/NTS|{NUM_TOTAL_SETS}_NSS|{NUM_SOLUTION_SETS}_seed|{seed}.txt"
    # Prepare output directory and file
    output_directory = f"./Exact_Cover"
    os.makedirs(output_directory, exist_ok=True)
    
    # Compile the modified main.cpp and capture output
    with open(output_file_path, 'w') as output_file:
        subprocess.run(['make'], stdout=output_file, stderr=subprocess.STDOUT)
    
    return output_file_path


if __name__ == "__main__":
  # Main directory to store the generated problems
  main_directory = "/Users/beratyenilen/Desktop/Thesis/CODE/exact-cover-generator-main/Exact_Cover"
  # Seeds to use in problem generation
  seeds = [range(51, 71), range(51,71)]
  # Number of instances to generate for each L
  num_instances = len(seeds)
  # number of tries for each embedding generation
  num_tries = 50
 
  # Lengths of the 1D Ising chains
  maincpp_dir = '/Users/beratyenilen/Desktop/Thesis/CODE/exact-cover-generator-main'

  NTSs = [24,30]

  min_NSS = 2
  fraction = 1/6
  # find embeddings for all the copied problem instancces
  sampler_id = 'Advantage_system5.4'
  sampler = DWaveSampler(solver=sampler_id, region='eu-central-1')

  if sampler.properties['chip_id'] != sampler_id:
    print(sampler.properties['chip_id'], sampler_id)
    raise Exception(f"chip_id and sampler_id don't match. chip_id={sampler.properties['chip_id']}, sampler_id={sampler_id}")
  
  sampler_graph = sampler.to_networkx_graph()
  # construct all the problems
  for NTS, sds in zip(NTSs, seeds):
    print(NTS)
    for seed in sds:
            # construct BQM
        problem_path = run_EC_generator(maincpp_directory=maincpp_dir, 
                                            NUM_TOTAL_SETS=NTS, 
                                            NUM_SOLUTION_SETS=max(math.ceil(NTS*fraction), min_NSS),
                                            seed=seed)
        
        bqm, gs, gs_energy = create_bqm_from_qubo(problem_path)
        # find an embedding
        direct_bqm_graph = dimod.to_networkx_graph(bqm)
        direct_emb = get_good_embedding(direct_bqm_graph, sampler_graph, num_tries=num_tries, sampler=sampler)
        # let's tile it
        tiled_direct_emb = tile_minor_embedding(bqm, direct_emb, sampler_graph, max_num_tiles=3)

        # copy BQM and find an embedding
        tri_copied_bqm = copy_BQM(bqm, gamma=0)
        tri_copied_bqm_graph = dimod.to_networkx_graph(tri_copied_bqm)
        tri_emb = get_good_embedding(tri_copied_bqm_graph, sampler_graph, num_tries=num_tries, sampler=sampler)

        # let's save the embeddings
        problem_dir  = '/'.join(problem_path.split('/')[:-1])
        problem_name = problem_path.split('/')[-1].split('.')[0]
        with open(f"{problem_dir}/{problem_name}_direct.json", 'w') as f:
            json.dump(direct_emb, f)

        with open(f"{problem_dir}/{problem_name}_direct-tiled.json", 'w') as f:
            json.dump(tiled_direct_emb, f)

        with open(f"{problem_dir}/{problem_name}_triangle.json", 'w') as f:
            json.dump(tri_emb, f)

