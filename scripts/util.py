# Imports 
#from BQMRunner import *
import networkx 
import dimod
import dwave_networkx as dnx
import json
import glob
import pandas as pd
import pickle 
import numpy as np
import os 
import matplotlib.lines as mlines


# ------------ GRAPH DRAWING -------------------------------------------------------------------------------------------------------------------------------------
def get_colors(N: int) -> list:
  '''
    Returns N different colors that are rather easy to distinguish.
  '''
  from matplotlib import colormaps
  # All the relevant qualitative cmaps ordered according to the number
  # of colors they have.
  qualitative_cmaps = ['Dark2', 'Set2', 'Set1', 'tab10']
  for cmap_name in qualitative_cmaps:
    cmap = colormaps[cmap_name]
    if cmap.N >= N:
      # If the colormap has enough values we can return it
      return cmap.colors[:N]
  # if 10 < N <= 20 then we can use tab20
  if N <= 20:
    cmap = colormaps['tab20']
    # the colors are ordered like:
    # dark blue, light blue, dark pink, light pink, etc
    # So I will just reorder them
    even_colors = cmap.colors[::2]
    odd_colors  = cmap.colors[1::2]
    all_colors = even_colors + odd_colors
    return all_colors[:N]
  
  # If N > 20 then we'll return a list of rainbow colors
  cmap = colormaps['rainbow']
  return cmap(np.linspace(0,1,N))

def draw_graph(G: networkx.Graph, original: bool=False, node_size: int=1000, font_size: int=20, font_color: int= 'white', tiled:bool=False):
  '''
    Draws a given problem graph in a Circular fashion.
  '''
  if original:
    networkx.draw_circular(G, with_labels=True, node_size=node_size, font_size=font_size, font_color=font_color )
    return
  # I should first generate positions for the nodes
  # Copy-pasted the nx.circular_layout function here
  # I will modify it a bit
  center = np.array([0,0])
  paddims = 0

  if len(G) == 0:
    pos = {}
  elif len(G) == 1:
    pos = {networkx.utils.arbitrary_element(G): center}
  else:
    # Discard the extra angle since it matches 0 radians.
    theta = np.linspace(0, 1, len(G) + 1)[:-1] * 2 * np.pi
    #theta = theta.astype(np.float32)
    pos = np.column_stack(
        [np.cos(theta), np.sin(theta), np.zeros((len(G), paddims))]
    )
    # I can do the following part manually to ensure that
    # nodes -> position is ordered properly
    # Original one was
    # pos = dict(zip(G, pos))
    # Transform the nodes to a list
    nodes = list(G.nodes)
    # Sort them 
    nodes.sort()
    # Now do the zipping thingy
    pos = dict(zip(nodes, pos))

  # I can also maybe color the nodes but later 
  # check the last node in the nodes.
  if not tiled:
    num_copies, num_variables = nodes[-1].split('_')
    num_variables = int(num_variables) + 1
    num_copies = int(num_copies) + 1

    # get num_copies many colors 
    colors = get_colors(num_copies)
    # for _ in range(num_copies):
    #   colors += ["#" + ''.join([random.choice("abcdef0123456789") for _ in range(6)])]
    # In order for the nodes to be colored correctly, I must order it as G.nodes is ordered
    node_colors = []
    for node in G.nodes:
      # copy number
      c = int(node.split('_')[0])
      node_colors.append(colors[c])
    # Then I can simply call nx.draw_networkx 
    nx.draw_networkx(G, pos=pos, with_labels=True, node_color=node_colors, font_size=font_size, node_size=node_size, font_color=font_color)
  else:
    num_tiles, num_copies, num_variables = nodes[-1].split('_')
    num_variables = int(num_variables) + 1
    num_copies = int(num_copies) + 1
    num_tiles = int(num_tiles) + 1

    # we should have num_copies x num_tiles many colors
    colors = get_colors(num_copies)
    # In order for the nodes to be colored correctly, I must order it as G.nodes is ordered
    node_colors = []
    for node in G.nodes:
      # copy number
      c = int(node.split('_')[1])
      node_colors.append(colors[c])
    # Then I can simply call nx.draw_networkx 
    nx.draw_networkx(G, pos=pos, with_labels=True, node_color=node_colors, font_size=font_size, node_size=node_size, font_color=font_color)

def get_markers(N:int ) -> list:
  '''
    Returns N distinct markers
  '''
  available_markers = list(mlines.Line2D.markers.keys())
  available_markers.remove('.')
  available_markers.remove(',')
  # Cycle through the available markers if N is greater than the number of unique markers
  markers = [marker for marker, _ in zip(available_markers, range(N)) if marker not in ['.', ',']]

  return markers

def get_linestyles(N: int) -> list:
  '''
    Returns N distinc linstyles
  '''
  available_linestyles = mlines.lineStyles.keys()

  # Filter out the 'None' linestyle, as it's not visible
  available_linestyles = [ls for ls in available_linestyles if ls is not None]

  # Cycle through the available line styles if N is greater than the number of unique line styles
  linestyles = [linestyle for linestyle, _ in zip(available_linestyles, range(N))]
  return linestyles

# ---------------- GRAPH FUNCTIONS ---------------------------------------------------------------------------------
def get_min_xy(qubit_labels, coords, graph_type="chimera", add_one=False):
  if graph_type == "chimera":
    min_x = min_y = 999999
    max_x = max_y = -999999
    for qb in qubit_labels.keys():
      (y,x,u,k) = coords.linear_to_chimera(qb)
      # coordinates are ordered like (y,x,u,k)
      if y <= min_y:
        min_y = y
      if y >= max_y:
        max_y = y
      if x <= min_x:
        min_x = x
      if x >= max_x:
        max_x = x
    #return min_x, max_x, min_y, max_y
  elif graph_type == "pegasus":
    # Let's work with the 'nice' coordinates
    # (t,y,x,u,k) where t in [0,1,2], u in [0,1] k in [0,1,2,3]
    # so we only need to get y,x again
    min_x = min_y = 999999
    max_x = max_y = -99999
    for qb in qubit_labels.keys():
      (t,y,x,u,k) = coords.linear_to_nice(qb)
      if y <= min_y:
        min_y = y
      if y >= max_y:
        max_y = y
      if x <= min_x:
        min_x = x
      if x >= max_x:
        max_x = x

  elif graph_type == "pegasus_nice":
    min_x = min_y = 999999
    max_x = max_y = -99999
    for qb in qubit_labels.keys():
      # in this case node labels are nice coordinates
      (t,y,x,u,k) = qb
      if y <= min_y:
        min_y = y
      if y >= max_y:
        max_y = y
      if x <= min_x:
        min_x = x
      if x >= max_x:
        max_x = x
        
  if add_one:
    return max(min_x-1, -1), min(max_x+1, 15), max(min_y-1, -1), min(max_y+1, 15)
  else:
    return min_x, max_x, min_y, max_y

def remove_irrelevant_nodes(graph: networkx.Graph, embedding: dict, graph_type: str="chimera", only_emb_qubits: bool=False, relevant_nodes: list=None, include_neighbors: bool = False, m :int=16, n: int=16, t: int=4) -> networkx.Graph:
  '''
    Removes the non-intereacting, irrelevant nodes from the graph and returns a new one.
    Param:
      graph: (networkx.Graph) : The original graph
      embedding (dict)        : embedding for the problem
      graph_type (str)        : It can be 'chimera', 'pegasus', 'pegasus_tube'.
      only_emb_qubits (bool)  : If True, only the qubits in the embedding are kept.
      include_neighbors (bool): If True, all the nodes that have edges to the qubits in the embedding 
                                stay in the graph.
                                If False, only the nodes in the 'bulk' stay.
      m (int)   : parameter for constructing the relevant graph
      n (int)   : used for Chimera
      t (int)   : used in Chimera and Zephyr
  '''

  return_graph = graph.copy()
  qubit_labels = {}
  if relevant_nodes == None:
    relevant_nodes = []
  else:
    # then we are given a list of nodes to keep
    # lets construct a 'fake' embedding
    relevant_nodes = relevant_nodes.copy()
    embedding = {i: node for i, node in enumerate(relevant_nodes)}

  if include_neighbors:
    # In this case the relevant nodes are the ones used in the Embedding
    # and their neighbors
    for chain in embedding.values():
      relevant_nodes += chain
    # we'll simply remove all the nodes which don't have an edge
    for node in graph.nodes:
      has_edge_to_relevant_nodes = [return_graph.has_edge(node, i) for i in relevant_nodes]
      if True not in has_edge_to_relevant_nodes:
        return_graph.remove_node(node)
    # Now we can return the resulting graph
    return return_graph
  
  if only_emb_qubits:
    # In this case the relevant nodes are the ones used in the Embedding
    for chain in embedding.values():
      relevant_nodes += chain
    
    for node in graph.nodes:
      if node not in relevant_nodes:
        return_graph.remove_node(node)
    
    return return_graph

  # If we wish to keep only the nodes in the 'bulk' 
  # we should first get the qubit numbers
  for logical, chain in embedding.items():
    for physical in chain:
      qubit_labels[physical] = logical
  
  if graph_type == "chimera":
    chimera_coords = dnx.chimera_coordinates(m, n, t)
    min_x, max_x, min_y, max_y = get_min_xy(qubit_labels, chimera_coords, graph_type)
    # The nodes are coordinated like: (y,x,u,k) and the label
    # can be calculated
    # label = y*n*2*t + x*2*t + u*t + k
    # Default values are: n=m and t=4
    # for our case m = 16
    # u in {0,1} and k in {0,1,...,t-1} = {0,1,2,3}

    for x in range(min_x, max_x+1):
      for y in range(min_y, max_y+1):
        relevant_nodes += [y*n*2*t + x*2*t + u*t + k for u in [0,1] for k in range(t)]
      
    for node in graph.nodes:
      if node not in relevant_nodes :
        return_graph.remove_node(node)
  
  elif graph_type == "pegasus":
    pegasus_coords = dnx.pegasus_coordinates(m)
    min_x, max_x, min_y, max_y = get_min_xy(qubit_labels, pegasus_coords, "pegasus")

    for x in range(min_x, max_x+1):
      for y in range(min_y, max_y+1):
        relevant_nodes += [pegasus_coords.nice_to_linear((t,y,x,u,k)) for t in [0,1,2] for u in [0,1] for k in [0,1,2,3]]
      
    for node in graph.nodes:
      if node not in relevant_nodes:
        return_graph.remove_node(node)
  
  elif graph_type == "pegasus_nice":
    # this means pegasus graph was constructed using nice coordinates
    # whose labels are tuples of nice coordinates (t,y,x,u,k)
    pegasus_coords = dnx.pegasus_coordinates(m)
    min_x, max_x, min_y, max_y = get_min_xy(qubit_labels, pegasus_coords, graph_type)

    for x in range(min_x, max_x+1):
      for y in range(min_y, max_y+1):
        relevant_nodes += [(t,y,x,u,k) for t in [0,1,2] for u in [0,1] for k in [0,1,2,3]]
      
    for node in graph.nodes:
      if node not in relevant_nodes:
        return_graph.remove_node(node)

  elif graph_type == "zephyr":
    zephyr_coords = dnx.zephyr_coordinates(m, t)
    
  
  return return_graph

def keep_nodes(tyx_list, uk_list):
  '''
    Given a list of t,y,x values and a list of u,k values, The function returns a graph with
    only the specificed coordinates.
  '''
  pegasus_graph = dnx.pegasus_graph(16, nice_coordinates=True)
  all_nodes = list(pegasus_graph.nodes)
  relevant_nodes = []
  for uk_range,tyx in zip(uk_list,tyx_list):
    t,y,x = tyx
    if uk_range == 'all':
      uk_range = [(0,k) for k in range(4)] + [(1,k) for k in range(4)]
    elif uk_range == 'horizontal':
      uk_range =  [(0,k) for k in range(4)] 
    elif uk_range == 'vertical':
      uk_range =  [(1,k) for k in range(4)] 
    elif uk_range == 'inner':
      uk_range = [(0,1), (0,2), (1,2), (1,1)]
    elif uk_range == 'outer':
      uk_range = [(0,0), (1,3), (0,3), (1,0)]
    elif uk_range == "upper":
      uk_range = [(1,0), (1,1), (0,2), (0,3)]
    elif uk_range == "lower":
      uk_range = [(0,0), (0,1), (1,2), (1,3)]
    
    for u,k in uk_range:
      relevant_nodes.append((t,y,x,u,k))
  
  for node in all_nodes:
    if node not in relevant_nodes:
      pegasus_graph.remove_node(node)

  return pegasus_graph

# ---------------- RESULTS FUNCTIONS ---------------------------------------------------------------------------------


def decode_copied_BQM(results: list[dimod.SampleSet], org_bqm: dimod.BinaryQuadraticModel, num_copies:int) -> list[list[dimod.SampleSet]]:
  '''
    Returns a list with the results of copies as elements
    Param:
      results (dimod.SampleSet) : Results as a SampleSet
      org_bqm (dimod.BinaryQuadraticModel) : original binary quadratic model to calculate the energies
      num_copies (int) : number of copies used 
  '''
  if type(results) == dimod.SampleSet:
    results = [results]
  copy_results = []
  bqm_var_type = type(list(org_bqm.variables)[0]) 

  # Let's do this copy by copy
  for copy in range(num_copies):
    copy_results.append([])
    for result in results:
      # get the samples for this copy
      c_samples = []
      num_occurs = []
      for sample, num_occur in zip(result.samples(sorted_by=None), result.record['num_occurrences']):
        # I NEED TO CHECK ABOUT THIS SORTED_BY=NONE BUSINESS
        # It is correct!
        c_sample = {bqm_var_type(s.split('_')[1]): sample[s] for s in sample if f'{copy}_' in s}
        c_samples.append(c_sample)
        # get the num occur
        num_occurs.append(num_occur)

      # construct the SampleSet
      # we need the energies
      c_energies = org_bqm.energies(c_samples)
      # info for the run -> this is going to be the same all
      info = result.info
      c_result = dimod.SampleSet.from_samples(c_samples, org_bqm.vartype, c_energies, info, num_occurs)
      # append these 
      copy_results[copy].append(c_result)

  return copy_results

def decode_tiled_BQM(results: list[dimod.SampleSet], org_bqm: dimod.BinaryQuadraticModel, num_copies:int, num_tiles:int) -> dict[list[list[dimod.SampleSet]]]:
  '''
    Returns a dictionary of list with the results of copies as elements
    Param:
      results (dimod.SampleSet) : Results as a SampleSet
      org_bqm (dimod.BinaryQuadraticModel) : original binary quadratic model to calculate the energies
      num_copies (int) : number of copies used 
      num_tiles (int) : number of tiles used
  '''
  if type(results) == dimod.SampleSet:
    results = [results]
  tiled_results = {}  
  
  bqm_var_type = type(list(org_bqm.variables)[0]) 

  for tile in range(num_tiles):
    copy_results = []
    # Let's do this copy by copy
    for copy in range(num_copies):
      copy_results.append([])
      for result in results:
        # get the samples for this copy
        c_samples = []
        num_occurs = []
        for sample, num_occur in zip(result.samples(sorted_by=None), result.record['num_occurrences']):
          # It is correct!
          c_sample = {bqm_var_type(s.split('_')[-1]): sample[s] for s in sample if f'{tile}_{copy}_' in s}
          c_samples.append(c_sample)
          # get the num occur
          num_occurs.append(num_occur)

        # construct the SampleSet
        # we need the energies
        c_energies = org_bqm.energies(c_samples)
        # info for the run -> this is going to be the same all
        info = result.info
        c_result = dimod.SampleSet.from_samples(c_samples, org_bqm.vartype, c_energies, info, num_occurs)
        # append these 
        copy_results[copy].append(c_result)
    
      tiled_results[tile] = copy_results
  return tiled_results

def combine_fitnesses(copy_fits: dict, method: str) -> list:
  '''
    Given the fitnesses for the copies, returns the combined result
  '''
  if method in ["average", "min_energy", "min_energy_weighted", "average_hamming_distance", "TTS"] or "elite" in method:
    # in thes cases lower fitness is better
    combined_fitnesses = [9999999] * len(copy_fits[0])
  elif method in ["success_rate", "pvalue"]:
    # in these cases higher fitness is better
    combined_fitnesses = [-9999999] * len(copy_fits[0])
  else:
    raise Exception("Unkown method in combine_fitnesses")

  for c, fits in copy_fits.items():
    for i,cf in enumerate(fits):
      if method in ["average", "min_energy", "min_energy_weighted", "average_hamming_distance", "TTS"] or "elite" in method:
        # in thes cases lower fitness is better
        if cf <= combined_fitnesses[i]:
          combined_fitnesses[i] = cf
      elif method in ["success_rate", "pvalue"]:
        # in these cases higher fitness is better
        if cf >= combined_fitnesses[i]:
          combined_fitnesses[i] = cf

  return combined_fitnesses

def get_direct_success_rate(org_bqm: dimod.BinaryQuadraticModel, undecoded_results: list[dimod.SampleSet], min_energy: float, tolerance: int=5) -> dict[list[float]]:
  '''
  Given the undecoded results for an original problem instance, this function returns the direct success rate.
  Params:
    org_bqm (BinaryQuadraticModel)  : Original BQM
    undecoded_results ([SampleSet]) : List SampleSet for the original BQM.
    min_energy (float) : Ground state energy
    tolerance  (int)   : If the difference betweeen two energies is smaller than 10^(-tolerance) they are tread to be equal.
  Returns:
    success rate (float)
  '''
  assert min_energy != None, "min_energy should be provided"

  if type(undecoded_results) == dimod.SampleSet:
    pass
  elif type(undecoded_results) == list and len(undecoded_results) == 1:
    undecoded_results = undecoded_results[0]
  else:
    raise Exception("Expected only 1 sample set for this")

  vartype = type(org_bqm.variables[0])
  # get the samples for this tile
  samples = []
  num_occurs = []
  for sample, num_occur in undecoded_results.data(fields=['sample', 'num_occurrences']):
    samples.append(sample)
    # get the num occur
    num_occurs.append(num_occur)

  num_reads = sum(num_occurs)
  # we need only the energies for each tile
  energies = org_bqm.energies(samples)
  # let's count how many times the mininumm energy occurred
  sr = 0
  for energy, num_occur in zip(energies, num_occurs):
    if abs(energy-min_energy) < 10**(-tolerance):
      sr += num_occur / num_reads

  return sr

def get_tiled_success_rate(org_bqm: dimod.BinaryQuadraticModel, undecoded_results: list[dimod.SampleSet], min_energy: float, num_tiles: int, tolerance: int=5) -> dict[list[float]]:
  '''
  Given the undecoded results for a tiled original problem instance, this function returns the success rate accordin by tile number and the average over the whole chip.
  Params:
    org_bqm (BinaryQuadraticModel)  : Original BQM
    undecoded_results ([SampleSet]) : List SampleSet for the tiled original BQM.
    min_energy (float) : Ground state energy
    num_tiles (int) : num tiles used
    tolerance  (int)   : If the difference betweeen two energies is smaller than 10^(-tolerance) they are tread to be equal.
  Returns:
    success rate (dict[float])
  '''
  assert min_energy != None, "min_energy should be provided"
  as_pandas = False
  
  if type(undecoded_results) == dimod.SampleSet:
    pass
  elif type(undecoded_results) == pd.DataFrame:
    as_pandas = True
  elif type(undecoded_results) == list and len(undecoded_results) == 1:
    undecoded_results = undecoded_results[0]
    if type(undecoded_results) == pd.DataFrame:
      as_pandas = True
  else:
    raise Exception("Expected only 1 sample set for this")
  

  # tile2sr[tile] = success rate for tile
  # tile2sr['average'] = average successs rate over all the tiles
  tile2sr = {}

  vartype = type(org_bqm.variables[0])
  L = len(org_bqm.variables)

  if num_tiles < 10:
    str_index = 2
  elif num_tiles < 100:
    str_index = 3
  else:
    str_index = 4

  for tile in range(num_tiles):
    # get the samples for this tile
    samples = []
    num_occurs = []
    if as_pandas:
      for _, row in undecoded_results.iterrows():
        # construct the sample
        sample = {vartype(var):row[f"{tile}_0_{var}"]  for var in range(L)}
        # get the num occurrences
        num_occurs.append(row['num_occurrences'])
        samples.append(sample)
    else:
      for sample, num_occur in undecoded_results.data(fields=['sample', 'num_occurrences']):
        sample = {vartype(s.split("_")[-1]): sample[s] for s in sample if f'{tile}_' in s[:str_index]}
        samples.append(sample)
        # get the num occur
        num_occurs.append(num_occur)

    num_reads = sum(num_occurs)
    # we need only the energies for each tile
    energies = org_bqm.energies(samples)
    # let's count how many times the mininumm energy occurred
    sr = 0
    for energy, num_occur in zip(energies, num_occurs):
      if abs(energy-min_energy) < 10**(-tolerance):
        sr += num_occur / num_reads
    # append the success rate 
    tile2sr[tile] = sr

  # add the average over all the tiles
  average_sr = 0
  for tile_index in range(num_tiles):
    average_sr += tile2sr[tile_index]

  tile2sr['average'] =  average_sr / num_tiles

  return tile2sr

def get_combined_success_rate(org_bqm: dimod.BinaryQuadraticModel, undecoded_results: list[dimod.SampleSet], method: str='AL-1', min_energy: float = None, num_copies:int=3, tolerance: int=5) -> list:
  '''
  Given the undecoded results for the copies, this function returns combined success rate according to the method.
  Params:
    org_bqm (BinaryQuadraticModel)  : Original BQM
    undecoded_results ([SampleSet]) : List SampleSet for the copied_BQM. 
    method (str)  : AL-1 -> at least one. If one copy is correct in one of the copies then it is counted as a success.
                    AL-2 -> at least two copies need to be correct to be counted as a success.
                    AL-3 -> at least three copies need to be correct to be counted as a succes.
                    TOTAL -> total num correct states / num_reads / num_copies
                    ALL -> returns AL-1, AL-2, AL-3, TOTAL

    min_energy (float) : Ground state energy
    num_copies (int)   : Number of copies
    tolerance  (int)   : If the difference betweeen two energies is smaller than 10^(-tolerance) they are tread to be equal.
  Returns:
    combined_success_rate (list[float])
  '''
  assert min_energy != None, "min_energy should be provided"
  
  allowed_methods = ['AL-1', "AL-2", "AL-3", "TOTAL", "ALL"]
  method = method.upper()
  assert method in allowed_methods, f"Allowed methods are {allowed_methods}"
  if method in ['AL-1', 'AL-2', 'AL-3']:
    gs_counter_threshold = int(method[-1])
  else:
    pass

  if type(undecoded_results) == dimod.SampleSet:
    undecoded_results = [undecoded_results]

  bqm_var_type = type(list(org_bqm.variables)[0]) 
  variables = org_bqm.variables

  if method != 'ALL':
    combined_success_rate = []
    for i, result in enumerate(undecoded_results):
      combined_success_rate.append(0)
      num_reads = sum(result.record['num_occurrences'])
      # for each sample in this result
      for sample, num_occur in result.data(fields=['sample', 'num_occurrences']):
        #print(sample)
        # count the number of ground states in the copies
        gs_counter = 0
        for copy in range(num_copies):
          #c_sample = {bqm_var_type(s.split('_')[1]): sample[s] for s in sample if f'{copy}_' in s}
          c_sample = {var: sample[f"{copy}_{var}"] for var in variables}
          #print("\t", c_sample)
          if min_energy != None and abs(org_bqm.energy(c_sample)-min_energy) <= 10**(-tolerance):
            #combined_success_rate[i] += num_occur
            #break
            #print('\t \t Correct energy')
            gs_counter += 1
        # if the number of ground states for this sample is greater than or equal to 
        # the threshold, count the sample as a succes.
        if method != 'TOTAL' and gs_counter >= gs_counter_threshold:
          #print(f"\t gs_counter={gs_counter} >= {gs_counter_threshold}")
          combined_success_rate[i] += num_occur / num_reads
        elif method == 'TOTAL':
          #print(f"\t num_occur={num_occur}, gs_counter={gs_counter}")
          combined_success_rate[i] += num_occur * gs_counter / num_copies 
      # divide by the num_reads
      #combined_success_rate[i] /= num_reads

    return combined_success_rate
  else:
    csr_al1 = []
    csr_al2 = []
    csr_al3 = []
    csr_total = []
    for i, result in enumerate(undecoded_results):
      csr_al1.append(0)
      csr_al2.append(0)
      csr_al3.append(0)
      csr_total.append(0)
      # for each sample in this result
      num_reads = sum(result.record['num_occurrences'])
      for sample, num_occur in result.data(fields=['sample', 'num_occurrences']):
        # count the number of ground states in the copies
        gs_counter = 0
        for copy in range(num_copies):
          c_sample = {bqm_var_type(s.split('_')[1]): sample[s] for s in sample if f'{copy}_' in s}
          if min_energy != None and abs(org_bqm.energy(c_sample)-min_energy) <= 10**(-tolerance):
            gs_counter += 1
        # if the number of ground states for this sample is greater than or equal to 
        # the threshold, count the sample as a succes.
        if gs_counter == 1:
          csr_al1[i] += (num_occur / num_reads)
        elif gs_counter == 2:
          csr_al1[i] += (num_occur / num_reads)
          csr_al2[i] += (num_occur / num_reads)
        elif gs_counter >= 3:
          csr_al1[i] += (num_occur / num_reads)
          csr_al2[i] += (num_occur / num_reads)
          csr_al3[i] += (num_occur / num_reads)

        csr_total[i] += num_occur * gs_counter / num_copies / num_reads

    return csr_al1, csr_al2, csr_al3, csr_total

def get_combined_success_rate_tiled(org_bqm: dimod.BinaryQuadraticModel, undecoded_results: list[dimod.SampleSet], min_energy: float, num_tiles: int, L:int, method: str='AL-1', num_copies:int=3, tolerance: int=5) -> dict[list[float]]:
  '''
  Given the undecoded results for a tiled problem instance, this function returns combined success rate according to the method as dictionary keyed by tile number and also the
  average over the whole chip.
  It assumes triangle copies within a tile.
  Params:
    org_bqm (BinaryQuadraticModel)  : Original BQM
    undecoded_results ([SampleSet]) : List SampleSet for the copied_BQM. 
    min_energy (float) : Ground state energy
    num_tiles (int) : num tiles used
    method (str)  : AL-1 -> at least one. If one copy is correct in one of the copies then it is counted as a success.
                    AL-2 -> at least two copies need to be correct to be counted as a success.
                    AL-3 -> at least three copies need to be correct to be counted as a succes.
                    TOTAL -> total num correct states / num_reads / num_copies
                    ALL -> returns AL-1, AL-2, AL-3, TOTAL
    tolerance  (int)   : If the difference betweeen two energies is smaller than 10^(-tolerance) they are tread to be equal.
  Returns:
    combined_success_rate (dict[list[float]])
  '''
  assert min_energy != None, "min_energy should be provided"
  as_pandas = False
  allowed_methods = ['AL-1', "AL-2", "AL-3"]
  method = method.upper()
  assert method in allowed_methods, f"Allowed methods are {allowed_methods}"

  if method in ['AL-1', 'AL-2', 'AL-3']:
    gs_counter_threshold = int(method[-1])
  else:
    pass

  if type(undecoded_results) == dimod.SampleSet or type(undecoded_results) == pd.DataFrame:
    undecoded_results = [undecoded_results]

  if type(undecoded_results[0]) != dimod.SampleSet:
    as_pandas = True

  variables = org_bqm.variables
  min_energies = np.array([min_energy, min_energy, min_energy])
  bqm_vartype = type(variables[0])

  tile2metric = {}
  for tile in range(num_tiles):
    tile2metric[tile] = np.zeros(len(undecoded_results))

  tile2metric['average'] = np.zeros(len(undecoded_results))
  # to store previously calculated energies
  # 111-1-1-11 -> energy
  calculated_energies = {}

  for res_index, result in enumerate(undecoded_results):
    # for each result
    if as_pandas:
      num_reads = sum(result['num_occurrences'])
      to_iterate = result.iterrows()
    else:
      num_reads = sum(result.record['num_occurrences'])
      to_iterate = result.data(fields=['num_occurrences', 'sample'])
      
    for num_occur, sample in to_iterate:
      if as_pandas:
        # this means num_occur is simply the row index
        # and sample is actually the row of data
        num_occur = sample['num_occurrences']
      # {'0_0_0':-1, '0_0_1':1, ....}
      # look at all the samples
      for tile in range(num_tiles):
        # For this tile get the samples for each copy
        c1_sample = {var: sample[f"{tile}_{0}_{var}"] for var in variables}
        c2_sample = {var: sample[f"{tile}_{1}_{var}"] for var in variables}
        c3_sample = {var: sample[f"{tile}_{2}_{var}"] for var in variables}
        # check if the energies for these samples exist
        # 1-11-1-1 -> 
        s1 = "".join([str(c1_sample[bqm_vartype(i)]) for i in range(L)])
        s2 = "".join([str(c2_sample[bqm_vartype(i)]) for i in range(L)])
        s3 = "".join([str(c3_sample[bqm_vartype(i)]) for i in range(L)])

        energies = []
        # assume that we'll calculate all at first
        to_calculate = [1, 2, 3]

        if s3 in calculated_energies:
          to_calculate.pop()
          energies.append(calculated_energies[s3])
        if s2 in calculated_energies:
          to_calculate.pop()
          energies.append(calculated_energies[s2])
        if s1 in calculated_energies:
          to_calculate.pop()
          energies.append(calculated_energies[s1])
        
        if len(energies) != 3:
          if 1 in to_calculate:
            en = org_bqm.energy(c1_sample)
            energies.append(en)
            calculated_energies[s1] = en
          if 2 in to_calculate:
            en = org_bqm.energy(c2_sample)
            energies.append(en)
            calculated_energies[s2] = en
          if 3 in to_calculate:
            en = org_bqm.energy(c3_sample)
            energies.append(en)
            calculated_energies[s3] = en

        
        # and the difference between the Ground state of these energies
        is_gs = [True if abs(x) <= 10**(-tolerance) 
                 else False 
                 for x in  (energies - min_energies) ]
        
        if is_gs.count(True) >= gs_counter_threshold:
          tile2metric[tile][res_index] += num_occur / num_reads
 
  # calculate the average metric
  for res_index in range(len(undecoded_results)):
    for tile_index in range(num_tiles):
      tile2metric['average'][res_index] += tile2metric[tile_index][res_index] 
    tile2metric['average'][res_index] /= num_tiles
  
  return tile2metric
  #return tile2results

def get_fraction_correct(combined_success_rates: list[list[float]]) -> list[float]:
  '''
    Given a list of combined success rates for a set of problem instances, it simply returns the average.
    combined_success_rates[0] -> combined success rates for the first problem instance, etc
  '''
  if type(combined_success_rates[0]) == float:
    combined_success_rates = [combined_success_rates]

  fraction_corrects = np.zeros(len(combined_success_rates[0]))
  for i,combined_sr in enumerate(combined_success_rates):
    try:
      fraction_corrects += np.array(combined_sr)
    except:
      print(f"For index {i} the combined sr has length {len(combined_success_rates[i])}")
      
  
  return fraction_corrects / len(combined_success_rates)


# ---------------- PRINT FUNCTIONS ---------------------------------------------------------------------------------
def pretty_print_sample(sample: dict, num_copies: int, variables: list):
  '''
    Given a set of results, sample, prints the results grouped by copies.
  '''
  ss = ""
  for var in variables:
    ss += f"{var}= "
    for copy in range(num_copies):
      ss += str(sample[f"{copy}_{var}"]) + ", "

  print(ss)


# ---------------- LOAD/SAVE FUNCTIONS ---------------------------------------------------------------------------------
def load_problems(L: int, seeds:list[int]=None, num_instances: int=None, main_directory: str = "./1D_Ising_Chain") -> list[tuple[dimod.BinaryQuadraticModel, dict, float, int]]:
  '''
    Returns num_instances many problem instances stored in a list as (bqm, ground state, ground state energy) and the seeds as list. 
  '''
  if seeds == None and num_instances != None:
    seeds = range(num_instances)
  if seeds == None and num_instances == None:
    raise Exception("Either seeds or num_instances must be provided!")

  pickle_directories = []
  for i in seeds:
    try:
      pickle_directories.append(glob.glob(f"{main_directory}/problems/L-{L}_seed-{i}.pickle")[0])
    except:
      pass
  
  problem_instances = []
  counter = 0
  num_instances = len(seeds)
  for dir in pickle_directories:
    with open(dir, "rb") as f:
      problem_instances.append(pickle.load(f))
    counter += 1
    if counter == num_instances:
      break
  
  if len(problem_instances) < num_instances:
    print(f"Only {len(problem_instances)} many problems were available")
  
  return problem_instances, seeds

def load_problem(L: int, seed:int=None, main_directory: str = "./1D_Ising_Chain") -> tuple[dimod.BinaryQuadraticModel, dict, float]:
  '''
    Returns num_instances many problem instances stored in a list as (bqm, ground state, ground state energy) and the seeds as list. 
  '''

  pickle_directory = glob.glob(f"{main_directory}/problems/L-{L}_seed-{seed}.pickle")[0]

  with open(pickle_directory, "rb") as f:
    problem_instance = pickle.load(f)
  return problem_instance

def load_results(L:int, seed:int, problem_type: str, solver_id:str, at:float, gamma:float=None, rcs:float=None, precision:int=5, error_model: str=None, tiled:bool = False) -> dimod.SampleSet:
  '''
    Loads the results from , for example, ./1D_Ising_Chain/annealing_results/copied_triangle_direct/L-5_seed-0/gamma-0.5_AT-10.json. 
    Param:
      L (int) : length for the 1D Ising chain
      seed (int) : seed used for the problem, it defines the problem instance
      problem_type (str) : Can be 'copied_triangle_direct', 'copied_triangle_minor', 'uncopied_direct' 
      at (float) : Annealing time 
      gamma (float) : Defines the gamma value between the triangle copies
      rcs (float) : Relative chain strength used for the problem, necessary for 'copied_triangle_minor'
  '''
  allowed_problem_types = ['copied_triangle_direct', 'copied_triangle_minor', 'uncopied_direct'] 
  assert problem_type in allowed_problem_types, f"Allowed problem types are f{allowed_problem_types}"

  if problem_type != 'uncopied_direct' and gamma == None:
    raise Exception(f"For {problem_type} gamma value should be provided")  

  if problem_type == 'copied_triangle_minor' and rcs == None:
    raise Exception(f"For {problem_type} rcs value should be provided")
  
  at_float = float(at)
  # create filename
  if error_model == None:
    if tiled:
       problem_path = f"./1D_Ising_Chain/tiled_annealing_results/wNoError/{solver_id}/{problem_type}/L|{L}_seed|{seed}"
    else:
      problem_path = f"./1D_Ising_Chain/annealing_results/wNoError/{solver_id}/{problem_type}/L|{L}_seed|{seed}"

    if problem_type == 'copied_triangle_direct':
      fname1 =f"gamma|{round(gamma, precision)}_AT|{round(at,precision)}"
      fname2 = f"gamma|{round(gamma, precision)}_AT|{round(at_float,precision)}"
    elif problem_type == 'copied_triangle_minor':
      fname1 = f"gamma|{round(gamma,precision)}_AT|{round(at,precision)}_RCS|*"
      fname2 = f"gamma|{round(gamma,precision)}_AT|{round(at_float,precision)}_RCS|*"
    elif problem_type == "uncopied_direct":
      fname1 = f"AT|{round(at,precision)}"
      fname2 = f"AT|{round(at_float,precision)}"
    else:
      raise Exception(f"Invalid problem_type={problem_type} for L={L}, seed={seed}, at={at}, gamma={gamma}, rcs={rcs}, precision={precision}, error_model={error_model}, tiled={tiled}")
        
  elif error_model == 'quantization_error':
    if tiled:
       problem_path = f"./1D_Ising_Chain/tiled_annealing_results/{error_model}/{solver_id}/precision-{precision}/{problem_type}/L|{L}_seed|{seed}"
    else:
      problem_path = f"./1D_Ising_Chain/annealing_results/{error_model}/{solver_id}/precision-{precision}/{problem_type}/L|{L}_seed|{seed}"

    if problem_type == 'copied_triangle_direct':
      fname1 = f"gamma|{round(gamma, precision)}_AT|{round(at,precision)}"
      fname2 = f"gamma|{round(gamma, precision)}_AT|{round(at_float,precision)}"
    elif problem_type == 'copied_triangle_minor':
      fname1 = f"gamma|{round(gamma,precision)}_AT|{round(at,precision)}_RCS|*"
      fname2 = f"gamma|{round(gamma,precision)}_AT|{round(at_float,precision)}_RCS|*"
    elif problem_type == "uncopied_direct":
      fname1 = f"AT|{round(at,precision)}"
      fname2 = f"AT|{round(at_float,precision)}"
    else:
      raise Exception(f"Invalid problem_type={problem_type} for L={L}, seed={seed}, at={at}, gamma={gamma}, rcs={rcs}, precision={precision}, error_model={error_model}, tiled={tiled}")
      
  fname1 = problem_path + '/' + fname1
  fname2 = problem_path + '/' + fname2

  # check which one of these file names exits
  fname1dirs = glob.glob(f"{fname1}*") 
  fname2dirs = glob.glob(f"{fname2}*")
  if len(fname1dirs):
    fname = fname1dirs[0]
  elif len(fname2dirs):
    fname = fname2dirs[0]
  else:
    raise Exception(f"Couldn't find the results file for L={L}, seed={seed}, solver_id={solver_id}, problem_type={problem_type}, at={at}, gamma={gamma}, rcs={rcs}, error_model={error_model}, precision={precision}")
  # check if the results are pandas or SampleSet objects
  if '.csv' in fname:
    as_pandas = True
  elif '.json' in fname:
    as_pandas = False
  else:
    raise Exception(f"Unknown extension type for L={L}, seed={seed}, solver_id={solver_id}, problem_type={problem_type}, at={at}, gamma={gamma}, rcs={rcs}, error_model={error_model}, precision={precision}")
    
  # load the results
  if as_pandas:
    results = pd.read_csv(fname, low_memory=False)
    # let's check if there are weird column names
    res_columns = results.columns
    to_remove = []
    for col in res_columns:
      if 'Unnamed' in col:
        to_remove.append(col)

    for col in to_remove:
      results = results.drop([col], axis=1)
      
  else:
    with open(fname, 'r') as f:
      results = dimod.SampleSet.from_serializable(json.load(f))

  return results
  
def load_results_w_fname(fname:str) -> dimod.SampleSet:
  '''
    Given the fname and the main directory, loads the results and returns them.
    Param:
      fname (str) : filename
  '''
  # check if the results are pandas or SampleSet objects
  if '.csv' in fname:
    as_pandas = True
  elif '.json' in fname:
    as_pandas = False
  else:
    raise Exception(f"Unknown extension type")
    
  # load the results
  if as_pandas:
    with open(fname, 'r') as f:
      results = pd.read_csv(f, low_memory=False)
    # let's check if there are weird column names
    res_columns = results.columns
    to_remove = []
    for col in res_columns:
      if 'Unnamed' in col:
        to_remove.append(col)

    for col in to_remove:
      results = results.drop([col], axis=1)
      
  else:
    with open(fname, 'r') as f:
      results = dimod.SampleSet.from_serializable(json.load(f))

  return results
  
def save_results_w_fname(results: dimod.SampleSet, fname:str, as_pandas:bool=True ) -> None:
  # first let's create the directories required for the file
  path = '/'.join(fname.split('/')[:-1])

  if not os.path.isdir(path):
    os.makedirs(path)
        
  if as_pandas:
    to_store = results.to_pandas_dataframe()
    # first let's check if such a file exists
    if os.path.isfile(fname):
      with open(fname) as f:
        existing_results = pd.read_csv(f, low_memory=False)
      # Check if they have the same number of columns
      if len(existing_results.columns) != len(to_store.columns):
        raise ValueError("Number of columns in the existing file and new data does not match.")

      # to distinguish that we have a new data set, we only need this if the error model
      # is not quantization_error. since in this case they are all gonna be the same
      if 'quantization_error' in fname or 'Exact_Cover' in fname:
        pass
      else:
        new_row = pd.DataFrame([['NEW_DATA_SET'] * len(to_store.columns)], columns=to_store.columns)
        to_store = pd.concat([new_row, to_store]).reset_index(drop=True)
      # Concatenate the dataframes (ignoring the first row of new data)
      combined_results = pd.concat([existing_results, to_store], ignore_index=True)
      combined_results.to_csv(fname, index=False)
    else:
      # then we can directly store
      to_store.to_csv(fname, index=False)
  else:
    # if not we will create a new file
    # serialize the samplesets
    try:
      serialized_results = results.to_serializable()
    except:
      # sometimes we can't serialize the warnings and they are not
      # relevant for our purposes. So I will remove the warnings
      info = results.info.copy()
      info['warnings'] = []
      new_results = dimod.SampleSet.from_samples(results.samples(sorted_by='energy'), 
                                                    vartype=results.vartype,
                                                    energy=results.data_vectors['energy'],
                                                    num_occurrences=results.data_vectors['num_occurrences'],
                                                    info=info,
                                                    aggregate_samples=True
                                                  )
      # save the results 
      serialized_results = new_results.to_serializable()

    try:
      with open(fname, 'w') as f:
        json.dump(serialized_results, f)
    except Exception as e:
      print(f"The following exception occurred while trying to store results with fname={fname}")
      print(e)

def get_fname_for_results_EC(num_total_sets:int, num_solution_sets: int, seed:int, problem_type:str, solver_id:str, AT:float, gamma:float,  main_directory:str="./Exact_Cover", as_pandas:bool=True) -> str:
  '''
    Given the necessary parameters, returns the filename for the storing/loading annealing results.
    Files are named as:
    main_dir/annealing_results/problem_type/solver_id/NTS|12_fraction|0.6_seed|3/AT|0.5_gamma|0.txt
    if gamma = 'ALL' then a list of filenames are returned without filtering according to gamma.

   '''
  AT = round(float(AT), 2)
  get_all_gamma = False
  if gamma == 'ALL':
    # then we'll not filter according to gamma
    get_all_gamma = True
  elif gamma is not None:
    gamma = round(gamma, 3)


  fname = f"{main_directory}/annealing_results/{problem_type}/{solver_id}/NTS|{num_total_sets}_NSS|{num_solution_sets}_seed|{seed}/AT|{AT}"

  if problem_type != 'direct':
    if get_all_gamma:
      fname += '*'
    else:
      fname += f"_gamma|{float(gamma)}"

  if as_pandas:
    fname += ".csv"
  else:
    fname += '.json'

  if get_all_gamma: 
    return glob.glob(fname)
  else:
    return fname
  
def get_fname_for_results(L:int, seed:int, error_model: str, precision:int, problem_type:str, solver_id:str, AT:float, gamma:float, main_directory:str="./1D_Ising_Chain", rcs:float=None, as_pandas:bool=True, tiled:bool = False) -> str:
  '''
    Given the necessary parameters, returns the filename for the storing/loading annaling results.
    Files are named as:
    main_dir/[tiled_]annealing_results/problem_type/error_model/solver_id/problem_index/fname.json
    if gamma = 'ALL' then a list of filenames are returned without filtering according to gamma.
   '''
  # first let's round the floating points to fixed precision
  # {main_directory}/tiled_annealing_results/{error_model}/{solver_id}/precision-{precision}/{problem_type}/L|{L}_seed|{seed}/AT|{at}_gamma|{gamma}.json"
  AT = round(float(AT), 2)
  get_all_gamma = False
  if gamma == 'ALL':
    # then we'll not filter according to gamma
    get_all_gamma = True
  elif gamma is not None:
    gamma = round(gamma, 3)
  if rcs is not None:
    rcs = round(rcs, 3)

  if tiled:
    fname = f"{main_directory}/tiled_annealing_results/{problem_type}/{error_model}/{solver_id}/"
  else:
    fname = f"{main_directory}/annealing_results/{problem_type}/{error_model}/{solver_id}/"

  if error_model != 'no_error':
    fname += f"precision-{int(precision)}/"

  fname += f"L|{L}_seed|{int(seed)}/AT|{float(AT)}"
  if problem_type != 'direct':
    if get_all_gamma:
      fname += '*'
    else:
      fname += f"_gamma|{float(gamma)}"
  
  if rcs is not None:
    print("\n\n I SHOULDNT BE HERE \n\n")
    fname += f"_rcs|{rcs}"
  
  if as_pandas:
    fname += ".csv"
  else:
    fname += '.json'

  if get_all_gamma: 
    return glob.glob(fname)
  else:
    return fname

def get_sigma_from_precision(p: int) -> float:
    """
    Calculate the sigma value for a Gaussian distribution given a precision p.

    Args:
    p (int): The precision value used in the quantization/random error model.

    Returns:
    float: The corresponding sigma value for the Gaussian distribution.
    """
    return (1 / (2 ** p)) + (1 / (2 ** (p + 1)))

# import os

# def rename_files(directory, old_text, new_text):
#     """
#     Renames files in the given directory by replacing the specified old text 
#     with new text in the file names.

#     Parameters:
#         directory (str): The path to the directory containing the files.
#         old_text (str): The text in the file names to be replaced.
#         new_text (str): The text to replace with.
#     """
#     for filename in os.listdir(directory):
#         if old_text in filename:
#             # Construct the new filename
#             new_filename = filename.replace(old_text, new_text)
            
#             # Renaming the file
#             os.rename(os.path.join(directory, filename), 
#                       os.path.join(directory, new_filename))
#             print(f"Renamed '{filename}' to '{new_filename}'")

# if __name__ == '__main__':
#   print("Renaming shit")
#   # Example Usage
#   directory_path = '/Users/beratyenilen/Desktop/Thesis/CODE/scripts/1D_Ising_Chain/embeddings/Advantage_system6.3'  # Replace with the path to your files
#   old_text = 'pegasus_emb'  # The text in the filename you want to replace
  
#   #new_text = 'triangle_direct'  # The new text you want to insert in the filename
#   new_text = 'direct'
#   rename_files(directory_path, old_text, new_text)

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
    '''
        Given the file_path for a problem instance returns the BQM, ground_state, ground_state_energy
    '''
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
