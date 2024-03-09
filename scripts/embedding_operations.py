# Imports
from dwave.system import DWaveSampler, EmbeddingComposite, FixedEmbeddingComposite
import dimod
import networkx 
import dwave_networkx as dnx
from bqm_operations import tile_BQM
import json
import glob
from bqm_operations import copy_BQM


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

def isDirectEmbedding(emb: dict, verbose: bool = False) -> bool:
  '''
    Returns True if the given embedding is a direct embedding, 
    Returns False, otherwise.
  '''
  verbose_list = []
  is_direct_embedding = True
  for var, chain in emb.items():
    if len(chain) > 1:
      is_direct_embedding = False
      if verbose:
        verbose_list.append(f"The chain for {var} is = {chain} with length {len(chain)}")
  
  return is_direct_embedding

def check_embedding(copied_bqm: dimod.BinaryQuadraticModel, embedding: dict, sampler_graph =None, verbose: bool =False) -> bool:
  '''
    Checks if the given embedding with the BQM with num_copies is a working one for the Pegasus graph.
    Returns True if it is a valid embedding.
  '''
  if sampler_graph != None:
    pegasus_graph = sampler_graph.copy()
  elif type(list(embedding.values())[0][0]) == tuple:
    # this means the embedding is listed with nice coordinates
    pegasus_graph = dnx.pegasus_graph(16, nice_coordinates=True)
  else:
    pegasus_graph = dnx.pegasus_graph(16)

  for var0,var1 in copied_bqm.quadratic.keys():
    chain0 = embedding[var0]
    chain1 = embedding[var1]
    flag = False
    for qb0 in chain0:
      for qb1 in chain1:
        if pegasus_graph.has_edge(qb0, qb1):
          flag = True
    if flag == False:
      if verbose:
        print(f"Chain of {var0}={chain0} is not connected to chain of {var1}={chain1}")
      return False
    
  return True

def tile_embedding(org_bqm, emb: dict, target_graph: networkx.Graph, max_num_tiles:int=None, graph_type: str='pegasus', return_num_tiles:bool=False,  gamma:float=None, copy_bqm: bool=False, verbose:bool = False,) -> dict:
  '''
    Given an embedding and a target graph, returns the tiled embedding over the whole graph.

  '''
  assert graph_type == 'pegasus', "Only pegasus-type graphs are allowed."

  assert (gamma == None and copy_bqm == False) or (gamma != None and copy_bqm == True), f"For copying a Gamma value should be provided"
  
  pegasus_max_x, pegasus_max_y = 16,16
  pegasus_coords = dnx.pegasus_coordinates(16)

  target_nodes = target_graph.nodes

  used_nodes = set()
  min_x, max_x = 9999, -1
  min_y, max_y = 9999, -1
  # get the minimum and max x,y values used in the embedding
  for chain in emb.values():
    t, y, x, u, k  = pegasus_coords.linear_to_nice(chain[0])
    if x < min_x:
      min_x = x
    if x > max_x:
      max_x = x

    if y < min_y:
      min_y = y
    if y > max_y:
      max_y = y

  # we will know increment the x,y values to tile the pegasus graph
  x_increment = max_x - min_x + 1
  y_increment = max_y - min_y + 1

  if verbose:
    print(f"min_x={min_x}, max_x={max_x}, min_y={min_y}, max_y={max_y}")
    print("x_increment=", x_increment, "y_increment=", y_increment)
    print(f"num x tiles = {(pegasus_max_x - max_x ) // x_increment + 1}")
    print(f"num y tiles = {(pegasus_max_y - max_y) // y_increment + 1}")
  
  all_embs = {}
  all_embs_final = {}
  tile_counter = 0
  tiled_bqm = tile_BQM(org_bqm=org_bqm, num_tiles=tile_counter+1, copy_bqm=copy_bqm, gamma=gamma)
  if not copy_bqm:
    # this means that the given embedding is for the original BQM and is of the form:
    # {0:[190], 1:[233], ...}
    # I can simply transform it into 
    # {'0_0:[190], '0_1':[233], ...}
    emb = {f"0_{q}":ch for q,ch in emb.items()}
  #  -------------------------- Increase x and y ---------------------------------
  if verbose:
    print(f"Increase x and increase y")

  for x_scaler in range((pegasus_max_x - max_x ) // x_increment + 1):
    if verbose:
      print(f"x_scaler={x_scaler}")
    for y_scaler in range((pegasus_max_y - max_y) // y_increment + 1):
      if verbose:
        print(f"\ty_scaler={y_scaler}")

      temp = {}
      flag = False
      for var, chain in emb.items():
        t,y,x,u,k = pegasus_coords.linear_to_nice(chain[0])

        new_nice_coords = (t, y + y_scaler*y_increment, x + x_scaler*x_increment, u, k)
        new_linear_coords = pegasus_coords.nice_to_linear(new_nice_coords)
        # check if the coordinates are valid or if this node is already being used
        if new_linear_coords not in target_nodes or new_linear_coords in used_nodes:
          if verbose:
            print("\t\tflag set to true for qubit", new_linear_coords, new_nice_coords)
          flag = True
          break
        
        temp[f"{tile_counter}_{var}"] = [new_linear_coords]
      #check_embedding(tiled_bqm, all_embs|temp, target_graph, verbose=verbose)
      if flag:
        continue
      elif tile_counter > 0 and not check_embedding(tiled_bqm, all_embs|temp, target_graph, verbose=verbose):
        continue
      else:
        tile_counter += 1
        tiled_bqm = tile_BQM(org_bqm=org_bqm, num_tiles=tile_counter+1, copy_bqm=copy_bqm, gamma=gamma)
        all_embs.update(temp)
        # add all the nodes used in this embedding to the used nodes
        for var,chain in temp.items():
          for qb in chain:
            used_nodes.add(qb)

        if max_num_tiles != None and tile_counter == max_num_tiles:
          if verbose:
            print(f"Total num tiles={tile_counter}")
          # this means we should stop tiling
          if return_num_tiles:
            return all_embs, tile_counter
          else:
            return all_embs
          
  # ------------------------------ Decrease x and y --------------------------------------
  if verbose:
    print(f"Decrease x and decrease y")

  for x_scaler in range(min_x // x_increment + 1, 0, -1):
    if verbose:
      print(f"x_scaler={x_scaler}")
    for y_scaler in range(min_y // y_increment + 1, 0, -1):
      if verbose:
        print(f"\ty_scaler={y_scaler}")

      temp = {}
      flag = False
      for var, chain in emb.items():
        t,y,x,u,k = pegasus_coords.linear_to_nice(chain[0])

        new_nice_coords = (t, y - y_scaler*y_increment, x - x_scaler*x_increment, u, k)
        new_linear_coords = pegasus_coords.nice_to_linear(new_nice_coords)
        if new_linear_coords not in target_nodes or new_linear_coords in used_nodes:
          if verbose:
            print("\t\tflag set to true for qubit", new_linear_coords, new_nice_coords)
          flag = True
          break
        
        temp[f"{tile_counter}_{var}"] = [new_linear_coords]
      #check_embedding(tiled_bqm, all_embs|temp, target_graph, verbose=verbose)
      if flag:
        continue
      elif tile_counter > 0 and not check_embedding(tiled_bqm, all_embs|temp, target_graph, verbose=verbose):
        continue
      else:
        tile_counter += 1
        tiled_bqm = tile_BQM(org_bqm=org_bqm, num_tiles=tile_counter+1, gamma=gamma, copy_bqm=copy_bqm)
        all_embs.update(temp)
        # add all the nodes used in this embedding to the used nodes
        for var,chain in temp.items():
          for qb in chain:
            used_nodes.add(qb)
        if max_num_tiles != None and tile_counter == max_num_tiles:
          if verbose:
            print(f"Total num tiles={tile_counter}")
          # this means we should stop tiling
          if return_num_tiles:
            return all_embs, tile_counter
          else:
            return all_embs
          
  # ------------------------------- Increase x and decrease y ----------------------------------
  if verbose:
    print(f"Increase x and decrease y")

  for x_scaler in range((pegasus_max_x - max_x ) // x_increment + 1):
    if verbose:
      print(f"x_scaler={x_scaler}")
    for y_scaler in range(min_y // y_increment + 1, 0, -1):
      if verbose:
        print(f"\ty_scaler={y_scaler}")

      temp = {}
      flag = False
      for var, chain in emb.items():
        t,y,x,u,k = pegasus_coords.linear_to_nice(chain[0])

        new_nice_coords = (t, y - y_scaler*y_increment, x + x_scaler*x_increment, u, k)
        new_linear_coords = pegasus_coords.nice_to_linear(new_nice_coords)
        if new_linear_coords not in target_nodes or new_linear_coords in used_nodes:
          if verbose:
            print("\t\tflag set to true for qubit", new_linear_coords, new_nice_coords)
          flag = True
          break
        
        temp[f"{tile_counter}_{var}"] = [new_linear_coords]
      #check_embedding(tiled_bqm, all_embs|temp, target_graph, verbose=verbose)
      if flag:
        continue
      elif tile_counter > 0 and not check_embedding(tiled_bqm, all_embs|temp, target_graph, verbose=verbose):
        continue
      else:
        tile_counter += 1
        tiled_bqm = tile_BQM(org_bqm=org_bqm, num_tiles=tile_counter+1, gamma=gamma, copy_bqm=copy_bqm)
        all_embs.update(temp)
        # add all the nodes used in this embedding to the used nodes
        for var,chain in temp.items():
          for qb in chain:
            used_nodes.add(qb)
        if max_num_tiles != None and tile_counter == max_num_tiles:
          if verbose:
            print(f"Total num tiles={tile_counter}")
          # this means we should stop tiling
          if return_num_tiles:
            return all_embs, tile_counter
          else:
            return all_embs
          
  #  -----------------------    Decrease x and increase y ------------------------------
  if verbose:
    print(f"Decrease x and decrease y")

  for x_scaler in range(min_x // x_increment + 1, 0, -1):
    if verbose:
      print(f"x_scaler={x_scaler}")
    for y_scaler in range((pegasus_max_y - max_y) // y_increment + 1):
      if verbose:
        print(f"\ty_scaler={y_scaler}")

      temp = {}
      flag = False
      for var, chain in emb.items():
        t,y,x,u,k = pegasus_coords.linear_to_nice(chain[0])

        new_nice_coords = (t, y + y_scaler*y_increment, x - x_scaler*x_increment, u, k)
        new_linear_coords = pegasus_coords.nice_to_linear(new_nice_coords)
        if new_linear_coords not in target_nodes or new_linear_coords in used_nodes:
          if verbose:
            print("\t\tflag set to true for qubit", new_linear_coords, new_nice_coords)
          flag = True
          break
        
        temp[f"{tile_counter}_{var}"] = [new_linear_coords]
      if flag:
        continue
      elif tile_counter > 0 and not check_embedding(tiled_bqm, all_embs|temp, target_graph, verbose=verbose):
        continue
      else:
        tile_counter += 1
        tiled_bqm = tile_BQM(org_bqm=org_bqm, num_tiles=tile_counter+1, gamma=gamma, copy_bqm=copy_bqm)
        all_embs.update(temp)
        # add all the nodes used in this embedding to the used nodes
        for var,chain in temp.items():
          for qb in chain:
            used_nodes.add(qb)
        if max_num_tiles != None and tile_counter == max_num_tiles:
          if verbose:
            print(f"Total num tiles={tile_counter}")
          # this means we should stop tiling
          if return_num_tiles:
            return all_embs, tile_counter
          else:
            return all_embs
          
  if verbose:
    print(f"Total num tiles={tile_counter}")

  if return_num_tiles:
    return all_embs, tile_counter
  else:
    return all_embs

def isValidTriangle(source, target, pegasus_graph, return_mapping: bool=False):
  '''
    Returns True if we can make a 1-to-1 mapping between the nodes of source and target.
  '''
  # check if they have intersecting qubits
  if len(set(source) & set(target)) != 0:
    return False
  source = list(source)
  target = list(target)

  # create a list of available edges between the two triangles
  edges = set()
  flag = False
  for s in range(3):
    counter = 0
    for t in range(3):
      counter += 1
      if pegasus_graph.has_edge(source[s],target[s]):
        counter = 0
        edges.add((s,t))
      if counter == 3:
        #print("No edges were added for", s)
        flag = True

  # valid mappings are
  valid_mappings = [{(0,0), (1,1), (2,2)}, {(0,0), (1,2), (2,1)}, 
                    {(0,1), (1,0), (2,2)}, {(0,1), (1,2), (2,0)},
                    {(0,2), (1,0), (2,1)}, {(0,2), (1,1), (2,0)}]
  #print(edges)
  if flag:
    return False
  if return_mapping:
    available_mappings = []
    for mapp in valid_mappings:
      if mapp & edges == mapp:
        available_mappings.append(mapp)
    return len(available_mappings) != 0, available_mappings
  else:
    for mapp in valid_mappings:
      if mapp & edges == mapp:
        return True
      
def triangle_embedding(bqm: dimod.BinaryQuadraticModel, target_graph = None, return_graph: bool=False, m:int=16, node_type:int=18) -> dict:
  '''
    Embeds the given BQM for the 1D Ising chain problem in a triangle copy. 
    If return_graph = False then it returns embedding
    If return_graph = True then it returns embedding, pegasus graph w/ only the relevant nodes
  '''
  # Get the graph, coords and nodes
  m = 16
  if target_graph == None:
    pegasus_graph = dnx.pegasus_graph(m, nice_coordinates=False)
    pegasus_nodes = list(pegasus_graph.nodes)
  else:
    pegasus_graph = target_graph.copy()
    pegasus_nodes = target_graph.nodes
      
  
  L = len(bqm.variables)
  # let's focus only on node_type nodes
  usable_nodes = list(pegasus_nodes)

  # get all triad cliques
  triad_cliques=[set(x) for x in networkx.enumerate_all_cliques(pegasus_graph) if len(x)==3]

  # construct a dictionary such that node2triangle[node] gives the list of triads this node forms
  node2triangles = {}
  for triad in triad_cliques:
    tri = list(triad)
    for i in range(3):
      if tri[i] not in node2triangles:
        node2triangles[tri[i]] = []
      node2triangles[tri[i]].append(tri)

  def recursive_part(L, initial_triad, used_nodes):
    if L == 0:
      return used_nodes
    initial_node = list(initial_triad)[0]
    # get the neighbors
    neighbors = [n for n in pegasus_graph.neighbors(initial_node) if n in usable_nodes and n not in used_nodes]
    # get all the valid neighbor triads
    neighbor_triads = []
    for neighbor in neighbors:
      for triad in node2triangles[neighbor]:
        if len(set(triad) & set(used_nodes)) == 0 and isValidTriangle(initial_triad, triad, pegasus_graph):
          neighbor_triads.append(triad)
    # try them out
    retval = None
    for new_triad in neighbor_triads:
      retval = recursive_part(L-1, new_triad, used_nodes + new_triad)
      if retval !=  None:
        return retval
    return None
  
  # Now we'll construct the relevant graph and an embedding
  # for node in usable nodes:
  #   for triad for ths node
  #     assume this is a correct decision and construct a graph with length L-1
  #     if it fails try the next triad
  #   if all fails try the next node
  break_flag = False
  node_list = None
  for node in usable_nodes:
    if break_flag:
      break
    for triad in node2triangles[node]:    
      # this variable will hold a list of nodes for triads
      # triangle 0 vertex 1 - > t0v1
      # [t0v1, t0v2, t0v3, t1v1, t2v2, t2v3, ...]
      node_list = recursive_part(L-1, triad, triad)
      if node_list != None:
        break_flag = True
        break

  if node_list == None:
    raise Exception("Couldn't find a graph")
  
  # if here, then a solution was found
  # lets construct a list which will be like:
  # [0,1,2, 2,1,0, 0,2,1, ...]
  # Which holds the copy number for each qubit
  copy_list = [0,1,2]
  for i in range(L-1):
    s, mapp = isValidTriangle(node_list[3*i:3*(i+1)], node_list[3*(i+1):3*(i+2)], pegasus_graph, return_mapping=True)
    chosen_map = mapp[0]
    temp = [-1,-1,-1]
    for ss, tt in chosen_map:
      temp[ss] = tt
    copy_list += temp
    
  # generate the embedding
  embedding = {}
  for i, var in enumerate(bqm.variables):
    for j in range(3):
      index = 3*i  + j
      embedding[f"{copy_list[index]}_{var}"] = [node_list[index]]

  if return_graph:
    for node in pegasus_nodes:
      if node not in node_list:
        pegasus_graph.remove_node(node)
    
    return embedding, pegasus_graph
  else:
    return embedding
      
def load_embedding(L:int, emb_type:str, solver_id: str, main_directory: str="./1D_Ising_Chain", tiled: bool=False) -> dict:
  '''
    Given the length of the 1D Ising Chain returns the embedding for the given emb_type.
    Assumes the 1D_Ising_Chain folder is in the working directory!
    Param:
      L (int) : length of the Ising chain
      emb_type (str) : It can be either 'direct', 'triangle_direct'
      solver_id (str) : It can be 'Advantage_system4.1', 'Advantage_system5.4', or 'Advantage_system6.3'. IF not specified
                        the embedding for the Pegasus graph is returned.
    Return:
      emb, [graph]
  '''
  allowed_emb_types = ['direct', 'triangle_direct']
  assert emb_type in allowed_emb_types, f"Allowed emb_types are {allowed_emb_types}"
  
  allowed_solver_ids = ["Advantage_system5.4", "Advantage_system6.3", "Advantage_system4.1"]
    
  assert solver_id in allowed_solver_ids, f"Allowed solver_id s are {allowed_solver_ids}"
  
  if tiled:
    fname = glob.glob(f"{main_directory}/tiled_embeddings/{solver_id}/{emb_type}_L-{L}*.json")[0]
    
    if len(glob.glob(f"{main_directory}/tiled_embeddings/{solver_id}/{emb_type}_L-{L}*.json")) != 1:
      raise Exception(f"File name is not uniquely defined for L={L} solver_id={solver_id} emb_type={emb_type}, tiled={tiled}, main_directory={main_directory}")
    
    num_tiles = int(fname.split("/")[-1].split('_')[-1].split('.')[0].split('-')[-1])
    with open(fname, 'r') as f:
      emb = json.load(f) 
  else:
    with open(f"{main_directory}/embeddings/{solver_id}/{emb_type}_L-{L}.json", 'r') as f:
      emb = json.load(f)  
        
  # if it is a direct embedding, we need to transform the variables to int
  if emb_type == "direct" and not tiled:
    emb = {int(i):j for i,j in emb.items()}  

  if tiled:
    return emb, num_tiles
  else:
    return emb

def tile_minor_embedding(org_bqm, emb, sampler_graph, max_num_tiles, sampler=None):
  '''
      Given a BQM, valid embedding for the BQM onto the sampler_graph, tiles and returns the tiled
      embedding. If max_num_tiles can't be reached the reached tile number is printed and it s returned.
  '''
  # Initialize the Pegasus coordinates helper for a Pegasus graph of size 16
  pegasus_coords = dnx.pegasus_coordinates(16)
  pegasus_max_x =  16
  pegasus_max_y =  16

  sampler_nodes = sampler_graph.nodes
  # Transform the original embedding to Pegasus nice coordinates
  nice_emb = {var: [pegasus_coords.linear_to_nice(node) for node in nodes] for var, nodes in emb.items()}
  
  # Find the bounding x,y values for the embedding
  min_x, min_y = float('inf'), float('inf')
  max_x, max_y = -float('inf'), -float('inf')
  for _, nodes in nice_emb.items():
    for t, y, x, u, k in nodes:
      min_x, max_x = min(min_x, x), max(max_x, x)
      min_y, max_y = min(min_y, y), max(max_y, y)

  # Calculate the size of the grid we're using
  grid_width, grid_height = max_x - min_x + 1, max_y - min_y + 1
  
  # Tile the embedding across the specified number of tiles, if possible
  tiled_emb = {}
  tile_num = 0
  valid_tiles = 0
  # let's set this as the maximum value tile_num can take so that
  # the while terminates
  max_tile_num = 1000
  while valid_tiles < max_num_tiles:
    if tile_num > max_tile_num:
      print(f"Reached maximum tile_num! Found {valid_tiles} many valid tiles!")
      return tiled_emb
    temp = {}
    flag = False
    for var, chain in nice_emb.items():
      if flag:
        break
      tiled_chain = []
      for t, y, x, u, k in chain:
        # Calculate new x, y within tile bounds, ensuring we stay within the Pegasus graph dimensions
        new_x = (x - min_x + (tile_num % grid_width) * grid_width) % pegasus_max_x
        new_y = (y - min_y + (tile_num // grid_width) * grid_height) % pegasus_max_y
        # Convert back to linear coordinates and append
        linear_node = pegasus_coords.nice_to_linear((t, new_y, new_x, u, k))
        if linear_node not in sampler_nodes:
          flag = True
          break
        tiled_chain.append(linear_node)
      temp[f"{valid_tiles}_{var}"] = tiled_chain
        #tiled_emb[f"{tile_num}_{var}"] = tiled_nodes
    # check if it is a valid embedding
    if flag:
      tile_num +=1
      continue
    tiled_bqm = copy_BQM(bqm=org_bqm, gamma=0, num_copies=valid_tiles+1, copy_type='unconnected' )
    if check_embedding(tiled_bqm, tiled_emb|temp, sampler_graph) :
        #let's do a further check
        try:
          trying = FixedEmbeddingComposite(sampler, tiled_emb | temp)
          # if this works awesome
          tiled_emb = tiled_emb | temp
          valid_tiles += 1
        except:
          pass
    tile_num += 1

  return tiled_emb