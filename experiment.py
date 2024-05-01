import sys
sys.path.append('build')
import isohash
import igraph
import numpy as np
import pandas as pd
import random
import time


def create_isomorphic_matrix(adj_matrix):
    # Number of nodes
    n = adj_matrix.shape[0]

    # Generate a random permutation of node labels
    perm = list(range(n))
    random.shuffle(perm)

    # Apply this permutation to both rows and columns
    return adj_matrix[perm, :][:, perm]


# test hash on 500 directed isomorphic erdos renyi random graphs with 200 nodes and p=0.5 with n=10
print("Testing Undirected Isomorphic Mappings...")
results_df = pd.DataFrame(columns=['Random_Graph_Seed', 'Node_Hash_Result', 'Elapsed_Node_Hash', 'Edge_Hash_Result', 'Elapsed_Edge_Hash'])
for i in range(500):
    print(i)
    random_graph_seed = random.random()
    random.seed(random_graph_seed)
    g1 = igraph.Graph.Erdos_Renyi(200, 0.5, directed=True)
    adjacency_matrix = np.array(g1.get_adjacency().data, dtype=np.int64)
    isomorphic_adjacency_matrix = create_isomorphic_matrix(adjacency_matrix)

    start = time.perf_counter()
    node_hash_result = isohash.nodeHashCompare(adjacency_matrix, isomorphic_adjacency_matrix, 10)
    end = time.perf_counter()
    elapsed_node_hash = end - start

    start = time.perf_counter()
    edge_hash_result = isohash.edgeHashCompare(adjacency_matrix, isomorphic_adjacency_matrix, 10)
    end = time.perf_counter()
    elapsed_edge_hash = end - start

    results_df.loc[i] = [random_graph_seed, node_hash_result, elapsed_node_hash, edge_hash_result, elapsed_edge_hash]

results_df.to_csv('directed_isomorphic.csv', index=False)


# test hash on 500 undirected isomorphic erdos renyi random graphs with 200 nodes and p=0.5 with n=10
print("Testing Directed Isomorphic Mappings...")
results_df = pd.DataFrame(columns=['Random_Graph_Seed', 'Node_Hash_Result', 'Elapsed_Node_Hash', 'Edge_Hash_Result', 'Elapsed_Edge_Hash'])
for i in range(500):
    print(i)
    random_graph_seed = random.random()
    random.seed(random_graph_seed)
    g1 = igraph.Graph.Erdos_Renyi(200, 0.5, directed=False)
    adjacency_matrix = np.array(g1.get_adjacency().data, dtype=np.int64)
    isomorphic_adjacency_matrix = create_isomorphic_matrix(adjacency_matrix)

    start = time.perf_counter()
    node_hash_result = isohash.nodeHashCompare(adjacency_matrix, isomorphic_adjacency_matrix, 10)
    end = time.perf_counter()
    elapsed_node_hash = end - start

    start = time.perf_counter()
    edge_hash_result = isohash.edgeHashCompare(adjacency_matrix, isomorphic_adjacency_matrix, 10)
    end = time.perf_counter()
    elapsed_edge_hash = end - start

    results_df.loc[i] = [random_graph_seed, node_hash_result, elapsed_node_hash, edge_hash_result, elapsed_edge_hash]

results_df.to_csv('undirected_isomorphic.csv', index=False)

# test hash on 500 directed non-isomorphic erdos renyi random graphs with 200 nodes and p=0.5 with n=10
print("Testing Directed Nonisomorphic Mappings...")
results_df = pd.DataFrame(columns=['Random_Graph_Seed', 'Node_Hash_Result', 'Elapsed_Node_Hash', 'Edge_Hash_Result', 'Elapsed_Edge_Hash'])
for i in range(500):
    random_graph_seed = random.random()
    random.seed(random_graph_seed)
    g1 = igraph.Graph.Erdos_Renyi(200, 0.5, directed=True)
    g2 = igraph.Graph.Erdos_Renyi(200, 0.5, directed=True)
    adjacency_matrix_1 = np.array(g1.get_adjacency().data, dtype=np.int64)
    adjacency_matrix_2 = np.array(g2.get_adjacency().data, dtype=np.int64)

    start = time.perf_counter()
    node_hash_result = isohash.nodeHashCompare(adjacency_matrix_1, adjacency_matrix_2, 10)
    end = time.perf_counter()
    elapsed_node_hash = end - start

    start = time.perf_counter()
    edge_hash_result = isohash.edgeHashCompare(adjacency_matrix_1, adjacency_matrix_2, 10)
    end = time.perf_counter()
    elapsed_edge_hash = end - start

    results_df.loc[i] = [random_graph_seed, node_hash_result, elapsed_node_hash, edge_hash_result, elapsed_edge_hash]

results_df.to_csv('directed_nonisomorphic.csv', index=False)

# test hash on 500 undirected non-isomorphic erdos renyi random graphs with 200 nodes and p=0.5 with n=10
print("Testing Undirected Nonisomorphic Mappings...")
results_df = pd.DataFrame(columns=['Random_Graph_Seed', 'Node_Hash_Result', 'Elapsed_Node_Hash', 'Edge_Hash_Result', 'Elapsed_Edge_Hash'])
for i in range(500):
    print(i)
    random_graph_seed = random.random()
    random.seed(random_graph_seed)
    g1 = igraph.Graph.Erdos_Renyi(200, 0.5, directed=False)
    g2 = igraph.Graph.Erdos_Renyi(200, 0.5, directed=False)
    adjacency_matrix_1 = np.array(g1.get_adjacency().data, dtype=np.int64)
    adjacency_matrix_2 = np.array(g2.get_adjacency().data, dtype=np.int64)

    start = time.perf_counter()
    node_hash_result = isohash.nodeHashCompare(adjacency_matrix_1, adjacency_matrix_2, 10)
    end = time.perf_counter()
    elapsed_node_hash = end - start

    start = time.perf_counter()
    edge_hash_result = isohash.edgeHashCompare(adjacency_matrix_1, adjacency_matrix_2, 10)
    end = time.perf_counter()
    elapsed_edge_hash = end - start

    results_df.loc[i] = [random_graph_seed, node_hash_result, elapsed_node_hash, edge_hash_result, elapsed_edge_hash]

results_df.to_csv('undirected_nonisomorphic.csv', index=False)

test hash on 10 undirected isomorphic erdos renyi random graphs with 1000 nodes and p=0.9 with n=10
print("Testing Integer Overflow...")
results_df = pd.DataFrame(columns=['Random_Graph_Seed', 'Node_Hash_Result', 'Elapsed_Node_Hash', 'Edge_Hash_Result', 'Elapsed_Edge_Hash'])
for i in range(10):
    print(i)
    random_graph_seed = random.random()
    random.seed(random_graph_seed)
    g1 = igraph.Graph.Erdos_Renyi(1000, 0.9, directed=False)
    adjacency_matrix = np.array(g1.get_adjacency().data, dtype=np.int64)
    isomorphic_adjacency_matrix = create_isomorphic_matrix(adjacency_matrix)

    start = time.perf_counter()
    node_hash_result = isohash.nodeHashCompare(adjacency_matrix, isomorphic_adjacency_matrix, 10)
    end = time.perf_counter()
    elapsed_node_hash = end - start

    start = time.perf_counter()
    edge_hash_result = isohash.edgeHashCompare(adjacency_matrix, isomorphic_adjacency_matrix, 10)
    end = time.perf_counter()
    elapsed_edge_hash = end - start

    results_df.loc[i] = [random_graph_seed, node_hash_result, elapsed_node_hash, edge_hash_result, elapsed_edge_hash]

results_df.to_csv('overflow_undirected_isomorphic.csv', index=False)

# test hash on increasingly large undirected isomorphic erdos renyi random graphs with p=0.5 with heuristic n=10
print("Testing Scaling Isomorphic Mappings...")
results_df = pd.DataFrame(columns=['Nodes', 'Random_Graph_Seed', 'Node_Hash_Result', 'Elapsed_Node_Hash', 'Edge_Hash_Result', 'Elapsed_Edge_Hash'])
for i in range(1, 100):
    print(i)
    random_graph_seed = random.random()
    random.seed(random_graph_seed)
    g1 = igraph.Graph.Erdos_Renyi(10*i, 0.5, directed=False)
    adjacency_matrix = np.array(g1.get_adjacency().data, dtype=np.int64)
    isomorphic_adjacency_matrix = create_isomorphic_matrix(adjacency_matrix)

    start = time.perf_counter()
    node_hash_result = isohash.nodeHashCompare(adjacency_matrix, isomorphic_adjacency_matrix, 5)
    end = time.perf_counter()
    elapsed_node_hash = end - start

    start = time.perf_counter()
    edge_hash_result = isohash.edgeHashCompare(adjacency_matrix, isomorphic_adjacency_matrix, 5)
    end = time.perf_counter()
    elapsed_edge_hash = end - start

    results_df.loc[i] = [10*i, random_graph_seed, node_hash_result, elapsed_node_hash, edge_hash_result, elapsed_edge_hash]

results_df.to_csv('scaling_isomorphic.csv', index=False)

# test hash on logarithically increasingly large undirected isomorphic erdos renyi random graphs with p=0.5 with heuristic n=10
print("Testing Logarithmically Scaling Nonisomorphic Mappings...")
results_df = pd.DataFrame(columns=['Nodes', 'Random_Graph_Seed', 'Node_Hash_Result', 'Elapsed_Node_Hash'])
for i in range(1, 40):
    print(i)
    random_graph_seed = random.random()
    random.seed(random_graph_seed)
    g1 = igraph.Graph.Erdos_Renyi(100*i, 0.5, directed=True)
    g2 = igraph.Graph.Erdos_Renyi(100*i, 0.5, directed=True)
    adjacency_matrix_1 = np.array(g1.get_adjacency().data, dtype=np.int64)
    adjacency_matrix_2 = np.array(g2.get_adjacency().data, dtype=np.int64)

    start = time.perf_counter()
    node_hash_result = isohash.nodeHashCompare(adjacency_matrix_1, adjacency_matrix_2, 10)
    end = time.perf_counter()
    elapsed_node_hash = end - start

    results_df.loc[i] = [100*i, random_graph_seed, node_hash_result, elapsed_node_hash]

results_df.to_csv('scaling_nonisomorphic_nodehash.csv', index=False)

# test hash on increasingly large non-isomorphic erdos renyi random graphs with p=0.5 with heuristic n=10
print("Testing Scaling Nonisomorphic Mappings...")
results_df = pd.DataFrame(columns=['Nodes', 'Random_Graph_Seed', 'Node_Hash_Result', 'Elapsed_Node_Hash', 'Edge_Hash_Result', 'Elapsed_Edge_Hash'])
for i in range(1, 100):
    print(i)
    random_graph_seed = random.random()
    random.seed(random_graph_seed)
    g1 = igraph.Graph.Erdos_Renyi(10*i, 0.5, directed=True)
    g2 = igraph.Graph.Erdos_Renyi(10*i, 0.5, directed=True)
    adjacency_matrix_1 = np.array(g1.get_adjacency().data, dtype=np.int64)
    adjacency_matrix_2 = np.array(g2.get_adjacency().data, dtype=np.int64)

    start = time.perf_counter()
    node_hash_result = isohash.nodeHashCompare(adjacency_matrix_1, adjacency_matrix_2, 10)
    end = time.perf_counter()
    elapsed_node_hash = end - start

    start = time.perf_counter()
    edge_hash_result = isohash.edgeHashCompare(adjacency_matrix_1, adjacency_matrix_2, 10)
    end = time.perf_counter()
    elapsed_edge_hash = end - start

    results_df.loc[i] = [10*i, random_graph_seed, node_hash_result, elapsed_node_hash, edge_hash_result, elapsed_edge_hash]

results_df.to_csv('scaling_nonisomorphic.csv', index=False)