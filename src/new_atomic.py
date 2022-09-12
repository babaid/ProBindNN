from typing import Any, Dict

import networkx as nx
import numpy as np
import pandas as pd

from graphein.protein.edges.distance import compute_distmat
from graphein.protein.resi_atoms import (
    BOND_LENGTHS,
    BOND_ORDERS,
    COVALENT_RADII,
    DEFAULT_BOND_STATE,
    RESIDUE_ATOM_BOND_STATE,
)

from graphein.protein.edges.atomic import assign_bond_states_to_dataframe, assign_covalent_radii_to_dataframe
def add_atomic_edges(G: nx.Graph, tolerance: float = 0.56) -> nx.Graph:
    """
    Computes covalent edges based on atomic distances. Covalent radii are assigned to each atom based on its bond assign_bond_states_to_dataframe
    The distance matrix is then thresholded to entries less than this distance plus some tolerance to create an adjacency matrix.
    This adjacency matrix is then parsed into an edge list and covalent edges added

    :param G: Atomic graph (nodes correspond to atoms) to populate with atomic bonds as edges
    :type G: nx.Graph
    :param tolerance: Tolerance for atomic distance. Default is ``0.56`` Angstroms. Commonly used values are: ``0.4, 0.45, 0.56``
    :type tolerance: float
    :return: Atomic graph with edges between bonded atoms added
    :rtype: nx.Graph
    """
    dist_mat = compute_distmat(G.graph["pdb_df"])

    # We assign bond states to the dataframe, and then map these to covalent radii
    G.graph["pdb_df"] = assign_bond_states_to_dataframe(G.graph["pdb_df"])
    G.graph["pdb_df"] = assign_covalent_radii_to_dataframe(G.graph["pdb_df"])

    # Create a covalent 'distance' matrix by adding the radius arrays with its transpose
    covalent_radius_distance_matrix = np.add(
        np.array(G.graph["pdb_df"]["covalent_radius"]).reshape(-1, 1),
        np.array(G.graph["pdb_df"]["covalent_radius"]).reshape(1, -1),
    )

    # Add the tolerance
    covalent_radius_distance_matrix = (
        covalent_radius_distance_matrix + tolerance
    )

    # Threshold Distance Matrix to entries where the eucl distance is less than the covalent radius plus tolerance and larger than 0.4
    dist_mat = dist_mat[dist_mat > 0.4]
    t_distmat = dist_mat[dist_mat < covalent_radius_distance_matrix]





    dist_mat_longrange = dist_mat[dist_mat > 0.4]
    t_distmat_longrange = dist_mat[dist_mat < 12.]
    
    # Store atomic adjacency matrix in graph
    G.graph["atomic_adj_mat"] = np.nan_to_num(t_distmat)

    # Get node IDs from non NaN entries in the thresholded distance matrix and add the edge to the graph
    inds = zip(*np.where(~np.isnan(t_distmat)))

    for i in inds:
        length = t_distmat[i[0]][i[1]]
        node_1 = G.graph["pdb_df"]["node_id"][i[0]]
        node_2 = G.graph["pdb_df"]["node_id"][i[1]]
        chain_1 = G.graph["pdb_df"]["chain_id"][i[0]]
        chain_2 = G.graph["pdb_df"]["chain_id"][i[1]]

        # Check nodes are in graph
        if not (G.has_node(node_1) and G.has_node(node_2)):
            continue

        # Check atoms are in the same chain
        if not (chain_1 and chain_2):
            continue

        if G.has_edge(node_1, node_2):
            G.edges[node_1, node_2]["kind"].add("covalent")
            G.edges[node_1, node_2]["bond_length"] = length
        else:
            G.add_edge(node_1, node_2, kind={"covalent"}, bond_length=length)
    
    inds = zip(*np.where(~np.isnan(t_distmat_longrange)))
    for i in inds:
        length = t_distmat[i[0]][i[1]]
        node_1 = G.graph["pdb_df"]["node_id"][i[0]]
        node_2 = G.graph["pdb_df"]["node_id"][i[1]]
        chain_1 = G.graph["pdb_df"]["chain_id"][i[0]]
        chain_2 = G.graph["pdb_df"]["chain_id"][i[1]]

        # Check nodes are in graph
        if not (G.has_node(node_1) and G.has_node(node_2)):
            continue

        # Check atoms are in the same chain
        if chain_1 != chain_2:

            if G.has_edge(node_1, node_2):
                G.edges[node_1, node_2]["kind"].add("long")
                G.edges[node_1, node_2]["bond_length"] = length
            else:
                G.add_edge(node_1, node_2, kind={"long"}, bond_length=length)

    # Todo checking degree against MAX_NEIGHBOURS

    return G