import re
import os


import pandas as pd

import torch
from torch_geometric.data import Data


from biopandas.pdb import PandasPdb

from tqdm import tqdm
from IPython.display import clear_output

from graphein.protein.config import ProteinGraphConfig
from graphein.protein.graphs import construct_graph
from graphein.ml.conversion import GraphFormatConvertor
from graphein.protein.edges.atomic import add_atomic_edges, add_bond_order, add_ring_status
from graphein.protein.edges.distance import add_hydrogen_bond_interactions, add_ionic_interactions, add_peptide_bonds
from graphein.protein.edges.distance import compute_distmat



from src.utils import *
from src.mutagenesis import create_mutations_pymol
import argparse
import logging
import sys
import threading
import time
#Logging
logger = logging.getLogger("Dataset creation logger")
handler = logging.StreamHandler(sys.stdout)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)





def make_dataset(index_xlsx: str,root:str):

    raw_dir = os.path.join(root, "raw")
    processed_dir = os.path.join(root, "processed")

    index_df = pd.read_excel(index_xlsx, converters={"pdb_id":str.lower, "mut_id":str.strip, "ddg":float, "res_renum":int})
    pdbids = index_df.pdb_id.unique()
    ddg = index_df.ddg.to_list()

    #fetch the pdbs if needed
    
    fetch_pdbs(pdbids, PDB_DIR, force = False)


    #create mutated structures

    create_mutations_pymol(index_df, raw_dir)

    params_to_change = {"granularity": "atom", "edge_construction_functions": [add_atomic_edges, add_bond_order, add_ring_status, add_hydrogen_bond_interactions, add_ionic_interactions, add_peptide_bonds]}

    config = ProteinGraphConfig(**params_to_change)
    
    format_convertor = GraphFormatConvertor('nx', 'pyg', verbose="default")

    for index, row in tqdm(index_df.iterrows()):
        if not os.path.isfile(os.path.join(processed_dir,str(index)+"_mutated.pt" )):
           
            pdb_id = row["pdb_id"]
            chain_id, mutation = row["mut_id"].split(':')
            base, resid, mut = re.split('(\d+)', mutation)
            
            
            file_mut = pdb_id + "_" + chain_id + "_" + mutation

            
            print("Protein: ", pdb_id, "Mutation: ", mutation)
            
            pdb_mutated = pdb_to_df(file_mut, raw_dir)
            pdb_non_mutated = pdb_to_df(pdb_id, PDB_DIR)
        
            
            print("Extracting relevant residues")
            print(chain_id, resid)

            interface_mutated = find_relevant(chain_id, int(resid), pdb_mutated,cutout=2, cutoff=12.)
            interface_non_mutated = find_relevant(chain_id, int(resid), pdb_non_mutated,cutout=2,cutoff = 12.)
            
            path_interface_mutated = os.path.join(raw_dir, "temp", str(index)+"_mutated_interface.pdb")
            path_interface_non_mutated = os.path.join(raw_dir,"temp",  str(index)+"_non_mutated_interface.pdb")

            save_to_pdb(interface_mutated, path_interface_mutated)
            save_to_pdb(interface_non_mutated, path_interface_non_mutated)

            graph_mutated = construct_graph(config=config,pdb_path=path_interface_mutated)
            graph_non_mutated = construct_graph(config=config,pdb_path=path_interface_non_mutated)
            
            dist_mat_mut = compute_distmat(graph_mutated.graph["pdb_df"]).to_numpy()
            dist_mat_non_mut  = compute_distmat(graph_non_mutated.graph["pdb_df"]).to_numpy()

            pyg_graph_mutated = format_convertor(graph_mutated)
            pyg_graph_non_mutated = format_convertor(graph_non_mutated)

            pyg_graph_mutated.y = ddg[index]
            pyg_graph_mutated.coords = torch.FloatTensor(pyg_graph_mutated.coords[0])

            pyg_graph_non_mutated.y = ddg[index]
            pyg_graph_non_mutated.coords = torch.FloatTensor(pyg_graph_non_mutated.coords[0])

           

            ei_mut = pyg_graph_mutated.edge_index
            ei_non_mut = pyg_graph_non_mutated.edge_index

            edge_weights_mut = []
            edge_weights_non_mut = []
            
            for i in range(ei_mut.shape[1]):

                edge_weights_mut.append(1/dist_mat_mut[ei_mut[0, i].item(), ei_mut[1, i].item()])

            for i in range(ei_non_mut.shape[1]):

                edge_weights_non_mut.append(1/dist_mat_non_mut[ei_non_mut[0, i].item(), ei_non_mut[1, i].item()])

            

            
            if pyg_graph_mutated.coords.shape[0] == len(pyg_graph_mutated.node_id):
                pass
            else:
                break
        
            
            
            if pyg_graph_non_mutated.coords.shape[0] == len(pyg_graph_non_mutated.node_id):
                mut = Data(x=node_id_to_feature_matrix(graph_mutated), edge_index=pyg_graph_mutated.edge_index, edge_weights=torch.tensor(edge_weights_mut))
               
                non_mut = Data(x=node_id_to_feature_matrix(graph_non_mutated), edge_index=pyg_graph_non_mutated.edge_index, edge_weights=torch.tensor(edge_weights_non_mut))

                torch.save(mut, os.path.join(processed_dir, str(index)+"_mutated.pt"))
                torch.save(non_mut, os.path.join(processed_dir, str(index)+"_non_mutated.pt"))

            else:
                pass
            clear_output(wait=True)


def make_dataset_threaded(index_xlsx: str,root:str, chunk_size=1):

    raw_dir = os.path.join(root, "raw")
    processed_dir = os.path.join(root, "processed")

    index_df = pd.read_excel(index_xlsx, converters={"pdb_id":str.lower, "mut_pdb":str.strip,"mut_id":str.strip,"aff_mut":float, "aff_wt":float,"temp":float,  "ddg":float,})
    


    index_df_chunks = np.array_split(index_df, chunk_size)
    ddg = index_df.ddg.to_list()
    

    exclude_indices = []
    print("Mutations...")

    start = time.time()
    
    create_mutations_pymol(index_df=index_df, raw_dir=raw_dir)
        

    end = time.time()
    delta = end-start
    print("Time needed for creating the mutations: {}s".format(delta.total_seconds()))

    params_to_change = {"granularity": "atom", "edge_construction_functions": [add_atomic_edges, add_bond_order, add_ring_status, add_hydrogen_bond_interactions, add_ionic_interactions, add_peptide_bonds]}
    config = ProteinGraphConfig(**params_to_change)
    format_convertor = GraphFormatConvertor('nx', 'pyg', verbose="default")
    processes = []

    start = time.time()
    for i, chunk in enumerate(index_df_chunks):
        print("Processing Threads {}-{}".format(i, i+chunk_size))
        threads = []
        for index, row in tqdm(chunk.iterrows()):
            ind = i*chunk_size + index
            if not os.path.isfile(os.path.join(processed_dir,str(ind)+"_mutated.pt" )):
                threads.append(threading.Thread(target = extract_and_save, args=(row, ind, raw_dir, processed_dir, config, format_convertor)))

        for p in threads:
            p.start()

        for p in threads:
            p.join()
    end = time.time()
    delta = start-end
    print("Dataset creation finished. Time needed: {}s".format(delta.total_seconds()))
        
        
           
            
            
            
            

def extract_and_save(df_row, index, raw_dir,processed_dir, config, format_convertor):


    
    pdb_id = df_row["pdb_id"]
    mutation = df_row["mut_id"].split(",")

    if len(mutation)>1:
        return
        
    mutation = mutation[0]
    wildtype, chain_id, resid, mut_target = mutation[0], mutation[1], mutation[1:-1], mutation[-1]
    
    
    ddg = df_row["ddg"]
    file_mut = pdb_id + "_" + mutation

            
    print("Protein: ", pdb_id, "Mutation: ", mut_target)
    
    pdb_mutated = pdb_to_df(file_mut, raw_dir)
    pdb_non_mutated = pdb_to_df(pdb_id, PDB_DIR)

    
    print("Extracting relevant residues")
    print(chain_id, resid)

    interface_mutated = find_relevant(chain_id, int(resid), pdb_mutated,cutout=2, cutoff=12.)
    interface_non_mutated = find_relevant(chain_id, int(resid), pdb_non_mutated,cutout=2,cutoff = 12.)
    
    path_interface_mutated = os.path.join(raw_dir, "temp", str(index)+"_mutated_interface.pdb")
    path_interface_non_mutated = os.path.join(raw_dir,"temp",  str(index)+"_non_mutated_interface.pdb")

    save_to_pdb(interface_mutated, path_interface_mutated)
    save_to_pdb(interface_non_mutated, path_interface_non_mutated)

    graph_mutated = construct_graph(config=config,pdb_path=path_interface_mutated)
    graph_non_mutated = construct_graph(config=config,pdb_path=path_interface_non_mutated)
    
    dist_mat_mut = compute_distmat(graph_mutated.graph["pdb_df"]).to_numpy()
    dist_mat_non_mut  = compute_distmat(graph_non_mutated.graph["pdb_df"]).to_numpy()

    pyg_graph_mutated = format_convertor(graph_mutated)
    pyg_graph_non_mutated = format_convertor(graph_non_mutated)

    pyg_graph_mutated.y = ddg[index]
    pyg_graph_mutated.coords = torch.FloatTensor(pyg_graph_mutated.coords[0])

    pyg_graph_non_mutated.y = ddg[index]
    pyg_graph_non_mutated.coords = torch.FloatTensor(pyg_graph_non_mutated.coords[0])

    

    ei_mut = pyg_graph_mutated.edge_index
    ei_non_mut = pyg_graph_non_mutated.edge_index

    edge_weights_mut = []
    edge_weights_non_mut = []
    
    for i in range(ei_mut.shape[1]):

        edge_weights_mut.append(1/dist_mat_mut[ei_mut[0, i].item(), ei_mut[1, i].item()])

    for i in range(ei_non_mut.shape[1]):

        edge_weights_non_mut.append(1/dist_mat_non_mut[ei_non_mut[0, i].item(), ei_non_mut[1, i].item()])

    

    
    if pyg_graph_mutated.coords.shape[0] == len(pyg_graph_mutated.node_id):
        pass
    else:
        raise ValueError

    
    
    if pyg_graph_non_mutated.coords.shape[0] == len(pyg_graph_non_mutated.node_id):
        mut = Data(x=node_id_to_feature_matrix(graph_mutated), edge_index=pyg_graph_mutated.edge_index, edge_weights=torch.tensor(edge_weights_mut))
        
        non_mut = Data(x=node_id_to_feature_matrix(graph_non_mutated), edge_index=pyg_graph_non_mutated.edge_index, edge_weights=torch.tensor(edge_weights_non_mut))

        torch.save(mut, os.path.join(processed_dir, str(index)+"_mutated.pt"))
        torch.save(non_mut, os.path.join(processed_dir, str(index)+"_non_mutated.pt"))

    else:
        pass
    clear_output(wait=True)
    


            
