import os
import time

from tqdm import tqdm
from IPython.display import clear_output

import pandas as pd
import torch
from torch_geometric.data import Data
from biopandas.pdb import PandasPdb



from graphein.protein.config import ProteinGraphConfig
from graphein.protein.graphs import construct_graph
from graphein.ml.conversion import GraphFormatConvertor
from graphein.protein.edges.atomic import add_bond_order, add_ring_status
from graphein.protein.edges.distance import add_hydrogen_bond_interactions, add_ionic_interactions, add_peptide_bonds



from utils import *
from mutagenesis import create_mutations_pymol
from new_atomic import add_atomic_edges




def make_dataset(index_xlsx: str,root:str, pdb_dir):

    raw_dir = os.path.join(root, "raw")
    processed_dir = os.path.join(root, "processed")
    
    index_df = pd.read_excel(index_xlsx, converters={"pdb_id":str.lower, "mut_pdb":str.strip,"mut_id":str.strip,"aff_mut":float, "aff_wt":float,"temp":float,  "ddg":float,})
    
    
    print("Mutations...")

    start = time.time()
    
    create_mutations_pymol(index_df=index_df, raw_dir=raw_dir)

    end = time.time()

    delta =(end-start)
    print("Time needed for creating the mutations: {}s".format(delta))

    params_to_change = {"granularity": "atom", "edge_construction_functions": [add_atomic_edges, add_bond_order, add_ring_status, add_hydrogen_bond_interactions, add_ionic_interactions, add_peptide_bonds]}

    config = ProteinGraphConfig(**params_to_change)
    
    format_convertor = GraphFormatConvertor('nx', 'pyg', verbose="default")

    for index, row in tqdm(index_df.iterrows()):
        if not (os.path.isfile(os.path.join(processed_dir,"mutated",str(index)+".pt" )) or os.path.isfile(os.path.join(processed_dir,"mutated",str(index)+".pt" ))):
            
            pdb_id = row["pdb_id"].split('_')[0]
            mutation = row["mut_id"]

            wildtype, chain_id, resid, mut_target = mutation[0], mutation[1], mutation[2:-1], mutation[-1]
            
            
            ddg = row["ddg"]
            file_mut = pdb_id + "_" + mutation

            
            
            file_mut = pdb_id + "_" + mutation

            
            print("Protein: ", pdb_id, "Mutation: ", mutation)
            
            pdb_mutated = pdb_to_df(file_mut, raw_dir)
            pdb_non_mutated = pdb_to_df(pdb_id, pdb_dir)

            path_interface_mutated = os.path.join(raw_dir, "temp", str(index)+"_mutated_interface.pdb")
            path_interface_non_mutated = os.path.join(raw_dir,"temp",  str(index)+"_non_mutated_interface.pdb")
            
            print("Extracting relevant residues")
            print(chain_id, resid)
            if not os.path.isfile(path_interface_mutated):

                interface_mutated = find_relevant(chain_id, int(resid), pdb_mutated,cutout=5, cutoff=12.)
                interface_non_mutated = find_relevant(chain_id, int(resid), pdb_non_mutated,cutout=5,cutoff = 12.)
                save_to_pdb(interface_mutated, path_interface_mutated)
                save_to_pdb(interface_non_mutated, path_interface_non_mutated)
            

            graph_mutated = construct_graph(config=config,pdb_path=path_interface_mutated)
            graph_non_mutated = construct_graph(config=config,pdb_path=path_interface_non_mutated)
        
            pyg_graph_mutated = format_convertor(graph_mutated)
            pyg_graph_non_mutated = format_convertor(graph_non_mutated)

            pyg_graph_mutated.y = ddg
            pyg_graph_mutated.coords = torch.FloatTensor(pyg_graph_mutated.coords[0])

            pyg_graph_non_mutated.y = ddg
            pyg_graph_non_mutated.coords = torch.FloatTensor(pyg_graph_non_mutated.coords[0])

         
            
            if pyg_graph_non_mutated.coords.shape[0] == len(pyg_graph_non_mutated.node_id):

                mut = Data(x=node_id_to_feature_matrix(graph_mutated), edge_index=pyg_graph_mutated.edge_index,edge_weights=edge_weights(graph_mutated), ddg=ddg)
                non_mut = Data(x=node_id_to_feature_matrix(graph_non_mutated), edge_index=pyg_graph_non_mutated.edge_index, edge_weights=edge_weights(graph_non_mutated))

                torch.save(mut, os.path.join(processed_dir,"mutated", str(index)+".pt"))
                torch.save(non_mut, os.path.join(processed_dir,"non_mutated",str(index)+".pt"))

            else:
                raise ValueError

            clear_output(wait=True)
    
if __name__ == "__main__":
    if not os.path.exists("dataset"):
        os.mkdir("dataset")
        os.mkdir("dataset/raw")
        os.mkdir("dataset/raw/temp")
        os.mkdir("dataset/processed")
        os.mkdir("dataset/processed/mutated")
        os.mkdir("dataset/processed/non_mutated")
    if not os.path.exists("./index.xlsx"):
        print("Error, index.xlsx not found in ./")
    make_dataset(index_xlsx="./index.xlsx", root="dataset")