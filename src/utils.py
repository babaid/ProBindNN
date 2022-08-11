import os
import re
import shutil
import torch
from torch.nn.functional import one_hot
from tqdm import tqdm
import pandas as pd
import numpy as np
from IPython.display import clear_output
from biopandas.pdb import PandasPdb
import errno
import glob
import shutil
from mendeleev import element

import multiprocessing as mp
AMINO_CODES = {
    'A': 'Ala', 'R': 'Arg', 'N': 'Asn', 'D': 'Asp', 'C': 'Cys', 'Q': 'Gln', 'E': 'Glu', 
    'G': 'Gly', 'H': 'His', 'I': 'Ile', 'L': 'Leu', 'K': 'Lys', 'M': 'Met', 'F': 'Phe', 
    'P': 'Pro', 'O': 'Pyl', 'S': 'Ser', 'U': 'Sec', 'T': 'Thr', 'W': 'Trp', 'Y': 'Tyr', 
    'V': 'Val', 'B': 'Asx', 'Z': 'Glx', 'X': 'Xaa', 'J': 'Xle'}
AMINO_CODES_R = dict((v.upper(), k) for k,v in AMINO_CODES.items())
COORDINATE_NAMES = ["x_coord", "y_coord", "z_coord"]
ELEMENTS = {'C':0, 'N':1, 'O':2, 'S':3, 'F':4, 'P':5, 'Cl':6, 'B':7, 'H':8}

AMINO_ACIDS = dict((v.upper(), i) for i, (k, v) in enumerate(AMINO_CODES.items()))
CHAIN_IDS = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9, 'K': 10, 'L': 11, 'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18, 'T': 19, 'U':20, 'V':21, 'W':22, 'X':23, 'Y':24, 'Z':25}
PDB_DIR = "PDBs"
RAW_DATASET_DIR="dataset/raw"




def pdb_to_df(pdb_id:str, root:str)->pd.DataFrame:
    """
    Opens a PDB file as a Dataframe.
    id: str
        Identificator of the pdb file without the .pdb extension
    root: str
        Root folder of the PDB File
    """
    path = os.path.join(root, pdb_id+".pdb")

    if os.path.isfile(path):
        return PandasPdb().read_pdb(path).df["ATOM"]
    else:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)





def save_to_pdb(structure:pd.DataFrame, path:str)->None:
    """
    Save extracted segments to a pdb file
    """
    pdb_saver = PandasPdb()
    pdb_saver.df["ATOM"] = structure
    pdb_saver.to_pdb(path, records = ["ATOM"])


def fetch_pdbs(pdbids: list, pdb_dir: str, force=False)->None:
    """
    Downloades the PDB Files from the PDB.
    """
    for pdbid in tqdm(pdbids):
        if pdbid is None or pdbid=="-":
            print("Skipping empty pdbid")
            continue
        print("Downloading ", pdbid)
        destination_dir = os.path.join(pdb_dir, pdbid+".pdb")
        if not os.path.isfile(destination_dir) or force:
            pdb = PandasPdb().fetch_pdb(pdbid, source="pdb")
            pdb.to_pdb(destination_dir, records=["ATOM"])
            print("Successfully donwoaded")
            clear_output(wait=True)    
        else:
            clear_output(wait=True)


def find_relevant(chid:str, res_n:int, structure: pd.DataFrame, cutoff:float = 30., cutout:int = 5)->pd.DataFrame:
    """
    Extracts the relevant residues from a trcuture.
    chid: str
        Chain ID of the chain where the mutation occurs
    res_n: int
        Residue number of the residue which mutates
    structure: pd.DataFrame
        The DataFrame of the structure
    cutoff: float
        Maximal cutoff distance from the central mutated residue
    cutout: int
        The number of neighboring residues taken into account
    """
    
    base_res = structure.loc[structure["chain_id"] == chid]
    base_res = base_res.loc[base_res["residue_number"] == res_n]
    
    base_pos = np.array([sum(base_res["x_coord"].to_list()), sum(base_res["y_coord"].to_list()), sum(base_res["z_coord"].to_list())])/len(base_res)
    
    relevant = []

    
    for chain in structure["chain_id"].unique():
        curr_chain = structure.loc[structure["chain_id"]==chain]
        for res in curr_chain["residue_number"].unique():
            residue = curr_chain.loc[curr_chain["residue_number"]==res]
            
            res_pos = np.array([sum(residue["x_coord"].to_list()), sum(residue["y_coord"].to_list()), sum(residue["z_coord"].to_list())])/len(base_res)
            dist = np.linalg.norm(res_pos-base_pos)
            if dist<cutoff:
                relevant.append([chain, res, dist])
            
    dist_df = pd.DataFrame(relevant, columns=["chain_id", "residue_number", "distance"])
   
    middle = []
    
    for chain in dist_df.chain_id.unique():
        mid = dist_df.loc[dist_df["chain_id"]==chain]
        mid = mid.loc[mid["distance"] == mid.distance.min()]
        middle.append([mid["chain_id"].values[0], mid["residue_number"].values[0]])
    
    
    
    rel = pd.DataFrame(middle, columns=["chain_id", "residue_number"])
    
    parts = []
    for index, row in rel.iterrows():
        start = row["residue_number"]-cutout
        end = row["residue_number"]+cutout
        
        while start<0:
            start+=1
        while end>structure["residue_number"].max():
            end-=1

        for i in range(start, end+1):
            parts.append(structure.loc[structure["chain_id"] == row["chain_id"]].loc[structure["residue_number"] == i])

    return pd.concat(parts)




def copy_pdbs(src, dst):
    files = glob.iglob(os.path.join(src, "*.pdb")) 
    for file in files:
        if os.path.isfile(file) and file[-10] != ".":
            shutil.copy2(file, dst)
            
    for file in os.listdir(dst):
        os.rename(os.path.join("PDBs", file), os.path.join("PDBs", file.lower()))
        
def mutid_to_poseid(pdb_id:str, chain_id:str, residue_number:int, root:str="PDBs"):
    structure = pdb_to_df(pdb_id, root)
    chain_len = {}
    for ch in structure.chain_id.unique():
        x = structure.loc[structure.chain_id == ch]
        l = len(x.residue_number.unique())
        chain_len[ch] = l
    pid = residue_number
    for k, v in chain_len.items():
        if k == chain_id:
            break
        pid+=v
    return pid
        
def renumber(index_path:str):
    index_df = pd.read_excel(index_path, converters={"pdb_id":str.strip, "mut_id":str.strip, "ddg": float})
    for index, row in tqdm(index_df.iterrows()):
        chain_id, mut_id = row["mut_id"].split(":")
        base, resid, mutation = re.split('(\d+)', mut_id)

        index_df.at[index, "res_renum"] = mutid_to_poseid(row["pdb_id"], chain_id, int(resid))
    index_df.to_excel("renumbered_index.xlsx")


def node_id_to_feature_matrix(x):
    features = []

    for node in x.nodes:
        chain, res, res_num, atom = node.split(":")
        atom = element(atom[0])
        elem = x.nodes[node]
        feature = torch.concat([one_hot(torch.tensor(ELEMENTS[atom.symbol]), num_classes=9),
                            
                            torch.tensor([atom.electronegativity()]),
                            torch.tensor([atom.nvalence()])])
        features.append(feature.unsqueeze(0))
    
    return torch.concat(features)
def nx_features(g):
    features = []
    for node in list(g.nodes):
        elem = g.nodes[node]
        print(elem)
        atom = element(elem["element_symbol"])
        feature = torch.concat([one_hot(torch.tensor(ELEMENTS[elem["element_symbol"]]), num_classes=9),
                            one_hot(torch.tensor(CHAIN_IDS[elem["chain_id"]]), num_classes=9).float(),
                            torch.tensor([atom.electronegativity()]),
                            torch.tensor([atom.nvalence()]),
                            torch.tensor([elem["b_factor"]]),
                            #torch.tensor([m for m in elem["meiler"]])
                            ])
        features.append(feature.unsqueeze(0))
    return torch.concat(features)