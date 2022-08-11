import os
import re
import shutil
import torch
from torch.nn.functional import one_hot
from tqdm import tqdm
#import pymol
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


def mutate_point(pdb_id: str, mutation_id: str, pdb_dir:str = PDB_DIR, destination_dir:str = RAW_DATASET_DIR):
    """
    Creates a pointmutation in a PDB Structure and saves it in the destionation folder.
    pdb_id: str
        PDB Id of the protein
    mutation: str
        3-digit code of the mutation
    pdb_dir: str
        Directory containing the PDB files.
    destination_dir: str
        Destionation directory in which the mutated PDB structure will be saved.
    """
    chain_id, mutation = mutation_id.split(':')
    base, resid, mut = re.split('(\d+)', mutation)
    selection = "chain "+ chain_id + " and residue " + resid
    pdb_file = os.path.join(os.getcwd(), pdb_dir, pdb_id.lower() + ".pdb")

    if os.path.isfile(pdb_file):
        print("Starting  mutation wizard...")
        mutant = AMINO_CODES[mut]

        pymol.finish_launching()
    	
        pymol.cmd.wizard("mutagenesis")
        pymol.cmd.load(pdb_file)
        pymol.cmd.refresh_wizard()
        print("Selecting ", selection )
        pymol.cmd.get_wizard().do_select(selection)
        print("Mutant:", mutant)
        pymol.cmd.get_wizard().set_mode(mutant)

        pymol.cmd.get_wizard().apply()
        
        pymol.cmd.set_wizard()

        save_path = os.path.join(destination_dir, pdb_id + "_" +chain_id + "_" + mutation + ".pdb")

        pymol.cmd.save(save_path, format="pdb")
        pymol.cmd.delete(name="all")
        pymol.cmd.refresh_wizard()
        pymol.cmd.refresh()
        pymol.finish_launching()
     
    else:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), pdb_file)


def mutate_multiple_points(pdb_id:str, mutations:list, pdb_dir:str=PDB_DIR, destination_dir: str=RAW_DATASET_DIR) -> None:
    """qst_mot_let.pdf
    Mutates multiple residues in a protein

    pdb_id: str
        The 4 letter PDB ID of the protein.
    mutations: list
        A list caontaining the IDs of the mutations.
    pdb_dir: str
        Directory of the raw PDB Files.
    destination_dit: str
        The mutated structures will be saved here.
    """
    pdb_file = os.path.join(os.getcwd(), pdb_dir, pdb_id.lower() + ".pdb")
    save_path = os.path.join(destination_dir, pdb_id)

    if os.path.isfile(pdb_file):

        for mutation in mutations:

            selection = mutation[0] + "/" + mutation[1:-1]  + "/"    
            mutant = AMINO_CODES[mutation[-1]]
            pymol.finish_launching()
            pymol.cmd.wizard("mutagenesis")
            pymol.cmd.load(pdb_file)
            pymol.cmd.refresh_wizard()
            pymol.cmd.get_wizard().do_select(selection)
            pymol.cmd.get_wizard().set_mode(mutant)
            pymol.cmd.get_wizard().apply()
            pymol.cmd.set_wizard()
            pymol.cmd.deselect()
            save_path += "_" + mutation

        save_path += ".pdb"
        pymol.cmd.save(save_path, format="pdb")
        pymol.cmd.delete(name="all")
        pymol.finish_launching()
        
    else:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), pdb_file)

def create_mutations_pyrosetta(index_df:pd.DataFrame):

    for _, row in tqdm(index_df.iterrows()):

            pdb_id = row["pdb_id"]
            m_res_num = row["res_renum"]
            chain_id, mutation = row["mut_id"].split(':')
            base, resid, mut = re.split('(\d+)', mutation)

            filepath = os.path.join(raw_dir, pdb_id+ "_" +chain_id +"_"+ mutation +".pdb")

            if  not os.path.isfile(filepath):
                if pdb_id not in ["1mlc", "1vfb", "1yy9", "1ak4", "1ktz", "1n8z"]:
                    print(pdb_id, mutation)
                    pose = pose_from_pdb(os.path.join(PDB_DIR, pdb_id+".pdb"));
                    toolbox.mutants.mutate_residue(pose, m_res_num , mut)
                    pose.dump_pdb(filepath)

            clear_output(wait=True)

def create_mutations_pymol(index_df:pd.DataFrame, raw_dir:str):
    
    for _, row in tqdm(index_df.iterrows()):

            pdb_id = row["pdb_id"]
            m_res_num = row["res_renum"]
            chain_id, mutation = row["mut_id"].split(':')
            base, resid, mut = re.split('(\d+)', mutation)

            filepath = os.path.join(raw_dir, pdb_id+ "_" +chain_id +"_"+ mutation +".pdb")

            if  not os.path.isfile(filepath):
                
                print(pdb_id, mutation)
                mutate_point(pdb_id, row["mut_id"])
                    

            clear_output(wait=True)

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