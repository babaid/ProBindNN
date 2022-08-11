import os
import re
import errno
import pymol
from tqdm import tqdm
from IPython.display import clear_output
import pandas as pd
PDB_DIR = "PDBs"
RAW_DATASET_DIR="dataset/raw"
AMINO_CODES = {
    'A': 'Ala', 'R': 'Arg', 'N': 'Asn', 'D': 'Asp', 'C': 'Cys', 'Q': 'Gln', 'E': 'Glu', 
    'G': 'Gly', 'H': 'His', 'I': 'Ile', 'L': 'Leu', 'K': 'Lys', 'M': 'Met', 'F': 'Phe', 
    'P': 'Pro', 'O': 'Pyl', 'S': 'Ser', 'U': 'Sec', 'T': 'Thr', 'W': 'Trp', 'Y': 'Tyr', 
    'V': 'Val', 'B': 'Asx', 'Z': 'Glx', 'X': 'Xaa', 'J': 'Xle'}



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
    
    wildtype, chain_id, resid, mut = mutation_id[0], mutation_id[1], mutation_id[2:-1], mutation_id[-1]

    selection = "chain "+ chain_id + " and residue " + resid

    pdb_file = os.path.join(os.getcwd(), pdb_dir, pdb_id.lower() + ".pdb")
    print("Selector:",selection, "--------------------------")
    
   
    if os.path.isfile(pdb_file):

        print("Starting  mutation wizard...")
        mutant = AMINO_CODES[mut]

       
    	
        pymol.cmd.wizard("mutagenesis")
        pymol.cmd.load(pdb_file)
        pymol.cmd.refresh_wizard()
        print("Selecting ", selection )
        pymol.cmd.get_wizard().do_select(selection)
        print("Mutant:", mutant)
        pymol.cmd.get_wizard().set_mode(mutant)

        pymol.cmd.get_wizard().apply()
        
        pymol.cmd.set_wizard()

        save_path = os.path.join(destination_dir, pdb_id + "_" + mutation_id + ".pdb")

        pymol.cmd.save(save_path, format="pdb")
        pymol.cmd.delete(name="all")
        pymol.cmd.refresh_wizard()
        pymol.cmd.refresh()
        
     
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
        
        
    else:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), pdb_file)

def create_mutations_pymol(index_df:pd.DataFrame, raw_dir:str):
    
    for _, row in tqdm(index_df.iterrows()):


        pdb_id = row["pdb_id"]
        mutation = row["mut_id"].split(",")

        if len(mutation)>1:
            continue
       
        else:
            mutation = mutation[0]
            pdb_id = row["pdb_id"].split('_')[0]
            wildtype, chain_id, resid, mut_target = mutation[0], mutation[1], mutation[2:-1], mutation[-1]
            
            filepath = os.path.join(raw_dir, pdb_id+ "_" + mutation +".pdb")

            if  not os.path.isfile(filepath):
                print(pdb_id, mut_target)
                mutate_point(pdb_id, row["mut_id"])
            clear_output(wait=True)