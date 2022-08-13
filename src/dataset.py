"""
The script containing our dataset which consists of molecular graphs,
"""


import os
from typing import Callable, Optional, Union
from torch_geometric.data import Dataset
import pandas as pd
import numpy as np

import torch
from torch_geometric.data import Data


from src.utils import *



class MutationDataset(Dataset):
    """
    Dataset of mutated structures.
    """

    def __init__(self, root: Optional[str] = None, index_xlsx: Optional[str]=None,  transform: Optional[Callable] = None, pre_transform: Optional[Callable] = None, pre_filter: Optional[Callable] = None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.mut_df = None
        self.index_xlsx = index_xlsx


        self.mut_df = pd.read_excel(index_xlsx, converters={"pdb_id":str.lower, "mut_pdb":str.strip,"mut_id":str.strip,"aff_mut":float, "aff_wt":float,"temp":float,  "ddg":float,})
        #self.mut_df = self.mut_df.loc[self.mut_df["mut_id"].str.len() == 1]
        
        self.non_mut_files = [x for x in os.listdir(os.path.join(self.processed_dir, "non_mutated")) if x.endswith(".pt")]
        self.mut_files = [x for x in os.listdir(os.path.join(self.processed_dir, "mutated")) if x.endswith(".pt")]

        
        


    @property
    def raw_file_names(self):
        """
        Collects and returns the raw filenames of the dataset.
        index_xlsx: str
            The name of the XLSX file containing the information about the data available. It has the format:
            pdbid: str | wildtype_aa: str | chain_id: str | residue_number: str | mutation: str | ddg: float

            pdbid: protein identifier
            chain_id: chain where the mutation occurs
            residue_number: the segment which mutates
            muatation: one-letter code of the muatation
            ddg: experimental ddG value
        """
        
        path = os.path.join(self.root, self.index_xlsx)
        filenames = [x for x in os.listdir(self.raw_dir) if x.endswith(".pdb")]
        return filenames

    @property
    def processed_file_names(self):
        """
        Collects and returns the processed filenames of the dataset.
        """
        filenames = [x for x in os.listdir(os.path.join(self.processed_dir, "mutated")) if x.endswith(".pt")]

        return filenames


    def len(self) -> int:
        return len(self.processed_file_names)
    
    def get(self, idx: Union[int, np.integer]) -> Union['Dataset', Data]:

        mutated = torch.load(os.path.join(self.processed_dir, "mutated", self.mut_files[idx]))
        non_mutated = torch.load(os.path.join(self.processed_dir, "non_mutated", self.non_mut_files[idx]))
        data = {"mutated": mutated,"non_mutated": non_mutated}
        return data

    def __cat_dim__(self, key, value, *args, **kwargs):
         if key == 'x' or key=='edge_index':
             return None
         else:
             return super().__cat_dim__(key, value, *args, **kwargs)