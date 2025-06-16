"""
===============================================================================
Title:      Network
Outline:    Network class to handle the protein-protein interaction networks
            from different databases. It allows to get species, interactors, 
            add negatives interactions per species and merge different networks.
Author:     Alejandro Sánchez Cano
Date:       15/10/2024
===============================================================================
"""

# Third-party modules
import pandas as pd
from tqdm import tqdm

# Custom modules
from src.misc import path
from src.misc.logger import logger
from src.entities.interactor import Interactor

class Network:

    def __init__(
            self,
            db: str = None,
            version: str = None,
            type: str = None, 
            standarized: bool = True
            ):

        standarized = '_standarized' if standarized else ''
        version = f'_{version}' if version else ''
        filepath = path.NETWORKS /f'{db}{version}_{type}{standarized}.tsv'
        self.df = pd.read_csv(filepath, sep = '\t') if db and type else None

    def __repr__(self) -> str:
        return self.df.__repr__()

    @property
    def species(self) -> list[int]:
        '''	
        Returns a list of unique species in the network.

        Returns
        -------
        list[int]
            List of unique species in the network.
        '''
        speciesA = self.df['Species_A'].unique()
        speciesB = self.df['Species_B'].unique()
        return list(set(speciesA) | set(speciesB))
    
    def interactors(self, df: pd.DataFrame = None) -> list[Interactor]:
        '''
        Returns a list of interactors in a network.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to extract the interactors from. If None, the 
            network dataframe is used.

        Returns
        -------
        list[Interactor]
            List of interactors in the network.
        '''
        df = self.df if df is None else df
        unique_interactors = set(df['A']) | set(df['B'])
        return [Interactor(uniprot_id) for uniprot_id in unique_interactors]

    def add_negatives_per_species(self) -> None:
        '''
        Add negatives interactions per species to the network. We assume
        that the set of interactors of a species that are not listed as 
        positive interactions are negatives interactions.
        '''
        # Initialize dictionary to store negatives per species
        df_negatives = {
            'A': [],
            'B': [],
            'A=B': [],
            'Species_A': [],
            'Species_B': [],
            'Seq_A': [],
            'Seq_B': [],
            'Seq': []
        }

        # Get interactors per species
        for species in tqdm(self.species):
            df = self.df[self.df.apply(lambda x: x['Species_A'] == species and x['Species_B'] == species, axis = 1)]
            interactors = self.interactors(df)
            # Add negatives
            for int1 in interactors:
                for int2 in interactors:
                    name = '-'.join(sorted([int1.uniprot_id, int2.uniprot_id]))
                    if name not in df['A=B']:
                        df_negatives['A'].append(int1.uniprot_id)
                        df_negatives['B'].append(int2.uniprot_id)
                        df_negatives['A=B'].append(f'{int1.uniprot_id}={int2.uniprot_id}')
                        df_negatives['Species_A'].append(species)
                        df_negatives['Species_B'].append(species)
                        df_negatives['Seq_A'].append(int1.seq)
                        df_negatives['Seq_B'].append(int2.seq)
                        df_negatives['Seq'].append(f'{int1.seq}:{int2.seq}')
        
        # Add interaction column
        self.df['Interaction'] = 1
        df_negatives = pd.DataFrame(df_negatives)
        df_negatives['Interaction'] = 0

        # Add negatives to the dataframe
        df = pd.concat([self.df, df_negatives], ignore_index = True)

        # Logging
        logger.info(f'{len(df_negatives)} negatives added to the network dim({self.df.shape}) -> dim({df.shape})')
        
        # Update network dataframe attribute
        self.df = df

    @classmethod
    def merge(cls, *networks: 'Network') -> 'Network':
        '''
        Merge different networks into a new network. This is intended to
        merge the networks of the BioGRID, IntAct, and PlaPPIsite 
        databases.

        Parameters
        ----------
        networks : Network
            List of networks to merge.

        Returns
        -------
        Network
            Merged network.
        '''

        # Merge networks
        network = cls()
        df = pd.concat([network.df for network in networks], ignore_index = True)
        network.df = df.drop_duplicates('A=B')
        network.df = network.df.drop_duplicates('Seq')

        # Logging
        # logger.info(f'Networks merged into a new network dim({network.df.shape})')
        # logger.info(f'Positive interactions: {network.df[network.df["Interaction"] == 1].shape[0]}')
        # logger.info(f'Negative interactions: {network.df[network.df["Interaction"] == 0].shape[0]}')

        return network
    
if __name__ == '__main__':
    '''Test class'''	
    network = Network(db='BioGRID', version='4.4.238', type='MADS_vs_MADS', standarized=True)
    print(network)