"""
===============================================================================
Title:      Scoring
Outline:    Scoring class to manage Isabella's interaction data. The way it is
            handled it is extremely messy and will need refactoring in the 
            future. Some files were not considered:
            - AGL2 etc.: BD sheet is missing
            - SOC1 (ALEX): not sure what the underlying sequences are
            - SOC1 NEW I-domains: not sure what the underlying sequences are
            Interaction values are handled in the form (SOC1&AGL14, [0, 0, 1]).
            This gives the possibility to incorporate multiple ways to handle
            discrepancies in the data.
Author:     Alejandro Sánchez Cano
Date:       20/06/2025
===============================================================================
"""

# Built-in modules
import re
from typing import Literal, Generator

# Third-party modules
import numpy as np
import pandas as pd

# Custom modules
from src.misc import path
from src.misc.logger import logger
from src.sources.ledge import Ledge

class Scoring:

    ledge = Ledge(path.SCORING / 'LEDGE.xlsx')

    def __init__(self, file_stem: str):
        self.file_stem = file_stem
        self.ad = self._read_file('AD')
        self.bd = self._read_file('BD')
        self.df = None
        self.processed_df = {
            'bioID_A': [],
            'bioID_B': [],
            'UniProtID_A': [],
            'UniProtID_B': [],
            'Sequence_A': [],
            'Sequence_B': [],
            'Interaction': [],
            'From': [],
            'TaxonID_A': [],
            'TaxonID_B': [],
            'Species_A': [],
            'Species_B': []
        }

    def _read_file(self, sheet: Literal['AD', 'BD']) -> pd.DataFrame:
        '''
        Read excel file and return a dataframe with the unsplitted data.

        Parameters
        ----------
        sheet : Literal['AD', 'BD']
            Sheet to read: AD (AD targets against BD library) or BD
            (BD targets against AD library).

        Returns
        -------
        pd.DataFrame
            Dataframe with merged parts.
        '''

        # AD and BD are complementary sheets
        other = 'BD' if sheet == 'AD' else 'AD'

        # Read file
        df = pd.read_excel(
            io = (path.SCORING / self.file_stem).with_suffix('.xlsx'), 
            sheet_name = f'{sheet}+LIBRARY {other}',
            header = None, 
            index_col = 0
            )

        # Split dataframe into parts
        nan_indices = np.where(df.index.isna())[0]
        group_size = nan_indices[1] - nan_indices[0]
        dfs = [group for _, group in df.groupby(pd.RangeIndex(start = 0, stop = len(df), step = 1) // group_size)]
        # Merge parts
        df_final = pd.DataFrame()
        for _, df_part in enumerate(dfs):
            # Drop columns with all NaN (useful for last dataframe)
            df_part = df_part.dropna(axis=1, how='all')
            # Set columns and index
            df_part.columns = df_part.iloc[0]
            df_part = df_part[1:]
            df_part.index.name = sheet
            df_part.columns.name = other
            # Merge df parts
            df_final = pd.concat([df_final, df_part], axis=1)

        return df_final
    
    def process(self) -> pd.DataFrame:
        '''
        Process the AD and BD dataframes. This means extract interaction values,
        fetch sequences from the ledge (and mutate them if neccessary),
        and create a final dataframe with the processed data.

        Returns
        -------
        pd.DataFrame
            Dataframe with processed data.
        '''

        # Redundancy dictionary
        redundancy = {}

        # Iterate over rows and columns of AD and BD dataframes
        name_generator_ad = ((name_a, name_b) for name_a in self.ad.index for name_b in self.ad.columns)
        name_generator_bd = ((name_a, name_b) for name_a in self.bd.index for name_b in self.bd.columns)

        for ad, bd in zip(name_generator_ad, name_generator_bd):
            # Check if names match, and use AD if so
            assert ad[0] == bd[0], f'AD and BD names do not match: {ad[0]} vs {bd[0]}'
            assert ad[1] == bd[1], f'AD and BD names do not match: {ad[1]} vs {bd[1]}'
            index_name = ad[0]
            column_name = ad[1]

            # Interaction values
            interaction_ad = self.ad.loc[index_name, column_name]
            interaction_bd = self.bd.loc[index_name, column_name]
            if isinstance(interaction_ad, pd.Series):
                interaction_ad = interaction_ad.tolist()
                interaction_bd = interaction_bd.tolist()
            else:
                interaction_ad = [interaction_ad]
                interaction_bd = [interaction_bd]
        
            # Get sequences from ledge and manage I-swapping and deletions mutations
            if '+I-' in index_name:
                name_wt, name_mut = index_name.split('+I-')
                logger.debug(f'Fetching sequence for {name_wt} and {name_mut}')
                m = Scoring.ledge.fetch(name_wt, 'bioID', 'M')
                i = Scoring.ledge.fetch(name_mut, 'bioID', 'I')
                k = Scoring.ledge.fetch(name_wt, 'bioID', 'K')
                c = Scoring.ledge.fetch(name_wt, 'bioID', 'C')
                seq_index = f'{m}{i}{k}{c}'

            elif 'δ' in index_name:
                digits = re.findall(r'\d+', index_name.split('δ')[1])
                name_wt = index_name.split('δ')[0]
                if len(digits) == 1:
                    deletion = int(digits[0]) - 1
                    seq_index = Scoring.ledge.fetch(name_wt, 'bioID', 'Seq')
                    seq_index = seq_index[:deletion] + seq_index[deletion + 1:]
                elif len(digits) == 2:
                    deletions = [int(d) - 1 for d in digits]
                    seq_index = Scoring.ledge.fetch(name_wt, 'bioID', 'Seq')
                    seq_index = seq_index[:deletions[0]] + seq_index[deletions[1] + 1:]
            else:
                logger.debug(f'Fetching sequence for {index_name}')
                seq_index = Scoring.ledge.fetch(index_name, 'bioID', 'Seq')
            
            if '+I-' in column_name:
                name_wt, name_mut = column_name.split('+I-')
                logger.debug(f'Fetching sequence for {name_wt} and {name_mut}')
                m = Scoring.ledge.fetch(name_wt, 'bioID', 'M')
                i = Scoring.ledge.fetch(name_mut, 'bioID', 'I')
                k = Scoring.ledge.fetch(name_wt, 'bioID', 'K')
                c = Scoring.ledge.fetch(name_wt, 'bioID', 'C')
                seq_column = f'{m}{i}{k}{c}'

            elif 'δ' in column_name:
                digits = re.findall(r'\d+', index_name.split('δ')[1])
                name_wt = index_name.split('δ')[0]
                if len(digits) == 1:
                    deletion = int(digits[0]) - 1
                    seq_column = Scoring.ledge.fetch(name_wt, 'bioID', 'Seq')
                    seq_column = seq_column[:deletion] + seq_column[deletion + 1:]
                elif len(digits) == 2:
                    deletions = [int(d) - 1 for d in digits]
                    seq_column = Scoring.ledge.fetch(name_wt, 'bioID', 'Seq')
                    seq_column = seq_column[:deletions[0]] + seq_column[deletions[1] + 1:]
            else:
                logger.debug(f'Fetching sequence for {column_name}')
                seq_column = Scoring.ledge.fetch(column_name, 'bioID', 'Seq')

            # Join sequences
            seqs = '='.join(sorted([seq_index, seq_column]))
            
            # Apply the sorting to the bioIDs
            if seq_index != seqs.split('=')[0]:
                index_name, column_name = column_name, index_name

            # Check for duplicates
            if seqs not in redundancy:
                redundancy[seqs] = (interaction_ad, interaction_bd)
            else:
                # Calculate indeces to update
                idx_a = set([idx for idx, seq in enumerate(self.processed_df['Sequence_A']) if seq == seqs.split('=')[0]])
                idx_b = set([idx for idx, seq in enumerate(self.processed_df['Sequence_B']) if seq == seqs.split('=')[1]])
                idx = idx_a & idx_b
                assert len(idx) == 1, f'Index error: {idx}'
                idx = idx.pop()

                # Duplications are due to multiple interations being tested (index and columns)
                self.processed_df['Interaction'][idx][0] = (self.processed_df['Interaction'][idx][0][0], self.processed_df['Interaction'][idx][0][1] + interaction_ad + interaction_bd)


                continue
            
            # Add to final data frame
            self.processed_df['bioID_A'].append(index_name)
            self.processed_df['bioID_B'].append(column_name)
            self.processed_df['UniProtID_A'].append(Scoring.ledge.fetch(index_name, 'bioID', 'UniProtID') if '+I-' not in index_name and 'δ' not in index_name else pd.NA)
            self.processed_df['UniProtID_B'].append(Scoring.ledge.fetch(column_name, 'bioID', 'UniProtID') if '+I-' not in column_name and 'δ' not in column_name else pd.NA)
            self.processed_df['Sequence_A'].append(seqs.split('=')[0])
            self.processed_df['Sequence_B'].append(seqs.split('=')[1])
            self.processed_df['Interaction'].append([(self.file_stem, interaction_ad + interaction_bd)])
            self.processed_df['From'].append(self.file_stem)
            self.processed_df['TaxonID_A'].append('3702')
            self.processed_df['TaxonID_B'].append('3702')
            self.processed_df['Species_A'].append('Arabidopsis thaliana')
            self.processed_df['Species_B'].append('Arabidopsis thaliana')
        
        df = pd.DataFrame(self.processed_df)
        return df[df['Interaction'] != 'NULL']

if __name__ == '__main__':
    '''Test class'''
    scoring = Scoring('AGL14&')
    df = scoring.process()
    df.to_excel('AG14&.xlsx', index=False)