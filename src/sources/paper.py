"""
===============================================================================
Title:      Paper
Outline:    Paper class to operate with the literature mining results for a 
            given paper. Using the author and the year, it fetches the excel
            file with the interactions, processes the data, and returns a
            DataFrame that can be merged with the rest of the data.
Author:     Alejandro Sánchez Cano
Date:       02/10/2024
===============================================================================
"""

# Built-in modules
from typing import Generator

# Third-party modules
import pandas as pd

# Custom modules
from src.misc import path
from src.misc.logger import logger
from src.sources.ledge import Ledge
from src.entities.mutation import Mutation

class Paper:

    ledge = Ledge(path.LITERATUREMINING / 'LEDGE.xlsx')

    def __init__(self, author: str, year: str):
        self.author = author
        self.year = year
        self.sheets = list(pd.read_excel(
            io = path.LITERATUREMINING / 'Excels' / f'{author}_{year}.xlsx', 
            sheet_name = None,
            # na_values = 'ND',
            header = 0
            ).values())
        self.redundancy = {}
        self.df = {
            'bioID_A': [],
            'bioID_B': [],
            'UniProtID_A': [],
            'UniProtID_B': [],
            'Sequence_A': [],
            'Sequence_B': [],
            'Interaction': [],
            'From': []
        }

    def __repr__(self):
        return f'{self.author} ({self.year})'
    
    @staticmethod
    def validate_filestems() -> bool:
        '''
        Validate that the files in the PDF and Excel folders match.

        Returns
        -------
        bool
            True if the files stem names are the same, False otherwise.
        '''
        pdf_folder = path.LITERATUREMINING / 'PDFs'
        excel_folder = path.LITERATUREMINING / 'Excels'
        pdf_stems = set([pdf.stem for pdf in pdf_folder.iterdir()])
        excel_stems = set([excel.stem for excel in excel_folder.iterdir()])
        if pdf_stems - excel_stems:
            logger.warning(f'PDFs without Excel files: {pdf_stems - excel_stems}')
            return False
        elif excel_stems - pdf_stems:
            logger.warning(f'Excel files without PDFs: {excel_stems - pdf_stems}')
            return False
        else:
            return True

    def _process_list(self, df: pd.DataFrame) -> Generator[tuple[str, str, str], None, None]:
        '''
        Convert a 'list' format DataFrame with columns 'A', 'B', and 
        'Interaction' into a generator of tuples (A, B, Interaction).

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with columns 'A', 'B', and 'Interaction'.

        Yields
        -------
        tuple[str, str, str]
            Tuples of the form (A, B, Interaction) where A and B are 
            bioIDs and Interaction is the interaction score.
        '''
        # Iterate over rows
        for _, (a, b, interaction) in df.iterrows():
            yield a, b, interaction

    
    def _process_matrix(self, df: pd.DataFrame) -> Generator[tuple[str, str, str], None, None]: 
        '''
        Convert a 'matrix' format DataFrame into a generator of tuples
        (A, B, Interaction).
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with the first column as bioIDs and the first row as 
            bioIDs, with the rest of the cells containing interaction scores.
        
        Yields
        -------
        tuple[str, str, str]
            Tuples of the form (A, B, Interaction) where A and B are 
            bioIDs and Interaction is the interaction score.
        '''
        # First column as index
        df = df.set_index(df.columns[0])

        # Iterate
        for a in df.index:
            for b in df.columns:
                interaction = df.loc[a, b]
                yield a, b, interaction

    def process(self) -> pd.DataFrame:
        '''
        Process the sheets of interaction data and convert them into a 
        DataFrame-like dictionary. The processing involves:
        - Detect list or matrix formats and process accordingly.
        - Identify bioIDs and verify their presence in the ledge.
        - Fetch sequences from the ledge using bioIDs if they are available.
        Else, we discard the PPI. Because we match at the bioID level, there
        could be cases where two sequences have different interaction scores
        and are counted twice because one may have a different bioID. I do not
        think this happens often, plus later we will reduce duplications at the
        sequence level.
        - Mutate sequences based on the mutations encoded in the bioID.
        - Sort sequences and join them (A=B with A < B alphabetically).
        - Detect duplicates within the paper and update the interaction scores.
        This duplicity is due to the AB/DB setting.

        Returns
        -------
        pd.DataFrame
            Processed sheet.
        '''
        for df in self.sheets:
            if df.columns[2] == 'Interaction':
                a_b_interaction = self._process_list(df)
            else:
                a_b_interaction = self._process_matrix(df)

            for a, b, interaction in a_b_interaction:

                # Split name
                name_a, *mut_a = a.split('_')
                name_b, *mut_b = b.split('_')

                # Check that the bioIDs are in ledge
                Paper.ledge.bioID_in_ledge(name_a)
                Paper.ledge.bioID_in_ledge(name_b)

                # Get sequences from ledge
                seq_a = Paper.ledge.fetch(name_a, 'bioID', 'Seq')
                seq_b = Paper.ledge.fetch(name_b, 'bioID', 'Seq')

                # Stop if sequence is NONE in ledge
                if seq_a == 'NONE' or seq_b == 'NONE':
                    continue

                # Mutate sequences
                mut_seq_a = Mutation.mutate(seq_a, mut_a)
                mut_seq_b = Mutation.mutate(seq_b, mut_b)

                # Join sequences
                mut_seqs = '='.join(sorted([mut_seq_a, mut_seq_b]))
                
                # Apply the sorting to the bioIDs
                if mut_seq_a != mut_seqs.split('=')[0]:
                    a, b = b, a

                # Check for duplicates within paper
                if mut_seqs not in self.redundancy:
                    self.redundancy[mut_seqs] = interaction
                else:

                    # Calculate indeces to update
                    idx_a = set([idx for idx, seq in enumerate(self.df['Sequence_A']) if seq == mut_seqs.split('=')[0]])
                    idx_b = set([idx for idx, seq in enumerate(self.df['Sequence_B']) if seq == mut_seqs.split('=')[1]])
                    idx = idx_a & idx_b
                    assert len(idx) == 1, f'Index error: {idx}'
                    idx = idx.pop()

                    # Duplications are due to AB/DB setting
                    self.df['Interaction'][idx][0][1].append(interaction)
                
                    continue

                # Add to final data frame
                self.df['bioID_A'].append(a)
                self.df['bioID_B'].append(b)
                self.df['UniProtID_A'].append(Paper.ledge.fetch(a.split('_')[0], 'bioID', 'UniProtID'))
                self.df['UniProtID_B'].append(Paper.ledge.fetch(b.split('_')[0], 'bioID', 'UniProtID'))
                self.df['Sequence_A'].append(mut_seqs.split('=')[0])
                self.df['Sequence_B'].append(mut_seqs.split('=')[1])
                self.df['Interaction'].append([(f'{self.author}_{self.year}', [interaction])])
                self.df['From'].append(self.__repr__())

    def interaction_df(self) -> pd.DataFrame:
        '''
        Create a DataFrame with the interactions.

        Returns
        -------
        pd.DataFrame
            DataFrame with the interactions.
        '''
        # Convert to data frame
        df = pd.DataFrame(self.df)

        # Add NaNs
        df = df.replace('NONE', pd.NA)

        # Remove NaNs (ND) on the interaction column
        df = df[df['Interaction'] != 'ND']

        # Logger
        logger.info(f'{self.author} ({self.year}) has {len(df)} interactions')

        return df

if __name__ == '__main__':
    '''Test class'''
    Paper.validate_filestems()
    paper = Paper('Gong', 2017)
    paper.process()
    df = paper.interaction_df()
    df.to_excel('Gong_2017.xlsx', index=False)