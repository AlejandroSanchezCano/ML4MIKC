"""
===============================================================================
Title:      BioGRID
Outline:    BioGRID class to download the data of any version of BioGRID,
            reduce it to only plant interactors, search for all MADS
            interactions, filter to only MADS vs. MADS interactions, and
            standarize the data frame to a common format that makes it 
            compatible to the Network class.
Docs:       https://thebiogrid.org/
Author:     Alejandro Sánchez Cano
Date:       17/10/2024
===============================================================================
"""

# Built-in modules
import subprocess
from io import StringIO

# Third-party modules
import pandas as pd
from multitax import NcbiTx

# Custom modules
from src.misc import path
from src.misc.logger import logger
from src.entities.interactor import Interactor

class BioGRID:

    def __init__(self, version: str):
        self.version = version

    def download_files(self) -> None:
        '''
        Downloads and unzips BioGRID 'ALL' file from the specified 
        BioGRID version.
        '''

        # Set up output directory
        output_dir = path.BIOGRID / self.version
        output_dir.mkdir(parents=True, exist_ok=True)

        # Download 'ALL' file
        url = f'https://downloads.thebiogrid.org/Download/BioGRID/Release-Archive/BIOGRID-{self.version}/'
        all_file = f'BIOGRID-ALL-{self.version}.tab3.zip'
        wget = f'wget {url}/{all_file} -P {output_dir} -q'
        subprocess.run(wget, shell = True)

        # Unzip file and remove compressed file
        unzip = f'unzip -qq {output_dir}/{all_file} -d {output_dir}'
        subprocess.run(unzip, shell = True)
        rm = f'rm {output_dir}/{all_file}'
        subprocess.run(rm, shell = True)

        # Logging
        logger.info(f'BioGRID {self.version} "ALL" file downloaded and unzipped')

    def reduce_to_plants(self) -> None:
        '''
        Reduces BioGRID 'ALL' file to only plant interactors to
        reduce computational burden when searching for UniProt IDs with
        grep in the whole file.
        '''

        # Set up file paths
        all_filepath = path.BIOGRID / self.version / f'BIOGRID-ALL-{self.version}.tab3.txt'
        plant_filepath = path.BIOGRID / self.version / f'BIOGRID-plants-{self.version}.tab3.txt'

        # Read BioGRID 'ALL' file as pandas DataFrame
        df = pd.read_csv(all_filepath, sep = '\t', dtype=str)

        # Filter only plant interactors
        ncbi_tx = NcbiTx()
        df_plants = df[df['Organism ID Interactor A'].apply(lambda x: 'Viridiplantae' in ncbi_tx.name_lineage(x))]

        # Save filtered file
        df_plants.to_csv(plant_filepath, sep = '\t', index = False)

        # Logging
        logger.info(f'BioGRID {self.version} "ALL" file ({len(df)} PPIs) reduced to only plant interactions ({len(df_plants)} PPIs)')

    def _grep(self, uniprot_id: str) -> pd.DataFrame:
        '''
        Searches for a specific UniProt ID in the BioGRID 'plants' file and
        retrieves its interactions in BioGRID.

        Parameters
        ----------
        uniprot_id : str
            UniProt ID to search for.

        Returns
        -------
        pd.DataFrame
            Interaction table of the given UniProt ID.
        '''

        # Set up file path
        file_path = path.BIOGRID / self.version / f'BIOGRID-plants-{self.version}.tab3.txt'

        # First line as columns
        cat = f'head -1 {file_path}'
        result = subprocess.run(cat, shell = True, text = True, stdout = subprocess.PIPE)
        columns = result.stdout.lstrip('#').rstrip().split('\t')

        # Search for UniProt ID in BioGRID 'ALL' file
        grep = f'grep {uniprot_id} {file_path}'
        result = subprocess.run(grep, shell = True, text = True, stdout = subprocess.PIPE)
        
        # Convert result to DataFrame
        if result.stdout == '':
            return pd.DataFrame(columns = columns)
        else:
            result = StringIO(result.stdout)
            return pd.read_csv(result, sep = '\t', names = columns)
        
    def mads_vs_all(self) -> None:
        '''
        Searches for MADS interactors in the BioGRID 'plants' file and
        retrieves their interactions in BioGRID.
        '''
        # Initialize DataFrame
        mads_vs_all = pd.DataFrame()

        # Iterate over MADS interactors
        for interactor in Interactor.iterate():
            # Search for UniProt ID in BioGRID 'plants' file
            df = self._grep(interactor.uniprot_id)
            # Append to DataFrame if not empty
            if not df.empty:
                mads_vs_all = pd.concat([mads_vs_all, df], ignore_index = True)
        
        # Save DataFrame
        filepath = path.NETWORKS / f'BioGRID_{self.version}_MADS_vs_ALL.tsv'
        mads_vs_all.to_csv(filepath, sep = '\t', index = False)

        # Logging
        logger.info(f'MADS vs. all PPIs in BioGRID {self.version} "plants" file -> dim({mads_vs_all.shape})')

    def mads_vs_mads(self) -> None:
        '''
        Filters MADS vs. MADS interactions from the MADS vs. ALL by 
        checking whether any of the SWISS-PROT or TrEMBL IDs are in the
        MIKC list from InterPro.
        '''
        # Load MADS_vs_ALL DataFrame
        filepath = path.NETWORKS / f'BioGRID_{self.version}_MADS_vs_ALL.tsv'
        mads_vs_all = pd.read_csv(filepath, sep = '\t')

        # MADS UniProt IDs
        mads = set([interactor.uniprot_id for interactor in Interactor.iterate()])

        # Concatenate UniProt IDs column A
        uniprot_columns_A = ['SWISS-PROT Accessions Interactor A', 'TREMBL Accessions Interactor A']
        concatenate = lambda row: '|'.join(row).replace('-|', '').rstrip('|-').split('|')
        uniprot_ids_A = mads_vs_all[uniprot_columns_A].apply(concatenate, axis = 1)
        is_there_mikc = lambda x: len(set(x) - mads) < len(x)
        mads_vs_mads_A = uniprot_ids_A.apply(is_there_mikc)

        # Concatenate UniProt IDs column B
        uniprot_columns_B = ['SWISS-PROT Accessions Interactor B', 'TREMBL Accessions Interactor B']
        concatenate = lambda row: '|'.join(row).replace('-|', '').rstrip('|-').split('|')
        uniprot_ids_B = mads_vs_all[uniprot_columns_B].apply(concatenate, axis = 1)
        is_there_mikc = lambda x: len(set(x) - mads) < len(x)
        mads_vs_mads_B = uniprot_ids_B.apply(is_there_mikc)

        # Filter MADS vs MADS interactions
        mads_vs_mads = mads_vs_all[mads_vs_mads_A & mads_vs_mads_B]

        # Save DataFrame
        filepath = path.NETWORKS / f'BioGRID_{self.version}_MADS_vs_MADS.tsv'
        mads_vs_mads.to_csv(filepath, sep = '\t', index = False)

        # Logging
        logger.info(f'MADS vs. MADS PPIs in BioGRID {self.version} "plants" file -> dim({mads_vs_mads.shape})')

    def __find_best_uniprot_id_candidate(self, uniprot_ids: list[str]) -> str:
        '''
        Given a list of UniProt IDs, returns the best candidate to be 
        used as the main UniProt ID in the database.

        Parameters
        ----------
        uniprot_ids : list[str]
            List of UniProt IDs to be considered.

        Returns
        -------
        str
            Main UniProt ID to be used
        '''
        # Initialize return list to be filled
        candidate = []

        # Iterate over UniProt IDs
        for uniprot_id in uniprot_ids:

            # Discard UniProt IDs not in database
            interactor = Interactor(uniprot_id)
            if interactor.domains == {}:
                continue
            
            # Discard TREBML
            if interactor.section != 'Swiss-Prot':
                continue
            
            # Append to candidate list
            candidate.append(uniprot_id)
        
        # Logging and manage errors
        if len(candidate) == 0:
            logger.error(f'No valid UniProt ID candidates found in {uniprot_ids}')
            return ''
        elif len(candidate) > 1:
            logger.error(f'More than one valid UniProt ID candidate found in {uniprot_ids} -> {candidate}')
            return candidate[0]
        else:
            return candidate[0]
        
    def standarize(self) -> None:
        '''
        Format data frame interaction network to accommodate standard naming
        convention of columns to homogenize the data frames from different 
        databases. It contains:
        - A: UniProt ID of interactor A
        - B: UniProt ID of interactor B
        - A=B: Concatenation of A and B sorted alphabetically
        - Species_A: Species ID of interactor A
        - Species_B: Species ID of interactor B
        - Seq_A: Sequence of interactor A
        - Seq_B: Sequence of interactor B
        - Seq: Sequence of interactor A : sequence of interactor B
        '''
        # Load MADS_vs_MADS DataFrame
        filepath = path.NETWORKS / f'BioGRID_{self.version}_MADS_vs_MADS.tsv'
        mads_vs_mads = pd.read_csv(filepath, sep = '\t')

        # Find best UniProt ID candicate in column A
        columns_uniprot_A = ['SWISS-PROT Accessions Interactor A', 'TREMBL Accessions Interactor A']
        concatenate = lambda row: '|'.join(row).replace('-|', '').rstrip('|-').split('|')
        uniprot_ids_A = mads_vs_mads[columns_uniprot_A].apply(concatenate, axis = 1).to_list()
        A = [self.__find_best_uniprot_id_candidate(uniprot_ids) for uniprot_ids in uniprot_ids_A]

        # Find best UniProt ID candicate in column B
        columns_uniprot_B = ['SWISS-PROT Accessions Interactor B', 'TREMBL Accessions Interactor B']
        concatenate = lambda row: '|'.join(row).replace('-|', '').rstrip('|-').split('|')
        uniprot_ids_B = mads_vs_mads[columns_uniprot_B].apply(concatenate, axis = 1).to_list()
        B = [self.__find_best_uniprot_id_candidate(uniprot_ids) for uniprot_ids in uniprot_ids_B]

        # Assign columns
        mads_vs_mads['A'] = A
        mads_vs_mads['B'] = B
        mads_vs_mads['A=B'] = mads_vs_mads[['A', 'B']].apply(lambda x: '='.join(sorted(x)), axis = 1)
        mads_vs_mads['Species_A'] = mads_vs_mads['Organism ID Interactor A']
        mads_vs_mads['Species_B'] = mads_vs_mads['Organism ID Interactor B']
        mads_vs_mads['Seq_A'] = [Interactor(uniprot_id).seq for uniprot_id in A]
        mads_vs_mads['Seq_B'] = [Interactor(uniprot_id).seq for uniprot_id in B]
        mads_vs_mads['Seq'] = mads_vs_mads['Seq_A'] + ':' + mads_vs_mads['Seq_B']
                                                                         
        # Remove duplicated columns
        mads_vs_mads = mads_vs_mads.drop_duplicates('A=B')
        mads_vs_mads = mads_vs_mads.drop_duplicates('Seq')

        # Save DataFrame
        filepath = path.NETWORKS / f'BioGRID_{self.version}_MADS_vs_MADS_standarized.tsv'
        mads_vs_mads = mads_vs_mads[['A', 'B', 'A=B', 'Species_A', 'Species_B', 'Seq_A', 'Seq_B', 'Seq']]
        mads_vs_mads.to_csv(filepath, sep = '\t', index = False)

        # Logging
        logger.info(f'MADS vs. MADS PPIs in BioGRID {self.version} file standarized -> dim({mads_vs_mads.shape})')