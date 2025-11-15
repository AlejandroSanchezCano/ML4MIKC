"""
===============================================================================
Title:      BioGRID
Outline:    BioGRID class to download the data of any version of BioGRID,
            reduce it to only plant interactors, search for all MADS
            interactions, and filter to only MADS vs. MADS interactions. They 
            are represented by their UniProt accessions.
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
from src.entities.collection import ProteinCollection

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
        
    def mads_vs_all(self, uniprots: set[str]) -> None:
        '''
        Searches for MADS interactors in the BioGRID 'plants' file and
        retrieves their interactions in BioGRID.

        Parameters
        ----------
        uniprots : set[str]
            List of UniProt IDs of MADS-box proteins.
        '''
        # Read BioGRID 'plants' file
        plant_filepath = path.BIOGRID / self.version / f'BIOGRID-plants-{self.version}.tab3.txt'
        df_plants = pd.read_csv(plant_filepath, sep = '\t', dtype=str)

        # Helper function
        def has_match(cell: str) -> bool:
            '''
            Helper function to check whether a cell contains any UniProt ID
            from the given set.

            Parameters
            ----------
            cell : str
                Cell content

            Returns
            -------
            bool
                Whether the cell contains any UniProt ID from the set
            '''

            if not cell:
                return False
            ids = cell.split('|')
            return any(i in uniprots for i in ids)

        # Filter
        mask = (
            df_plants["SWISS-PROT Accessions Interactor A"].apply(has_match) |
            df_plants["SWISS-PROT Accessions Interactor B"].apply(has_match) |
            df_plants["TREMBL Accessions Interactor A"].apply(has_match) |
            df_plants["TREMBL Accessions Interactor B"].apply(has_match)
        )
        mads_vs_all = df_plants[mask]
        
        # Save DataFrame
        filepath = path.BIOGRID / self.version / 'MADS_vs_ALL.tsv'
        mads_vs_all.to_csv(filepath, sep = '\t', index = False)

        # Logging
        logger.info(f'MADS vs. all PPIs in BioGRID {self.version} "plants" file -> dim{mads_vs_all.shape}')

    def mads_vs_mads(self, uniprots: set[str]) -> None:
        '''
        Filters MADS vs. MADS interactions from the MADS vs. ALL by 
        checking whether any of the SWISS-PROT or TrEMBL IDs are in the
        MIKC list from InterPro.

        Parameters
        ----------
        uniprots : set[str]
            List of UniProt IDs of MADS-box proteins.
        '''
        # Load MADS_vs_ALL DataFrame
        filepath = path.BIOGRID / self.version / 'MADS_vs_ALL.tsv'
        mads_vs_all = pd.read_csv(filepath, sep = '\t')

        # Concatenate UniProt IDs column A
        uniprot_columns_A = ['SWISS-PROT Accessions Interactor A', 'TREMBL Accessions Interactor A']
        concatenate = lambda row: '|'.join(row).replace('-|', '').rstrip('|-').split('|')
        uniprot_ids_A = mads_vs_all[uniprot_columns_A].apply(concatenate, axis = 1)
        is_there_mikc = lambda x: len(set(x) - uniprots) < len(x)
        mads_vs_mads_A = uniprot_ids_A.apply(is_there_mikc)

        # Concatenate UniProt IDs column B
        uniprot_columns_B = ['SWISS-PROT Accessions Interactor B', 'TREMBL Accessions Interactor B']
        concatenate = lambda row: '|'.join(row).replace('-|', '').rstrip('|-').split('|')
        uniprot_ids_B = mads_vs_all[uniprot_columns_B].apply(concatenate, axis = 1)
        is_there_mikc = lambda x: len(set(x) - uniprots) < len(x)
        mads_vs_mads_B = uniprot_ids_B.apply(is_there_mikc)

        # Filter MADS vs MADS interactions
        mads_vs_mads = mads_vs_all[mads_vs_mads_A & mads_vs_mads_B]

        # Save DataFrame
        filepath = path.BIOGRID / self.version / 'MADS_vs_MADS.tsv'
        mads_vs_mads.to_csv(filepath, sep = '\t', index = False)

        # Logging
        logger.info(f'MADS vs. MADS PPIs in BioGRID {self.version} "plants" file -> dim({mads_vs_mads.shape})')

    def ppi_uniprot_accessions(self) -> list[tuple[str, str]]:
        '''
        Retrieves all PPIs from the MADS vs. MADS DataFrame as a list of
        tuples with the UniProt accessions of both interactors. Take SWISS-PROT
        accession if available, otherwise take the first TrEMBL accession from
        the list.

        Returns
        -------
        list[tuple[str, str]]
            List of tuples with the UniProt accessions of both interactors
            in each PPI.
        '''
        # Load MADS_vs_MADS DataFrame
        filepath = path.BIOGRID / self.version / 'MADS_vs_MADS.tsv'
        mads_vs_mads = pd.read_csv(filepath, sep = '\t')

        # Retrieve UniProt accessions
        ppi_uniprot_accessions = []
        for _, row in mads_vs_mads.iterrows():
            swissprot_A = row['SWISS-PROT Accessions Interactor A']
            swissprot_B = row['SWISS-PROT Accessions Interactor B']
            trembl_A = row['TREMBL Accessions Interactor A'].split('|')
            trembl_B = row['TREMBL Accessions Interactor B'].split('|')
            uniprot_A = swissprot_A if swissprot_A != '-' else trembl_A[0]
            uniprot_B = swissprot_B if swissprot_B != '-' else trembl_B[0]
            ppi_uniprot_accessions.append((uniprot_A, uniprot_B))
        
        # Logging
        logger.info(f'Retrieved {len(ppi_uniprot_accessions)} UniProt pairs accessions from MADS_vs_MADS file')
        
        return ppi_uniprot_accessions

if __name__ == '__main__':
    # Gather UniProt IDs of MADS-box proteins
    from src.misc import path
    from src.entities.collection import ProteinCollection
    file_path = path.DATA / 'mikc_proteins.h5'
    collection = ProteinCollection(file_path=file_path, datasets=['uniprot'])
    uniprots = set([protein.uniprot for protein in collection])
    
    # Run BioGRID processing
    biogrid = BioGRID('5.0.251')
    #biogrid.download_files()
    #biogrid.reduce_to_plants()
    biogrid.mads_vs_all(uniprots)
    biogrid.mads_vs_mads(uniprots)
    ppis = biogrid.ppi_uniprot_accessions()
    #print(ppis)