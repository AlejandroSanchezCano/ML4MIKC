"""
===============================================================================
Title:      PlaPPISite
Outline:    PlaPPISite class to access the current web content of PlaPPISite
            (no version system is used), search for all MADS interactions,
            filter to only MADS vs. MADS interactions, and standarize the data 
            frame to a common format that makes it compatible to the Network 
            class.
Docs:       http://zzdlab.com/plappisite/
Author:     Alejandro Sánchez Cano
Date:       17/10/2024
===============================================================================
"""

# Built-in modules
import asyncio

# Third-party modules
import bs4
import aiohttp
import pandas as pd
from tqdm.asyncio import tqdm_asyncio

# Custom modules
from src.misc import path
from src.misc.logger import logger
from src.entities.interactor import Interactor


class PlaPPISite:

    async def _fetch(
            self, 
            session: aiohttp.client.ClientSession, 
            semaphore: asyncio.locks.Semaphore, 
            uniprot_id: str
            ) -> str:
        '''
        Fetches the web content of an UniProt ID from PlaPPISite using
        an asynchronous HTTP request. 
        A limit amoount of attemps is set to avoid the error 
        "aiohttp.client_exceptions.ServerDisconnectedError" due to momentary
        server disconnections. 

        Parameters
        ----------
        session : aiohttp.client.ClientSession
            Asynchronous HTTP session.
        
        semaphore : asyncio.locks.Semaphore
            Semaphore to limit the number of simultaneous requests.
        
        uniprot_id : str
            UniProt ID to fetch interactions of.
        
        Returns
        -------
        str
            Web content of the UniProt ID.
        '''
        url = f'http://zzdlab.com/plappisite/single_idmap.php?protein={uniprot_id}'
        async with semaphore:
            for attempt in range(3):
                try:
                    async with session.get(url) as response:
                        return await response.text()
                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    logger.info(f"Attempt {attempt+1} with {uniprot_id} failed: {e}")
                    await asyncio.sleep(2)

    async def _fetch_all(self, uniprot_ids: list[str]) -> list[str]:
        '''
        Fetches the web text content of a list of UniProt IDs from 
        PlaPPISite using asynchronous HTTP requests because to process
        18K UniProt IDs sequencially could take +8h.

        Parameters
        ----------
        uniprot_ids : list[str]
            List of UniProt IDs to fetch interactions of.

        Returns
        -------
        list[str]
            List of web text contents of the UniProt IDs.
        '''
        # Limit the number of simultaneous requests with a semaphore
        semaphore = asyncio.Semaphore(100)

        # Fetch all responses
        async with aiohttp.ClientSession() as session:
            tasks = [self._fetch(session, semaphore, uniprot_id) for uniprot_id in uniprot_ids]
            responses = await tqdm_asyncio.gather(*tasks)
    
        # Logging
        logger.info(f'PlaPPISite -> Retrieved {len(responses)} responses')

        return responses

    def _get_table(self, soup: bs4.BeautifulSoup) -> pd.DataFrame:
        '''
        Parses the web content of an accession and retrieves the PPI 
        table.

        Parameters
        ----------
        soup : bs4.BeautifulSoup
            Accession's web content.

        Returns
        -------
        pd.DataFrame
            Interaction table.
        '''
        table = soup.find('div', attrs = {'id':'container_table'})
        columns = [th.text for th in table.find_all('th')]
        tds = [td.text for td in table.find_all('td')]
        rows = [tds[i : i + len(columns)] for i in range(0, len(tds), len(columns))]

        return pd.DataFrame(rows, columns = columns)

    def mads_vs_all(self) -> None:
        '''
        Searches for MADS interactors in PlaPPISite and retrieves their
        PPIs.
        PlaPPISite uses UniProt IDs as main IDs, so each UniProt IDs is 
        checked to have PPIs in PlaPPISite by parsing the web content, 
        retrieving the PPI table and removing the predicted PPIs. If the
        resulting PPI table is not empty, the UniProt ID is considered
        to have PPIs in PlaPPISite and the Interactor.plappisite_id 
        attribute is updated. 
        The predicted PPIs are removed because STRING is arguably the
        best source of predicted PPIs, so PlaPPISite predicted PPIs will 
        not be likely used.
        '''
        # Initialize DataFrame
        mads_vs_all = pd.DataFrame()

        # Retrieve non-predicted PPI table of all MADS proteins
        uniprot_ids = [interactor.uniprot_id for interactor in Interactor.iterate()]
        responses = asyncio.run(self._fetch_all(uniprot_ids))
        soups = [bs4.BeautifulSoup(response, features = 'lxml') for response in responses]
        tables = [self._get_table(soup) for soup in soups]
        non_predicted_tables = [table[table['PPI source'].apply(lambda x: x not in ['Predicted', 'prediction'])] for table in tables]

        # Append to DataFrame if not empty (non-predicted PPIs)
        for non_predicted_table in non_predicted_tables:
            if not non_predicted_table.empty:
                mads_vs_all = pd.concat([mads_vs_all, non_predicted_table], ignore_index = True)
        
        # Save DataFrame
        file_path = path.NETWORKS / 'PlaPPISite_MADS_vs_ALL.tsv'
        mads_vs_all.to_csv(file_path, sep = '\t', index = False)

        # Logging
        logger.info(f'MADS vs. all PPIs in PlaPPISite -> dim({mads_vs_all.shape})')

    def mads_vs_mads(self) -> None:
        '''
        Filters MADS vs. MADS interactions from the MADS vs. ALL
        '''
        # Load MADS_vs_ALL DataFrame
        filepath = path.NETWORKS / 'PlaPPISite_MADS_vs_ALL.tsv'
        mads_vs_all = pd.read_csv(filepath, sep = '\t')

        # MADS UniProt IDs
        mads = set([interactor.uniprot_id for interactor in Interactor.iterate()])

        # Filter MADS vs. ALL DataFrame
        is_there_mikc = lambda x: set(x.split(' - ')).issubset(mads)
        mads_vs_mads = mads_vs_all[mads_vs_all['PPI'].apply(is_there_mikc)]

        # Save DataFrame
        filepath = path.NETWORKS / 'PlaPPISite_MADS_vs_MADS.tsv'
        mads_vs_mads.to_csv(filepath, sep = '\t', index = False)

        # Logging
        logger.info(f'MADS vs. MADS PPIs in PlaPPISite -> dim({mads_vs_mads.shape})')

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
        filepath = path.NETWORKS / 'PlaPPISite_MADS_vs_MADS.tsv'
        mads_vs_mads = pd.read_csv(filepath, sep = '\t')

        # Assign columns
        mads_vs_mads['A'] = mads_vs_mads['PPI'].apply(lambda x: x.split(' - ')[0])
        mads_vs_mads['B'] = mads_vs_mads['PPI'].apply(lambda x: x.split(' - ')[1])
        mads_vs_mads['A=B'] = mads_vs_mads[['A', 'B']].apply(lambda x: '='.join(sorted(x)), axis = 1)
        mads_vs_mads['Species_A'] = mads_vs_mads['A'].apply(lambda x: Interactor(x).taxon_id)
        mads_vs_mads['Species_B'] = mads_vs_mads['B'].apply(lambda x: Interactor(x).taxon_id)
        mads_vs_mads['Seq_A'] = mads_vs_mads['A'].apply(lambda x: Interactor(x).seq)
        mads_vs_mads['Seq_B'] = mads_vs_mads['B'].apply(lambda x: Interactor(x).seq)
        mads_vs_mads['Seq'] = mads_vs_mads['Seq_A'] + ':' + mads_vs_mads['Seq_B']
                                                                         
        # Remove duplicated columns
        mads_vs_mads = mads_vs_mads.drop_duplicates('A=B')
        mads_vs_mads = mads_vs_mads.drop_duplicates('Seq')

        # Save DataFrame
        filepath = path.NETWORKS / 'PlaPPISite_MADS_vs_MADS_standarized.tsv'
        mads_vs_mads = mads_vs_mads[['A', 'B', 'A=B', 'Species_A', 'Species_B', 'Seq_A', 'Seq_B', 'Seq']]
        mads_vs_mads.to_csv(filepath, sep = '\t', index = False)

        # Logging
        logger.info(f'Standarized MADS vs. MADS PPIs in PlaPPISite -> dim({mads_vs_mads.shape})')