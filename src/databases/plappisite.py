"""
===============================================================================
Title:      PlaPPISite
Outline:    PlaPPISite class to access the current web content of PlaPPISite
            (no version system is used), search for all MADS interactions, and
            filter to only MADS vs. MADS interactions. They are represented by 
            their UniProt accessions.
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
            responses = await tqdm_asyncio.gather(*tasks, desc="Fetching PlaPPISite data")
    
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

    def mads_vs_all(self, mads_uniprots: list[str]) -> None:
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

        Parameters
        ----------
        mads_uniprots : list[str]
            List of UniProt IDs of MADS-box proteins.
        '''
        # Initialize DataFrame
        mads_vs_all = pd.DataFrame()

        # Retrieve non-predicted PPI table of all MADS proteins
        responses = asyncio.run(self._fetch_all(mads_uniprots))
        soups = [bs4.BeautifulSoup(response, features = 'lxml') for response in responses]
        tables = [self._get_table(soup) for soup in soups]
        non_predicted_tables = [table[table['PPI source'].apply(lambda x: x not in ['Predicted', 'prediction'])] for table in tables]

        # Append to DataFrame if not empty (non-predicted PPIs)
        for non_predicted_table in non_predicted_tables:
            if not non_predicted_table.empty:
                mads_vs_all = pd.concat([mads_vs_all, non_predicted_table], ignore_index = True)
        
        # Save DataFrame
        file_path = path.PLAPPISITE / 'MADS_vs_ALL.tsv'
        mads_vs_all.to_csv(file_path, sep = '\t', index = False)

        # Logging
        logger.info(f'MADS vs. all PPIs in PlaPPISite -> dim({mads_vs_all.shape})')

    def mads_vs_mads(self, mads_uniprots: set[str]) -> None:
        '''
        Filters MADS vs. MADS interactions from the MADS vs. ALL

        Parameters
        ----------
        mads_uniprots : set[str]
            List of UniProt IDs of MADS-box proteins to filter MADS vs. MADS
            interactions.
        '''
        # Load MADS_vs_ALL DataFrame
        filepath = path.PLAPPISITE / 'MADS_vs_ALL.tsv'
        mads_vs_all = pd.read_csv(filepath, sep = '\t')

        # Filter MADS vs. ALL DataFrame
        is_there_mikc = lambda x: set(x.split(' - ')).issubset(mads_uniprots)
        mads_vs_mads = mads_vs_all[mads_vs_all['PPI'].apply(is_there_mikc)]

        # Save DataFrame
        filepath = path.PLAPPISITE / 'MADS_vs_MADS.tsv'
        mads_vs_mads.to_csv(filepath, sep = '\t', index = False)

        # Logging
        logger.info(f'MADS vs. MADS PPIs in PlaPPISite -> dim({mads_vs_mads.shape})')

    def get_ppi_accessions(self) -> list[tuple[str, str]]:
        '''
        Retrieves all UniProt accessions of the PPIs in the MADS vs. MADS 
        interactions file.

        Returns
        -------
        list[tuple[str, str]]
            List of tuples with the UniProt accessions of the interactors
            A and B.
        '''
        # Load MADS_vs_MADS DataFrame
        filepath = path.PLAPPISITE / 'MADS_vs_MADS.tsv'
        mads_vs_mads = pd.read_csv(filepath, sep = '\t')

        # Retrieve UniProt accessions
        ppi_accessions = []
        for _, row in mads_vs_mads.iterrows():
            uniprot_A, uniprot_B = row['PPI'].split('-')
            ppi_accessions.append((uniprot_A, uniprot_B))

        # Logging
        logger.info(f'Retrieved {len(ppi_accessions)} UniProt pairs accessions from MADS_vs_MADS file')
        
        return ppi_accessions

if __name__ == '__main__':
    # Gather UniProt IDs of MADS-box proteins
    from src.misc import path
    from src.entities.collection import ProteinCollection
    file_path = path.DATA / 'mikc_proteins.h5'
    collection = ProteinCollection(file_path=file_path, datasets=['uniprot'])
    mads_uniprots = set([protein.uniprot for protein in collection])
    
    # PlaPPISite
    plappisite = PlaPPISite()
    #plappisite.mads_vs_all(mads_uniprots)
    plappisite.mads_vs_mads(mads_uniprots)
    ppis = plappisite.get_ppi_accessions()
    print(ppis)