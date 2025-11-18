# Third-party modules
import pandas as pd
from tqdm import tqdm
from multitax import NcbiTx

# Custom modules
from src.misc import path
from src.entities.ppi import PPI
from src.misc.logger import logger
from src.interactions.paper import Paper
from src.entities.protein import Protein
from src.interactions.scoring import Scoring
from src.entities.collection import ProteinCollection, PPICollection
logger.setLevel(20)
logger.info('Importing modules completed')

######################
#  ONLINE DATABASES  #
######################

# Load interaction file
file_path = path.DATABASES / 'ppi_uniprot_accessions.txt'
online_df = pd.read_csv(file_path, sep = '\t')
# Add interaction column
online_df['Interaction'] = 1
# Add, remove and reorder columns
online_df['bioID_A'] = ''
online_df['bioID_B'] = ''
online_df.rename(columns = {'Database': 'Origin'}, inplace = True)
online_df.rename(columns = {'Seq_A': 'Sequence_A', 'Seq_B': 'Sequence_B'}, inplace = True)
online_df = online_df[[
    'bioID_A', 'bioID_B', 'UniProtID_A', 'UniProtID_B', 'Sequence_A', 
    'Sequence_B', 'Interaction', 'Origin', 'TaxonID_A', 'TaxonID_B', 
    ]]
# Logging
logger.info(f'{len(online_df)} PPIs from online databases')

#######################
#  LITERATURE MINING  #
#######################

# Validate filestems -> must be in the OnHold folder
Paper.validate_filestems()
# Process papers
paper_dfs = []
excel_folder = path.LITERATUREMINING / 'Excels'
excel_filenames = sorted(list(excel_folder.iterdir()))
for excel_filename in tqdm(excel_filenames):
    author, year = excel_filename.stem.split('_')
    paper = Paper(author, year)
    paper.process()
    df = paper.interaction_df()
    paper_dfs.append(df)
# Concatenate dataframes
mining_df = pd.concat(paper_dfs, ignore_index=True)
mining_df.reset_index(drop = True, inplace = True)
# Add taxon IDs column
guidelines = pd.read_excel(path.LITERATUREMINING / 'Guidelines.xlsx', sheet_name = 'Species')
prefix2taxonID = dict(zip(guidelines['Abbreviation'], guidelines['NCBI'].astype(int)))
mining_df['TaxonID_A'] = mining_df['bioID_A'].str[:2].map(prefix2taxonID)
mining_df['TaxonID_B'] = mining_df['bioID_B'].str[:2].map(prefix2taxonID)
# Logging
logger.info(f'{len(mining_df)} PPIs from literature mining')

#############
#  SCORING  #
#############

# Process scoring files
scoring_dfs = []
for file in path.SCORING.iterdir():
    logger.info(f'Processing scoring file: {file.name}')
    if 'LEDGE.xlsx' in file.name: continue
    scoring = Scoring(file.stem)
    df = scoring.process()
    scoring_dfs.append(df)
# Concatenate dataframes
scoring_df = pd.concat(scoring_dfs, ignore_index=True)
scoring_df.reset_index(drop = True, inplace = True)
# Logging
logger.info(f'{len(scoring_df)} PPIs from scoring datasets')

###################
#  MERGE SOURCES  #
###################

# Concatenate all sources
df = pd.concat([scoring_df, mining_df, online_df])
df.reset_index(drop = True, inplace = True)
df['Interaction'] = df['Interaction'].apply(lambda x: [x])
logger.info(f'{len(df)} total PPIs after merging all sources')

#######################
#  REMOVE REDUNDANCY  #
#######################

# Detect and remove redundancies
drop_col = []
redundancy = []
for idx, row in df.iterrows():
    # Concatenate sequences
    seqs = '='.join([row['Sequence_A'], row['Sequence_B']])
    # No redundancy
    if seqs not in redundancy:
        redundancy.append(seqs)
        drop_col.append(False)
    # Redundancy
    else:
        # Dummy indicator of redundancy
        redundancy.append('')
        drop_col.append(True)
        # Calculate original index
        original_idx = redundancy.index(seqs)
        # Update values
        df.at[original_idx, 'Interaction'] = df.at[original_idx, 'Interaction'] + df.at[idx, 'Interaction']
        df.at[original_idx, 'Origin'] = df.at[original_idx, 'Origin'] + '|' + df.at[idx, 'Origin']
# Drop redundancies
df = df[~pd.Series(drop_col)]
df.reset_index(drop = True, inplace = True)
# Split 'Origin' column
df['Origin'] = df['Origin'].apply(lambda x: x.split('|'))
# Fix 'Interaction' column to be list of int or str
listoflists = lambda x: [item if isinstance(item, list) else [item] for item in x]
df['Interaction'] = df['Interaction'].apply(listoflists)
# Logging
logger.info(f'{len(df)} non-redundant PPIs after removing redundancies')

###############################
#  REMOVE EMPTY INTERACTIONS  #
###############################

# Drop interactions with only 'AUTO', 'NLW' or 'ND' values
drop_col = []
for idx, row in df.iterrows():
    interaction = [item for lst in row['Interaction'] for item in lst]
    if 1 in interaction or 0 in interaction or 'NC' in interaction:
        drop_col.append(False)
    else:
        drop_col.append(True)
# Drop empty interactions
df = df[~pd.Series(drop_col)]
df.reset_index(drop = True, inplace = True)
# Logging
logger.info(f'{len(df)} PPIs after removing empty interactions')

####################################
#  REMOVE NON-CANONICAL SEQUENCES  #
####################################

# Remove sequences with non-canonical amino acids
df = df[~df['Sequence_A'].str.contains('[^ACDEFGHIKLMNPQRSTVWY]', regex = True)]
df = df[~df['Sequence_B'].str.contains('[^ACDEFGHIKLMNPQRSTVWY]', regex = True)]
df.reset_index(drop = True, inplace = True)
# Identify errors in sequences like '*', '-' or non-canonical amino acids like 'X
aminoacids = set('ACDEFGHIKLMNPQRSTVWY')
for idx, row in df.iterrows():
    seqA = row['Sequence_A']
    seqB = row['Sequence_B']
    if set(seqA) - aminoacids:
        logger.warning(f'Non canonical amino acids in row {idx}: {set(seqA) - aminoacids}')
    if set(seqB) - aminoacids:
        logger.warning(f'Non canonical amino acids in row {idx}: {set(seqB) - aminoacids}')
# Logging
logger.info(f'{len(df)} PPIs after removing non-canonical sequences')

############################
#  REMOVE BANNED PROTEINS  #
############################

# Remove specific interactions
banned = [
        'SlRIN',            # It has 2 K domains
        'SlTM3',            # Does not have an M domain
        'ZaMADS70',         # Does not have an M domain
        'SlMBP13',          # Not a MIKC protein
        'TtAG1-del3-del13', # Deletion of kink between K1 and K2
        'Ta42G17',          # Does not have K2 and K3 domains
        'Ta57H08',          # Does not have an M domain
        ]
df = df[~df['bioID_A'].isin(banned)]
df = df[~df['bioID_B'].isin(banned)]
df.reset_index(drop = True, inplace = True)
# Logging
logger.info(f'{len(df)} PPIs after removing banned proteins')

########################
#  CREATE COLLECTIONS  #
########################

# Protein and PPI collections
ppis = []
proteins = []
for idx, row in tqdm(df.iterrows(), total=len(df)):
    # Protein A
    proteinA = Protein(
        name = row['bioID_A'] if not pd.isna(row['bioID_A']) else '',
        seq = row['Sequence_A'],
        uniprot = row['UniProtID_A'] if not pd.isna(row['UniProtID_A']) else '',
        taxon = int(row['TaxonID_A']),
    )
    # Protein B
    proteinB = Protein(
        name = row['bioID_B'] if not pd.isna(row['bioID_B']) else '',
        seq = row['Sequence_B'],
        uniprot = row['UniProtID_B'] if not pd.isna(row['UniProtID_B']) else '',
        taxon = int(row['TaxonID_B']),
    )
    # Add to collection if not present
    if proteinA not in proteins:
        proteins.append(proteinA)
    if proteinB not in proteins:
        proteins.append(proteinB)
    # PPI
    ppi = PPI(
        p1 = proteinA,
        p2 = proteinB,
        interaction = row['Interaction'],
        origin = row['Origin'],
    )
    ppis.append(ppi)

# Create protein collection
file_path = path.DATA / 'predict_proteins.h5'
collection = ProteinCollection(file_path = file_path, items = proteins)
collection.to_hdf5()

# Create PPI collection
file_path = path.DATA / 'predict_ppis.h5'
protein_path = path.DATA / 'predict_proteins.h5'
collection = PPICollection(
    file_path = file_path, 
    protein_path = protein_path,
    items = ppis,
    )
collection.to_hdf5()