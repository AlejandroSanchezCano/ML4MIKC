

# Built-in modules
import json

# Custom modules
from fasta import Fasta
from src.misc.logger import logger

# Parse sequence file
seq_file_path = '/home/asanchez/chonky/ML4MIKC/src/quick/yichun/seqs.fasta'
fasta = Fasta.from_file(seq_file_path)
records = []
for header, sequence in fasta.records:
    splitted_header = header.split('_')
    locus = splitted_header[0]
    bioID = splitted_header[1]
    has_interactions = splitted_header[-1] != 'NA'
    subtype = splitted_header[-1] if has_interactions else splitted_header[-2]
    alternative_bioID = '_'.join([split for split in splitted_header if split not in [locus, subtype, bioID, 'NA']])
    records.append({
        'locus': locus,
        'bioID': bioID,
        'alternative_bioID': alternative_bioID,
        'subtype': subtype,
        'sequence': sequence,
        'interacts':[]
    })
logger.info(f'{len(records)} parsed sequences')

# Remove alternative splicing variants
to_be_removed = []
for record in records:
    to_be_removed.append(False)
    if 'AT3G05860' in record['locus'] and not '.1' in record['locus']:
        to_be_removed[-1] = True
    if 'AT5G27090' in record['locus'] and not '.1' in record['locus']:
        to_be_removed[-1] = True
    if 'AT1G22590' in record['locus'] and not '.2' in record['locus']:
        to_be_removed[-1] = True
    if 'AT1G17310' in record['locus'] and not '.1' in record['locus']:
        to_be_removed[-1] = True
records = [record for record, remove in zip(records, to_be_removed) if not remove]
logger.info(f'{len(records)} sequences remain after removing alternative splicing variants')

# Parse interactions file
interactions_file_path = '/home/asanchez/chonky/ML4MIKC/src/quick/yichun/interaction_data.txt'
with open(interactions_file_path, 'r') as file:
    lines = [line.strip().split('\t\t') for line in file]
bioID2idx = {record['bioID']: idx for idx, record in enumerate(records)}
for line in lines:
    if len(line) != 2: continue
    if line[0].startswith('M'): continue
    if line[0] not in bioID2idx: continue
    if line[1] not in bioID2idx: continue
    records[bioID2idx[line[0]]]['interacts'].append(line[1])
    records[bioID2idx[line[1]]]['interacts'].append(line[0])

# Save JSON
out_file = '/home/asanchez/chonky/ML4MIKC/src/quick/yichun/parsed_records.json'
with open(out_file, 'w') as file:
    json.dump(records, file)
logger.info(f'JSON files saved')