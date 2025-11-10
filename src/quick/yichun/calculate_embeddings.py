
# Built-in modules
import json

# Third-party modules
import numpy as np

# Custom modules
from esm2 import ESM2
from src.misc.logger import logger

# Load JSON
file_path = '/home/asanchez/chonky/ML4MIKC/src/quick/yichun/parsed_records.json'
with open(file_path, 'r') as file:
    records = json.load(file)
logger.info(f'{len(records)} records loaded from JSON')

# ESM2 embeddings
embeddings = []
esm2 = ESM2('650M')
for record in records:
    data = [(record['bioID'], record['sequence'])]
    esm2.prepare_data(data)
    esm2.run_model(layers = [33])
    r, s = esm2.extract_representations(layer = 33)
    embeddings.append(s)
embeddings = np.vstack(embeddings)
logger.info(f'Embeddings of shape {embeddings.shape} calculated')

# Save embeddings
out_file = '/home/asanchez/chonky/ML4MIKC/src/quick/yichun/embeddings.npy'
np.save(out_file, embeddings)
