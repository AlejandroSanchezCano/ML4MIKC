
# Built-in modules
from pathlib import Path

# Third-party modules
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

# Custom modules
from src.misc.logger import logger

# Load JSON files
records = {}
folder = Path('/home/asanchez/chonky/ML4MIKC/src/type1')
for json_file in folder.glob('*.json'):
    with open(json_file, 'r') as infile:
        data = json.load(infile)
    records[data['bioID']] = data
logger.info(f'{len(records)} loaded JSON files')

# Create dataset
X = []
y = []
for bioID1 in tqdm(records, desc='Creating dataset'):
    for bioID2 in records:
        emb1 = records[bioID1]['embedding']
        emb2 = records[bioID2]['embedding']
        combined_emb = np.concatenate((emb1, emb2))
        X.append(combined_emb)
        if bioID2 in records[bioID1]['interacts']:
            y.append(1)
        else:
            y.append(0)
X = np.array(X)
y = np.array(y)
logger.info(f'Created dataset with {X.shape[0]} samples and {X.shape[1]} features')
logger.info(f'Positive samples: {np.sum(y==1)}, Negative samples: {np.sum(y==0)}')

# Proteins per subfamily
from collections import defaultdict
famandfams = defaultdict(int)
subfamilies = ['Mg', 'Mb', 'Ma', 'Mas']
for prot in records:
    fam1 = records[prot]['subtype']
    interacts = records[prot]['interacts']
    for prot_int in interacts:
        fam2 = records[prot_int]['subtype']
        logger.info(f'Interaction: {prot} ({fam1}) - {prot_int} ({fam2})')
        tuple_fam = tuple(sorted([fam1, fam2]) )
        famandfams[tuple_fam] += 1
print('Interactions per subfamily pair:')
for k, v in famandfams.items():
    print(f'{k}: {v}')

# Random Forest Classifier
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

indices = np.arange(X.shape[0])
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, indices, test_size=0.2, random_state=42, shuffle=True
    )
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
balanced_acc = sklearn.metrics.balanced_accuracy_score(y_test, y_pred)
f1 = sklearn.metrics.f1_score(y_test, y_pred)
logger.info(f'Balanced Accuracy: {balanced_acc}')
logger.info(f'F1 Score: {f1}')
cm = confusion_matrix(y_test, y_pred)
cm = pd.DataFrame(cm, index=['Actual_Negative', 'Actual_Positive'], columns=['Predicted_Negative', 'Predicted_Positive'])
logger.info(f'Confusion Matrix:\n{cm}')


# Print BioId pairs for false negatives
false_negatives = np.where((y_test == 1) & (y_pred == 0))
print(false_negatives)
print(idx_test[false_negatives])
for idx in idx_test[false_negatives]:
    total_records = len(records)
    bioID1_idx = idx // total_records
    bioID2_idx = idx % total_records
    bioID1 = list(records.keys())[bioID1_idx]
    bioID2 = list(records.keys())[bioID2_idx]
    subtype1 = records[bioID1]['subtype']
    subtype2 = records[bioID2]['subtype']
    print(f'False Negative Pair: {bioID1} - {bioID2}, Subtypes: {subtype1} - {subtype2}')

# Split based on subfamily
idx_split = []
for bioID1 in tqdm(records, desc='Creating dataset'):
    for bioID2 in records:
        fam1 = records[bioID1]['subtype']
        fam2 = records[bioID2]['subtype']
        tuple_fam = tuple(sorted([fam1, fam2]) )
        if tuple_fam == ('Ma', 'Mb'):
            idx_split.append(False)
        else:
            idx_split.append(True)
        
X_train = X[np.array(idx_split)]
y_train = y[np.array(idx_split)]
X_test = X[~np.array(idx_split)]
y_test = y[~np.array(idx_split)]
logger.info(f'Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}')
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
balanced_acc = sklearn.metrics.balanced_accuracy_score(y_test, y_pred)
f1 = sklearn.metrics.f1_score(y_test, y_pred)
logger.info(f'Balanced Accuracy: {balanced_acc}')
logger.info(f'F1 Score: {f1}')
cm = confusion_matrix(y_test, y_pred)
cm = pd.DataFrame(cm, index=['Actual_Negative', 'Actual_Positive'], columns=['Predicted_Negative', 'Predicted_Positive'])
logger.info(f'Confusion Matrix:\n{cm}')
