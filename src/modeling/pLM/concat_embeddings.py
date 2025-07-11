
# Third-party modules
import numpy as np
import pandas as pd
import sklearn as sk
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

# Custom modules
from src.entities.ppi import PPI
from src.misc.logger import logger

# Load data
ppis = [ppi for ppi in PPI.iterate() if ppi.interact() != '?']

# Iterate over ESM2 models and seeds
esm2_models = ['8M', '35M', '150M', '650M','3B']
seeds = range(10)
results = pd.DataFrame(columns=[
    'Model',
    'Partition',
    'Seed',
    'Balanced Accuracy',
])
for model in esm2_models:
    for seed in seeds:

        # Collect data
        X, y = [], []
        for ppi in ppis:
            emb1 = ppi.p1.esm2_embeddings[model].copy().mean(0)
            emb2 = ppi.p2.esm2_embeddings[model].copy().mean(0)
            x = np.concatenate([emb1, emb2])
            X.append(x)
            y.append(ppi.interact())
        X = np.stack(X)
        y = np.array(y)

        # Split data and evaluate models
        splitting_strategies = ['random', 'INTRA0-INTER', 'INTRA1-INTER', 'INTRA0-INTRA1' ]
        intra0 = [ppi.partition == 'INTRA0' for ppi in ppis]
        intra1 = [ppi.partition == 'INTRA1' for ppi in ppis]
        inter = [ppi.partition == 'INTER' for ppi in ppis]
        assert len(X) == len(y) == len(intra0) == len(intra1) == len(inter)
        for strategy in splitting_strategies:
            if strategy == 'random':
                X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(X, y, test_size=0.2, random_state=42)
            elif strategy == 'INTRA0-INTER':
                if len(intra0) > len(inter):
                    X_train, y_train = X[intra0], y[intra0]
                    X_test, y_test = X[inter], y[inter]
                else:
                    X_train, y_train = X[inter], y[inter]
                    X_test, y_test = X[intra0], y[intra0]
            elif strategy == 'INTRA1-INTER':
                if len(intra1) > len(inter):
                    X_train, y_train = X[intra1], y[intra1]
                    X_test, y_test = X[inter], y[inter]
                else:
                    X_train, y_train = X[inter], y[inter]
                    X_test, y_test = X[intra1], y[intra1]
            elif strategy == 'INTRA0-INTRA1':
                if len(intra0) > len(intra1):
                    X_train, y_train = X[intra0], y[intra0]
                    X_test, y_test = X[intra1], y[intra1]
                else:
                    X_train, y_train = X[intra1], y[intra1]
                    X_test, y_test = X[intra0], y[intra0]

            # Train a random forest model
            rf = sk.ensemble.RandomForestClassifier(n_estimators=100, random_state=seed)
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_test)
            y_proba = rf.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = sk.metrics.roc_curve(y_test, y_proba)
            roc_auc = sk.metrics.auc(fpr, tpr)
            balanced_accuracy = sk.metrics.balanced_accuracy_score(y_test, y_pred)
            logger.info(f"Model {model}, seed {seed}")
            logger.info(f"Strategy: {strategy}, ROC AUC: {roc_auc:.2f}, Balanced Accuracy: {balanced_accuracy:.2f}")

            # Store results
            results = pd.concat([
                results,
                pd.DataFrame({
                    'Model': [model],
                    'Partition': [strategy],
                    'Seed': [seed],
                    'Balanced Accuracy': [balanced_accuracy],
                })
            ], ignore_index=True)

# Plot results
sns.boxplot(data=results, x='Partition', y='Balanced Accuracy', hue='Model')
plt.title("Balanced Accuracy across Partition Strategies")
plt.savefig('balanced_accuracy_results.png', dpi=300, bbox_inches='tight')