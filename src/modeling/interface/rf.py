
#known error with 54kjsdhfksh because of its weird structure
# use performance class
"""
===============================================================================
Title:      Random forest model on interface features
Outline:    Interface features are used to train a random forest model to
            predict the interation of MIKC PPIs using different splitting
            strategies from more to less data leakage:
            - Random split
            - INTRA0-INTER
            - INTRA1-INTER
            - INTRA0-INTRA1
Author:     Alejandro Sánchez Cano
Date:       07/07/2025
Time:       3 min
===============================================================================
"""

# Built-in modules
from collections import defaultdict

# Third-party modules
import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt

# Custom modules
from src.entities.ppi import PPI
from src.misc.logger import logger

# Load data
ppis = [ppi for ppi in PPI.iterate()]
valid_ppis = []
for idx, ppi in enumerate(ppis):
    if not hasattr(ppi, 'interface_features'):
        logger.warning(f"PPI {idx} does not have interface features. Skipping.")
    elif ppi.interact() != '?':
        valid_ppis.append(ppi)
X = pd.DataFrame([ppi.interface_features.copy() for ppi in valid_ppis])
y = pd.Series([ppi.interact() for ppi in valid_ppis])
X_std = sk.preprocessing.StandardScaler().fit_transform(X)

# Splits
scores = defaultdict(dict)
splitting_strategies = ['random', 'INTRA0-INTER', 'INTRA1-INTER', 'INTRA0-INTRA1' ]
intra0 = [ppi.partition == 'INTRA0' for ppi in valid_ppis]
intra1 = [ppi.partition == 'INTRA1' for ppi in valid_ppis]
inter = [ppi.partition == 'INTER' for ppi in valid_ppis]
assert len(X_std) == len(y) == len(intra0) == len(intra1) == len(inter)
for strategy in splitting_strategies:
    if strategy == 'random':
        X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(X_std, y, test_size=0.2, random_state=42)
    elif strategy == 'INTRA0-INTER':
        if len(intra0) > len(inter):
            X_train, y_train = X_std[intra0], y[intra0]
            X_test, y_test = X_std[inter], y[inter]
        else:
            X_train, y_train = X_std[inter], y[inter]
            X_test, y_test = X_std[intra0], y[intra0]
    elif strategy == 'INTRA1-INTER':
        if len(intra1) > len(inter):
            X_train, y_train = X_std[intra1], y[intra1]
            X_test, y_test = X_std[inter], y[inter]
        else:
            X_train, y_train = X_std[inter], y[inter]
            X_test, y_test = X_std[intra1], y[intra1]
    elif strategy == 'INTRA0-INTRA1':
        if len(intra0) > len(intra1):
            X_train, y_train = X_std[intra0], y[intra0]
            X_test, y_test = X_std[intra1], y[intra1]
        else:
            X_train, y_train = X_std[intra1], y[intra1]
            X_test, y_test = X_std[intra0], y[intra0]

    # Train a random forest model and plot ROC curve
    rf = sk.ensemble.RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = sk.metrics.roc_curve(y_test, y_proba)
    roc_auc = sk.metrics.auc(fpr, tpr)
    balanced_accuracy = sk.metrics.balanced_accuracy_score(y_test, y_pred)
    logger.info(f"Strategy: {strategy}, ROC AUC: {roc_auc:.2f}, Balanced Accuracy: {balanced_accuracy:.2f}")
    scores[strategy]['roc_auc'] = roc_auc
    scores[strategy]['balanced_accuracy'] = balanced_accuracy
    scores[strategy]['fpr'] = fpr
    scores[strategy]['tpr'] = tpr

# Plot ROC curves
plt.figure(figsize=(10, 6))
for strategy, score in scores.items():
    plt.plot(score['fpr'], score['tpr'], label=f"{strategy} (AUC = {score['roc_auc']:.2f})")
plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Different Splitting Strategies')
plt.legend()
plt.show()