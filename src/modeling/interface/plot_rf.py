"""
===============================================================================
Title:      Plot RF results
Outline:    Fetch the best result of the Optuna hyperparameter optimization per
            splitting strategy and plot the ROC and PR curves.
Author:     Alejandro Sánchez Cano
Date:       07/07/2025
Time:       ~2s/trial -> 2h for 500 trials and 4 strategies
===============================================================================
"""

# Built-in modules
from collections import defaultdict

# Third-party modules
import optuna
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt

# Custom modules
from src.misc import path
from src.modeling.utils.performance import Performance

splitting_strategies = [
    'random', 
    'INTRA0-INTER', 
    'INTRA1-INTER', 
    'INTRA0-INTRA1'
]

# Fetch best eprfromances
performances = defaultdict(dict)
for strategy in splitting_strategies:
    filename = 'rf'
    db_path = path.OPTUNA / filename / f"{strategy}.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        direction="maximize",
        storage=f"sqlite:///{str(db_path)}",
        load_if_exists=True,
        study_name=f"{str(filename)}_{strategy}"
    )
    best_trial = study.best_trial
    best_performance = best_trial.user_attrs['performance']
    performances[strategy] = Performance(**best_performance)    

    print(f"Best performance for {strategy}: {performances[strategy].balanced_accuracy}")

# Plot ROC curves
plt.figure(figsize=(10, 6))
for strategy, perf in performances.items():
    plt.plot(perf.fpr, perf.tpr, label=f"{strategy} (AUC = {perf.auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Different Splitting Strategies')
plt.legend()
plt.savefig('roc_curves.png')

# Plot PR curves
plt.figure(figsize=(10, 6))
for strategy, perf in performances.items():
    precision, recall, _ = sk.metrics.precision_recall_curve(perf.true, perf.prob)
    plt.plot(recall, precision, label=f"{strategy} (AUPRC = {perf.auprc:.2f})")
plt.axhline(y=np.mean(perf.true), color='k', linestyle='--', label='Random Guessing')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('PR Curves for Different Splitting Strategies')
plt.legend()
plt.savefig('pr_curves.png')