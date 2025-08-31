"""
===============================================================================
Title:      Random forest model on interface features
Outline:    Interface features are used to train a random forest (RF) model to
            predict the interation of MIKC PPIs using different splitting
            strategies from more to less data leakage:
            - Random split
            - INTRA0-INTER
            - INTRA1-INTER
            - INTRA0-INTRA1
            Use optuna to optimize the hyperparameters of the RF model.
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
import pandas as pd
import sklearn as sk

# Custom modules
from src.misc import path
from src.entities.ppi import PPI
from src.misc.logger import logger
from src.modeling.utils.performance import Performance

# Load data
ppis = [ppi for ppi in PPI.iterate('interface_features', 'partition', 'interaction', 'origin')]
valid_ppis = []
for idx, ppi in enumerate(ppis):
    if ppi.interface_features is None:
        logger.warning(f"PPI {idx} does not have interface features. Skipping.") # Known error with PPI 5483 -> weird structure
    elif ppi.interact() != '?':
        valid_ppis.append(ppi)
X = pd.DataFrame([ppi.interface_features.copy() for ppi in valid_ppis])
y = pd.Series([ppi.interact() for ppi in valid_ppis])
X_std = sk.preprocessing.StandardScaler().fit_transform(X)
intra0 = [ppi.partition == 'INTRA0' for ppi in valid_ppis]
intra1 = [ppi.partition == 'INTRA1' for ppi in valid_ppis]
inter = [ppi.partition == 'INTER' for ppi in valid_ppis]
assert len(X_std) == len(y) == len(intra0) == len(intra1) == len(inter)

# Optimization function
def train_objective(
    trial: optuna.Trial, 
    splitted_sets: tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]
    ) -> float:
    '''
    Objective function for Optuna to optimize hyperparameters using 
    balanced accuracy on the validation set as metric.

    Parameters
    ----------
    trial : optuna.Trial
        The current trial object.
    splitted_sets : tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]
        X_train, X_test, y_train, y_test.

    Returns
    -------
    float
        The balanced accuracy of the model.
    '''

    # Hyperparameter space
    n_estimators = trial.suggest_int("n_estimators", 50, 1000, step=10)
    max_depth = trial.suggest_int("max_depth", 5, 70, step=5)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 20, step=2)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20, step=1)
    max_features = trial.suggest_categorical("max_features", ["auto", "sqrt", "log2"])

    # Define RF model
    rf = sk.ensemble.RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )

    # Train and evaluate
    X_train, X_test, y_train, y_test = splitted_sets
    rf.fit(X_train, y_train)
    y_proba = rf.predict_proba(X_test)[:, 1]
    y_test = y_test.to_numpy()
    performance = Performance(true = y_test, probs = y_proba)
    trial.set_user_attr(                    # Performance object is not  
        'performance',                      # JSON serializable
        {'true': y_test.tolist(), 'probs': y_proba.tolist()}
        )
    return performance.balanced_accuracy

# Split function
def split(strategy: str) -> tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]:
    """
    Split the data into training and testing sets based on the given splitting 
    strategy.

    Parameters
    ----------
    strategy : str
        The splitting strategy to use.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]
        The training and testing sets.
    """

    if strategy == 'random':
        split = sk.model_selection.train_test_split(
            X_std, y, test_size=0.2, random_state=42
            )
        X_train, X_test, y_train, y_test = split
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

    return X_train, X_test, y_train, y_test

# Default hyperparameters
default = {
    "n_estimators": 100,
    "max_depth": 50,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": "auto"
}

# Train and optimize
n_trials = 500
performances = defaultdict(dict)
splitting_strategies = [
    'random', 
    'INTRA0-INTER', 
    'INTRA1-INTER', 
    'INTRA0-INTRA1'
]
for strategy in splitting_strategies:
    splitted_sets = split(strategy)
    filename = 'rf'
    db_path = path.OPTUNA / filename / f"{strategy}.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        direction="maximize",
        storage=f"sqlite:///{str(db_path)}",
        load_if_exists=True,
        study_name=f"{str(filename)}_{strategy}"
    )
    objective = lambda trial: train_objective(trial, splitted_sets)
    study.enqueue_trial(default)
    study.optimize(
        objective, 
        n_trials=n_trials - len(study.trials) if n_trials - len(study.trials) > 0 else 0, 
        show_progress_bar=True, 
        n_jobs=4)
    best_trial = study.best_trial
    best_performance = best_trial.user_attrs['performance']
    performances[strategy] = Performance(**best_performance)