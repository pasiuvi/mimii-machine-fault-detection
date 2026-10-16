"""
Question 3a – classical baseline: MFCC statistics + tuned classical models.

Features per clip (computed from the SAME cached 128-band log-mel, channel 0):
    20 MFCCs -> mean (20) + std (20) over time
    delta-MFCCs -> std (20)                    (how much the timbre changes)
    machine type one-hot (2)                   (the same extra input the CNN gets)
  = 62 features, standardised with TRAIN statistics.

Search space (selected on VALIDATION TPR@FPR5, same split as the CNN):
    Logistic regression : C in {0.01, 0.1, 1, 10, 100}, class_weight balanced         ->  5
    SVM (RBF)           : C in {0.1, 1, 10} x gamma in {scale, 0.01, 0.001}           ->  9
    Random forest       : n_estimators {300, 600} x max_depth {None, 12} x
                          min_samples_leaf {1, 5}, class_weight balanced_subsample    ->  8
    HistGradientBoosting: learning_rate {0.03, 0.1} x max_depth {3, None} x
                          l2 {0, 1}                                                   ->  8
                                                                              total   = 30 configs
"""
from __future__ import annotations

import itertools

import librosa
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from src import config


def mfcc_features(split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.load(config.FEATURE_DIR / f"{split}.npy", mmap_mode="r")
    idx = pd.read_csv(config.FEATURE_DIR / f"{split}_index.csv")
    feats = []
    for i in range(len(X)):
        S = np.asarray(X[i], dtype=np.float32)
        mf = librosa.feature.mfcc(S=S, n_mfcc=20)
        d = librosa.feature.delta(mf)
        feats.append(np.concatenate([mf.mean(1), mf.std(1), d.std(1)]))
    F = np.stack(feats)
    onehot = np.eye(2)[idx.machine_type.map(config.MACHINE_TYPE_INDEX).to_numpy()]
    return np.hstack([F, onehot]).astype(np.float32), idx.label.to_numpy(), idx.machine_type.to_numpy()


def search_space():
    for C in [0.01, 0.1, 1, 10, 100]:
        yield "LogReg", {"C": C}, lambda C=C: LogisticRegression(C=C, class_weight="balanced", max_iter=5000)
    for C, g in itertools.product([0.1, 1, 10], ["scale", 0.01, 0.001]):
        yield "SVM-RBF", {"C": C, "gamma": g}, lambda C=C, g=g: SVC(C=C, gamma=g, class_weight="balanced")
    for n, d, leaf in itertools.product([300, 600], [None, 12], [1, 5]):
        yield "RandomForest", {"n_estimators": n, "max_depth": d, "min_samples_leaf": leaf}, \
            lambda n=n, d=d, leaf=leaf: RandomForestClassifier(
                n_estimators=n, max_depth=d, min_samples_leaf=leaf, class_weight="balanced_subsample",
                n_jobs=-1, random_state=config.SEED)
    for lr, d, l2 in itertools.product([0.03, 0.1], [3, None], [0.0, 1.0]):
        yield "HistGB", {"learning_rate": lr, "max_depth": d, "l2_regularization": l2}, \
            lambda lr=lr, d=d, l2=l2: HistGradientBoostingClassifier(
                learning_rate=lr, max_depth=d, l2_regularization=l2, class_weight="balanced",
                max_iter=300, random_state=config.SEED)


def scores(model, X) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)
