import optuna
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier

from Pesquisa_principal.silver.otimizadores.dataset_balanceado import dataset

from datetime import datetime
from pathlib import Path
from typing import Any

from sklearn.preprocessing import LabelEncoder

def objective(trial):

    _dataset = dataset()

    X_train, y_train = _dataset['X_train'], _dataset['y_train']

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)

    num_classes = len(label_encoder.classes_)

    model = XGBClassifier(
        booster='dart',
        n_estimators=trial.suggest_int("n_estimators", 50, 500),
        max_depth=trial.suggest_int("max_depth", 3, 12),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        objective="multi:softprob",
        eval_metric="mlogloss",
        num_class=num_classes,
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )

    score = cross_val_score(
        model,
        X_train,
        y_train_encoded,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
    ).mean()

    return score

def xgboost_study():

    print("\n" + "=" * 80)
    print("OTIMIZANDO MODELO - XGBOOST")
    print("=" * 80)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=1)

    print("\n" + "=" * 80)
    print("OTIMIZANDO MODELO - MELHORES PARAMETROS - XGBOOST")
    print(study.best_params)
    print("=" * 80)

    print("\n" + "=" * 80)
    print("OTIMIZANDO MODELO - MELHOR ACURÁCIA - XGBOOST")
    print(study.best_value)
    print("=" * 80)
