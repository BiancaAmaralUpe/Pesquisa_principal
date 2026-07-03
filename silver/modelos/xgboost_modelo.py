# ======================================================================================
# xgboost_modelo.py
# ======================================================================================
# Responsabilidade:
# - Treinar o modelo XGBoost
# - Avaliar o modelo XGBoost
# - Registrar os experimentos em arquivo .txt
# ======================================================================================

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight


def obter_classe_xgboost() -> Any:
    """
    Importa o XGBClassifier somente quando o modelo for executado.

    Isso evita quebrar a pipeline caso o XGBoost ainda não esteja instalado
    e o modelo esteja desativado na orquestração.
    """

    try:
        from xgboost import XGBClassifier

        return XGBClassifier

    except ModuleNotFoundError as erro:
        raise ModuleNotFoundError(
            "A biblioteca xgboost não está instalada. "
            "Instale com: pip install xgboost"
        ) from erro


def treinar_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    subsample: float,
    colsample_bytree: float,
    tree_method: str,
    n_jobs: int,
) -> Any:
    """
    Treina o modelo XGBoost.

    O XGBoost trabalha melhor com classes numéricas.
    Por isso, o y_train é codificado com LabelEncoder.
    """

    print("\n" + "=" * 80)
    print("TREINAMENTO DO MODELO - XGBOOST")
    print("=" * 80)

    print(f"Formato de X_train: {X_train.shape}")
    print(f"Formato de y_train: {y_train.shape}")

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)

    sample_weight = compute_sample_weight(
        class_weight="balanced",
        y=y_train_encoded,
    )

    quantidade_classes = len(label_encoder.classes_)

    XGBClassifier = obter_classe_xgboost()

    print("\nParâmetros do modelo:")
    print(f"n_estimators: {n_estimators}")
    print(f"max_depth: {max_depth}")
    print(f"learning_rate: {learning_rate}")
    print(f"subsample: {subsample}")
    print(f"colsample_bytree: {colsample_bytree}")
    print(f"tree_method: {tree_method}")
    print("objective: multi:softprob")
    print("eval_metric: mlogloss")
    print("sample_weight: balanced")
    print(f"num_class: {quantidade_classes}")
    print(f"random_state: {random_state}")
    print(f"n_jobs: {n_jobs}")

    modelo = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        objective="multi:softprob",
        eval_metric="mlogloss",
        num_class=quantidade_classes,
        tree_method=tree_method,
        random_state=random_state,
        n_jobs=n_jobs,
    )

    modelo.fit(
        X_train,
        y_train_encoded,
        sample_weight=sample_weight,
    )

    modelo.label_encoder_classes_ = label_encoder.classes_

    print("\nModelo XGBoost treinado com sucesso.")

    return modelo


def decodificar_predicoes_xgboost(
    modelo: Any,
    y_pred_encoded,
) -> pd.Series:
    """
    Converte as predições numéricas do XGBoost de volta para os nomes das classes.
    """

    if not hasattr(modelo, "label_encoder_classes_"):
        raise ValueError(
            "O modelo XGBoost não possui as classes do LabelEncoder salvas."
        )

    label_encoder = LabelEncoder()
    label_encoder.classes_ = modelo.label_encoder_classes_

    y_pred_decoded = label_encoder.inverse_transform(
        y_pred_encoded.astype(int)
    )

    return pd.Series(y_pred_decoded)


def avaliar_xgboost(
    modelo: Any,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[dict, pd.Series]:
    """
    Avalia o modelo XGBoost.
    """

    print("\n" + "=" * 80)
    print("AVALIAÇÃO DO MODELO - XGBOOST")
    print("=" * 80)

    y_pred_train_encoded = modelo.predict(X_train)
    y_pred_test_encoded = modelo.predict(X_test)

    y_pred_train = decodificar_predicoes_xgboost(
        modelo=modelo,
        y_pred_encoded=y_pred_train_encoded,
    )

    y_pred_test = decodificar_predicoes_xgboost(
        modelo=modelo,
        y_pred_encoded=y_pred_test_encoded,
    )

    y_pred_train.index = y_train.index
    y_pred_test.index = y_test.index

    accuracy_train = accuracy_score(
        y_train,
        y_pred_train,
    )

    accuracy_test = accuracy_score(
        y_test,
        y_pred_test,
    )

    balanced_accuracy_test = balanced_accuracy_score(
        y_test,
        y_pred_test,
    )

    macro_precision_test = precision_score(
        y_test,
        y_pred_test,
        average="macro",
        zero_division=0,
    )

    macro_recall_test = recall_score(
        y_test,
        y_pred_test,
        average="macro",
        zero_division=0,
    )

    macro_f1_test = f1_score(
        y_test,
        y_pred_test,
        average="macro",
        zero_division=0,
    )

    weighted_f1_test = f1_score(
        y_test,
        y_pred_test,
        average="weighted",
        zero_division=0,
    )

    metricas = {
        "modelo": "XGBoost",
        "accuracy_train": accuracy_train,
        "accuracy_test": accuracy_test,
        "balanced_accuracy_test": balanced_accuracy_test,
        "macro_precision_test": macro_precision_test,
        "macro_recall_test": macro_recall_test,
        "macro_f1_test": macro_f1_test,
        "weighted_f1_test": weighted_f1_test,
    }

    print("\nMétricas principais:")
    print(f"Accuracy treino: {accuracy_train:.4f}")
    print(f"Accuracy teste: {accuracy_test:.4f}")
    print(f"Balanced accuracy teste: {balanced_accuracy_test:.4f}")
    print(f"Macro precision teste: {macro_precision_test:.4f}")
    print(f"Macro recall teste: {macro_recall_test:.4f}")
    print(f"Macro F1-score teste: {macro_f1_test:.4f}")
    print(f"Weighted F1-score teste: {weighted_f1_test:.4f}")

    print("\nClassification report - Teste:")
    print(
        classification_report(
            y_test,
            y_pred_test,
            zero_division=0,
        )
    )

    return metricas, y_pred_test


def registrar_treinamento_xgboost(
    caminho_log: Path,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    subsample: float,
    colsample_bytree: float,
    tree_method: str,
    random_state: int,
    n_jobs: int,
    X_train_shape: tuple[int, int],
    X_test_shape: tuple[int, int],
    y_train_shape: tuple[int],
    y_test_shape: tuple[int],
    metricas: dict,
    observacao: str = "",
) -> None:
    """
    Registra em arquivo .txt os parâmetros e métricas do treinamento
    do XGBoost.
    """

    caminho_log.parent.mkdir(parents=True, exist_ok=True)

    data_execucao = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(caminho_log, "a", encoding="utf-8") as arquivo:
        arquivo.write("\n")
        arquivo.write("=" * 80)
        arquivo.write("\n")
        arquivo.write("EXECUÇÃO DO MODELO FINAL - XGBOOST\n")
        arquivo.write("=" * 80)
        arquivo.write("\n")

        arquivo.write(f"Data/hora da execução: {data_execucao}\n\n")

        arquivo.write("Parâmetros utilizados:\n")
        arquivo.write(f"- N_ESTIMATORS_XGBOOST: {n_estimators}\n")
        arquivo.write(f"- MAX_DEPTH_XGBOOST: {max_depth}\n")
        arquivo.write(f"- LEARNING_RATE_XGBOOST: {learning_rate}\n")
        arquivo.write(f"- SUBSAMPLE_XGBOOST: {subsample}\n")
        arquivo.write(f"- COLSAMPLE_BYTREE_XGBOOST: {colsample_bytree}\n")
        arquivo.write(f"- TREE_METHOD_XGBOOST: {tree_method}\n")
        arquivo.write(f"- RANDOM_STATE: {random_state}\n")
        arquivo.write(f"- N_JOBS_XGBOOST: {n_jobs}\n")
        arquivo.write("- objective: multi:softprob\n")
        arquivo.write("- eval_metric: mlogloss\n")
        arquivo.write("- sample_weight: balanced\n\n")

        arquivo.write("Formato dos dados:\n")
        arquivo.write(f"- X_train: {X_train_shape}\n")
        arquivo.write(f"- X_test: {X_test_shape}\n")
        arquivo.write(f"- y_train: {y_train_shape}\n")
        arquivo.write(f"- y_test: {y_test_shape}\n\n")

        arquivo.write("Métricas obtidas:\n")
        arquivo.write(f"- Accuracy treino: {metricas['accuracy_train']:.4f}\n")
        arquivo.write(f"- Accuracy teste: {metricas['accuracy_test']:.4f}\n")
        arquivo.write(
            f"- Balanced accuracy teste: "
            f"{metricas['balanced_accuracy_test']:.4f}\n"
        )
        arquivo.write(
            f"- Macro precision teste: "
            f"{metricas['macro_precision_test']:.4f}\n"
        )
        arquivo.write(
            f"- Macro recall teste: "
            f"{metricas['macro_recall_test']:.4f}\n"
        )
        arquivo.write(f"- Macro F1-score teste: {metricas['macro_f1_test']:.4f}\n")
        arquivo.write(
            f"- Weighted F1-score teste: "
            f"{metricas['weighted_f1_test']:.4f}\n"
        )

        if observacao:
            arquivo.write("\nObservação:\n")
            arquivo.write(f"{observacao}\n")

        arquivo.write("\n")