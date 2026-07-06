# ======================================================================================
# arvore_decisao.py
# ======================================================================================
# Responsabilidade:
# - Treinar o modelo de Árvore de Decisão
# - Gerar predições para treino e teste
# - Calcular métricas iniciais de avaliação
# ======================================================================================

import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.tree import DecisionTreeClassifier

def treinar_arvore_decisao(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int,
    max_depth: int,
    min_samples_leaf: int,
) -> DecisionTreeClassifier:
    """
    Treina um modelo de Árvore de Decisão.

    O class_weight='balanced' é usado porque o target possui classes desbalanceadas.
    """

    print("\n" + "=" * 80)
    print("TREINAMENTO DO MODELO - ÁRVORE DE DECISÃO")
    print("=" * 80)

    print(f"Formato de X_train: {X_train.shape}")
    print(f"Formato de y_train: {y_train.shape}")

    modelo = DecisionTreeClassifier(
        criterion="gini",
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced",
        random_state=random_state,
    )

    print("\nParâmetros do modelo:")
    print(f"criterion: gini")
    print(f"max_depth: {max_depth}")
    print(f"min_samples_leaf: {min_samples_leaf}")
    print("class_weight: balanced")
    print(f"random_state: {random_state}")

    modelo.fit(X_train, y_train)

    print("\nModelo Árvore de Decisão treinado com sucesso.")

    return modelo

def avaliar_arvore_decisao(
    modelo: DecisionTreeClassifier,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[dict[str, float | str], np.ndarray]:
    """
    Avalia o modelo de Árvore de Decisão no treino e no teste.
    """

    print("\n" + "=" * 80)
    print("AVALIAÇÃO DO MODELO - ÁRVORE DE DECISÃO")
    print("=" * 80)

    y_pred_train = modelo.predict(X_train)
    y_pred_test = modelo.predict(X_test)

    acuracia_treino = accuracy_score(y_train, y_pred_train)
    acuracia_teste = accuracy_score(y_test, y_pred_test)

    balanced_accuracy = balanced_accuracy_score(y_test, y_pred_test)

    macro_precision = precision_score(
        y_test,
        y_pred_test,
        average="macro",
        zero_division=0,
    )

    macro_recall = recall_score(
        y_test,
        y_pred_test,
        average="macro",
        zero_division=0,
    )

    macro_f1 = f1_score(
        y_test,
        y_pred_test,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_test,
        y_pred_test,
        average="weighted",
        zero_division=0,
    )

    metricas = {
        "modelo": "Árvore de Decisão",
        "accuracy_train": acuracia_treino,
        "accuracy_test": acuracia_teste,
        "balanced_accuracy_test": balanced_accuracy,
        "macro_precision_test": macro_precision,
        "macro_recall_test": macro_recall,
        "macro_f1_test": macro_f1,
        "weighted_f1_test": weighted_f1,
    }

    print("\nMétricas principais:")
    print(f"Accuracy treino: {acuracia_treino:.4f}")
    print(f"Accuracy teste: {acuracia_teste:.4f}")
    print(f"Balanced accuracy teste: {balanced_accuracy:.4f}")
    print(f"Macro precision teste: {macro_precision:.4f}")
    print(f"Macro recall teste: {macro_recall:.4f}")
    print(f"Macro F1-score teste: {macro_f1:.4f}")
    print(f"Weighted F1-score teste: {weighted_f1:.4f}")

    print("\nClassification report - Teste:")
    print(
        classification_report(
            y_test,
            y_pred_test,
            zero_division=0,
        )
    )

    return metricas, y_pred_test

def registrar_treinamento_arvore_decisao(
    caminho_log: Path,
    max_depth: int,
    min_samples_leaf: int,
    random_state: int,
    X_train_shape: tuple[int, int],
    X_test_shape: tuple[int, int],
    y_train_shape: tuple[int],
    y_test_shape: tuple[int],
    metricas: dict,
    observacao: str = "",
) -> None:
    """
    Registra em arquivo .txt os parâmetros e métricas do treinamento
    da Árvore de Decisão.
    """

    caminho_log.parent.mkdir(parents=True, exist_ok=True)

    data_execucao = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(caminho_log, "a", encoding="utf-8") as arquivo:
        arquivo.write("\n")
        arquivo.write("=" * 80)
        arquivo.write("\n")
        arquivo.write("EXPERIMENTO - ÁRVORE DE DECISÃO\n")
        arquivo.write("=" * 80)
        arquivo.write("\n")

        arquivo.write(f"Data/hora da execução: {data_execucao}\n\n")

        arquivo.write("Parâmetros utilizados:\n")
        arquivo.write(f"- MAX_DEPTH_ARVORE_DECISAO: {max_depth}\n")
        arquivo.write(f"- MIN_SAMPLES_LEAF_ARVORE_DECISAO: {min_samples_leaf}\n")
        arquivo.write(f"- RANDOM_STATE: {random_state}\n")
        arquivo.write("- criterion: gini\n")
        arquivo.write("- class_weight: balanced\n\n")

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