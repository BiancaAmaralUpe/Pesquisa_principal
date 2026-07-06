# ======================================================================================
# random_forest.py
# ======================================================================================
# Responsabilidade:
# - Treinar o modelo Random Forest
# - Avaliar o modelo Random Forest
# - Registrar os experimentos em arquivo .txt
# ======================================================================================

from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

# ===================================================================================== #
# def treinar_random_forest
# ===================================================================================== #
def treinar_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int,
    n_estimators: int,
    max_depth: int,
    min_samples_leaf: int,
    max_features: str,
    n_jobs: int,
) -> RandomForestClassifier:
    """
    Treina o modelo Random Forest.
    """

    print("\n" + "=" * 80)
    print("TREINAMENTO DO MODELO - RANDOM FOREST")
    print("=" * 80)

    print(f"Formato de X_train: {X_train.shape}")
    print(f"Formato de y_train: {y_train.shape}")

    print("\nParâmetros do modelo:")
    print(f"n_estimators: {n_estimators}")
    print(f"criterion: gini")
    print(f"max_depth: {max_depth}")
    print(f"min_samples_leaf: {min_samples_leaf}")
    print(f"max_features: {max_features}")
    print("class_weight: balanced_subsample")
    print(f"random_state: {random_state}")
    print(f"n_jobs: {n_jobs}")

    modelo = RandomForestClassifier(
        n_estimators=n_estimators,
        criterion="gini",
        max_depth=10,
        min_samples_leaf=min_samples_leaf,
        max_features="sqrt",
        random_state=random_state,
        n_jobs=n_jobs,
        max_samples=0.2,
        warm_start=True,
        ccp_alpha=0.01
    )

    modelo.fit(X_train, y_train)

    print("\nModelo Random Forest treinado com sucesso.")

    return modelo

# ===================================================================================== #
# def treinar_random_forest
# ===================================================================================== #
def avaliar_random_forest(
    modelo: RandomForestClassifier,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[dict, pd.Series]:
    """
    Avalia o modelo Random Forest.
    """

    print("\n" + "=" * 80)
    print("AVALIAÇÃO DO MODELO - RANDOM FOREST")
    print("=" * 80)

    y_pred_train = modelo.predict(X_train)
    y_pred_test = modelo.predict(X_test)

    accuracy_train = accuracy_score(y_train, y_pred_train)
    accuracy_test = accuracy_score(y_test, y_pred_test)

    balanced_accuracy_test = balanced_accuracy_score(y_test, y_pred_test)
    # ===================================================================================== #
    # def treinar_random_forest
    # ===================================================================================== #
    macro_precision_test = precision_score(
        y_test,
        y_pred_test,
        average="macro",
        zero_division=0,
    )
    # ===================================================================================== #
    # def treinar_random_forest
    # ===================================================================================== #
    macro_recall_test = recall_score(
        y_test,
        y_pred_test,
        average="macro",
        zero_division=0,
    )
    # ===================================================================================== #
    # def treinar_random_forest
    # ===================================================================================== #
    macro_f1_test = f1_score(
        y_test,
        y_pred_test,
        average="macro",
        zero_division=0,
    )
    # ===================================================================================== #
    # def treinar_random_forest
    # ===================================================================================== #
    weighted_f1_test = f1_score(
        y_test,
        y_pred_test,
        average="weighted",
        zero_division=0,
    )

    metricas = {
        "modelo": "Random Forest",
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

# ===================================================================================== #
# def treinar_random_forest
# ===================================================================================== #
def registrar_treinamento_random_forest(
    caminho_log: Path,
    n_estimators: int,
    max_depth: int,
    min_samples_leaf: int,
    max_features: str,
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
    da Random Forest.
    """

    caminho_log.parent.mkdir(parents=True, exist_ok=True)

    data_execucao = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(caminho_log, "a", encoding="utf-8") as arquivo:
        arquivo.write("\n")
        arquivo.write("=" * 80)
        arquivo.write("\n")
        arquivo.write("EXECUÇÃO DO MODELO FINAL - RANDOM FOREST\n")
        arquivo.write("=" * 80)
        arquivo.write("\n")

        arquivo.write(f"Data/hora da execução: {data_execucao}\n\n")

        arquivo.write("Parâmetros utilizados:\n")
        arquivo.write(f"- N_ESTIMATORS_RANDOM_FOREST: {n_estimators}\n")
        arquivo.write(f"- MAX_DEPTH_RANDOM_FOREST: {max_depth}\n")
        arquivo.write(f"- MIN_SAMPLES_LEAF_RANDOM_FOREST: {min_samples_leaf}\n")
        arquivo.write(f"- MAX_FEATURES_RANDOM_FOREST: {max_features}\n")
        arquivo.write(f"- RANDOM_STATE: {random_state}\n")
        arquivo.write(f"- N_JOBS_RANDOM_FOREST: {n_jobs}\n")
        arquivo.write("- criterion: gini\n")
        arquivo.write("- class_weight: balanced_subsample\n\n")

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