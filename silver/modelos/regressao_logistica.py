# ======================================================================================
# regressao_logistica.py
# ======================================================================================
# Responsabilidade:
# - Treinar um modelo linear equivalente à Regressão Logística usando SGDClassifier
# - Avaliar o modelo
# - Registrar os experimentos em arquivo .txt
# ======================================================================================

from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

def calcular_alpha_sgd(
    C: float,
    quantidade_linhas_treino: int,
) -> float:
    """
    Converte o parâmetro C em alpha para o SGDClassifier.

    Para bases grandes, evita alpha muito pequeno, pois isso pode deixar
    o treinamento lento e instável.
    """

    if C <= 0:
        return 0.0001

    if quantidade_linhas_treino <= 0:
        return 0.0001

    alpha_calculado = 1 / (C * quantidade_linhas_treino)

    alpha_minimo = 0.0001

    if alpha_calculado < alpha_minimo:
        return alpha_minimo

    return alpha_calculado

def treinar_regressao_logistica(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int,
    max_iter: int,
    C: float,
    solver: str,
    l1_ratio: float,
) -> SGDClassifier:
    """
    Treina um modelo linear com loss='log_loss'.

    Esse modelo funciona como uma Regressão Logística treinada por descida
    de gradiente estocástica, sendo mais leve para bases grandes.
    """

    print("\n" + "=" * 80)
    print("TREINAMENTO DO MODELO - REGRESSÃO LOGÍSTICA VIA SGD")
    print("=" * 80)

    print(f"Formato de X_train: {X_train.shape}")
    print(f"Formato de y_train: {y_train.shape}")

    loss = "log_loss"
    
    if l1_ratio <= 0:
        penalty = "l2"
    else:
        penalty = "elasticnet"

    alpha = calcular_alpha_sgd(
        C=C,
        quantidade_linhas_treino=X_train.shape[0],
    )

    print("\nParâmetros do modelo:")
    print(f"max_iter: {max_iter}")
    print(f"C recebido: {C}")
    print(f"alpha calculado para SGDClassifier: {alpha}")
    print(f"solver recebido: {solver}")
    print(f"loss: {loss}")
    print(f"penalty: {penalty}")
    print(f"l1_ratio: {l1_ratio}")
    print("class_weight: None")
    print("early_stopping: True")
    print("n_jobs: 1")
    print(f"random_state: {random_state}")

    modelo = SGDClassifier(
        loss=loss,
        penalty=penalty,
        alpha=alpha,
        l1_ratio=l1_ratio,
        max_iter=max_iter,
        tol=1e-3,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=5,
        class_weight=None,
        random_state=random_state,
        n_jobs=1,
        verbose=0,
    )

    modelo.fit(
        X_train,
        y_train,
    )
    print("\nModelo Regressão Logística via SGD treinado com sucesso.")

    return modelo

def avaliar_regressao_logistica(
    modelo: SGDClassifier,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[dict, object]:
    """
    Avalia o modelo Regressão Logística via SGD.
    """

    print("\n" + "=" * 80)
    print("AVALIAÇÃO DO MODELO - REGRESSÃO LOGÍSTICA VIA SGD")
    print("=" * 80)

    y_pred_train = modelo.predict(X_train)
    y_pred_test = modelo.predict(X_test)

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
        "modelo": "Regressão Logística via SGD",
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


def registrar_treinamento_regressao_logistica(
    caminho_log: Path,
    max_iter: int,
    C: float,
    solver: str,
    l1_ratio: float,
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
    da Regressão Logística via SGD.
    """

    caminho_log.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data_execucao = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    alpha = calcular_alpha_sgd(
        C=C,
        quantidade_linhas_treino=X_train_shape[0],
    )

    with open(caminho_log, "a", encoding="utf-8") as arquivo:
        arquivo.write("\n")
        arquivo.write("=" * 80)
        arquivo.write("\n")
        arquivo.write("EXECUÇÃO DO MODELO FINAL - REGRESSÃO LOGÍSTICA VIA SGD\n")
        arquivo.write("=" * 80)
        arquivo.write("\n")

        arquivo.write(f"Data/hora da execução: {data_execucao}\n\n")

        arquivo.write("Parâmetros utilizados:\n")
        arquivo.write(f"- MAX_ITER_REGRESSAO_LOGISTICA: {max_iter}\n")
        arquivo.write(f"- C_REGRESSAO_LOGISTICA: {C}\n")
        arquivo.write(f"- ALPHA_CALCULADO_SGD: {alpha}\n")
        arquivo.write(f"- SOLVER_REGRESSAO_LOGISTICA: {solver}\n")
        arquivo.write("- loss: log_loss\n")
        arquivo.write("- penalty: elasticnet\n")
        arquivo.write(f"- L1_RATIO_REGRESSAO_LOGISTICA: {l1_ratio}\n")
        arquivo.write(f"- RANDOM_STATE: {random_state}\n")
        arquivo.write("- class_weight: balanced\n")
        arquivo.write("- n_jobs: -1\n\n")

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

