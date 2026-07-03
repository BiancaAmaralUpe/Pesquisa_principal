# ======================================================================================
# grafico_xgboost.py
# ======================================================================================
# Responsabilidade:
# - Avaliar possível overfitting do XGBoost
# - Testar diferentes valores de max_depth
# - Gerar gráficos de Accuracy e Macro F1-score
# - Registrar os resultados em arquivo .txt
# ======================================================================================

from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

from Pesquisa_principal.silver.modelos.xgboost_modelo import obter_classe_xgboost


def registrar_analise_overfitting_xgboost(
    caminho_log: Path,
    dataframe_resultados: pd.DataFrame,
    n_estimators: int,
    learning_rate: float,
    subsample: float,
    colsample_bytree: float,
    tree_method: str,
    random_state: int,
    n_jobs: int,
) -> None:
    """
    Registra no arquivo .txt os resultados da análise de overfitting
    do XGBoost para diferentes valores de max_depth.
    """

    caminho_log.parent.mkdir(parents=True, exist_ok=True)

    data_execucao = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(caminho_log, "a", encoding="utf-8") as arquivo:
        arquivo.write("\n")
        arquivo.write("=" * 80)
        arquivo.write("\n")
        arquivo.write("ANÁLISE DE OVERFITTING - XGBOOST\n")
        arquivo.write("=" * 80)
        arquivo.write("\n")

        arquivo.write(f"Data/hora da execução: {data_execucao}\n\n")

        arquivo.write("Parâmetros fixos utilizados:\n")
        arquivo.write(f"- N_ESTIMATORS_XGBOOST: {n_estimators}\n")
        arquivo.write(f"- LEARNING_RATE_XGBOOST: {learning_rate}\n")
        arquivo.write(f"- SUBSAMPLE_XGBOOST: {subsample}\n")
        arquivo.write(f"- COLSAMPLE_BYTREE_XGBOOST: {colsample_bytree}\n")
        arquivo.write(f"- TREE_METHOD_XGBOOST: {tree_method}\n")
        arquivo.write(f"- RANDOM_STATE: {random_state}\n")
        arquivo.write(f"- N_JOBS_XGBOOST: {n_jobs}\n")
        arquivo.write("- objective: multi:softprob\n")
        arquivo.write("- eval_metric: mlogloss\n")
        arquivo.write("- sample_weight: balanced\n\n")

        arquivo.write("Resultados por profundidade testada:\n\n")

        for _, linha in dataframe_resultados.iterrows():
            arquivo.write("-" * 80)
            arquivo.write("\n")

            arquivo.write(
                f"MAX_DEPTH_XGBOOST: {int(linha['max_depth'])}\n"
            )

            arquivo.write("\nMétricas obtidas:\n")
            arquivo.write(
                f"- Accuracy treino: {linha['accuracy_treino']:.4f}\n"
            )
            arquivo.write(
                f"- Accuracy teste: {linha['accuracy_teste']:.4f}\n"
            )
            arquivo.write(
                f"- Macro F1-score treino: {linha['macro_f1_treino']:.4f}\n"
            )
            arquivo.write(
                f"- Macro F1-score teste: {linha['macro_f1_teste']:.4f}\n"
            )

            arquivo.write("\n")

        arquivo.write("=" * 80)
        arquivo.write("\n")


def treinar_xgboost_para_overfitting(
    X_train: pd.DataFrame,
    y_train_encoded,
    sample_weight,
    quantidade_classes: int,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    subsample: float,
    colsample_bytree: float,
    tree_method: str,
    random_state: int,
    n_jobs: int,
) -> Any:
    """
    Treina uma instância do XGBoost para a análise de overfitting.
    """

    XGBClassifier = obter_classe_xgboost()

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

    return modelo


def gerar_grafico_overfitting_xgboost(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    caminho_saida: str,
    caminho_log: Path | None = None,
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    tree_method: str = "hist",
    random_state: int = 42,
    n_jobs: int = -1,
) -> None:
    """
    Gera gráficos para analisar possível overfitting do XGBoost
    comparando desempenho de treino e teste em diferentes profundidades.
    """

    Path(caminho_saida).mkdir(parents=True, exist_ok=True)

    label_encoder = LabelEncoder()

    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    quantidade_classes = len(label_encoder.classes_)

    sample_weight = compute_sample_weight(
        class_weight="balanced",
        y=y_train_encoded,
    )

    profundidades = [
        2,
        4,
        6,
        8,
        10,
    ]

    resultados = {
        "max_depth": [],
        "accuracy_treino": [],
        "accuracy_teste": [],
        "macro_f1_treino": [],
        "macro_f1_teste": [],
    }

    print("\n" + "=" * 80)
    print("ANÁLISE DE OVERFITTING - XGBOOST")
    print("=" * 80)

    for profundidade in profundidades:
        modelo = treinar_xgboost_para_overfitting(
            X_train=X_train,
            y_train_encoded=y_train_encoded,
            sample_weight=sample_weight,
            quantidade_classes=quantidade_classes,
            n_estimators=n_estimators,
            max_depth=profundidade,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            tree_method=tree_method,
            random_state=random_state,
            n_jobs=n_jobs,
        )

        y_pred_train = modelo.predict(X_train)
        y_pred_test = modelo.predict(X_test)

        accuracy_treino = accuracy_score(
            y_train_encoded,
            y_pred_train,
        )

        accuracy_teste = accuracy_score(
            y_test_encoded,
            y_pred_test,
        )

        macro_f1_treino = f1_score(
            y_train_encoded,
            y_pred_train,
            average="macro",
            zero_division=0,
        )

        macro_f1_teste = f1_score(
            y_test_encoded,
            y_pred_test,
            average="macro",
            zero_division=0,
        )

        resultados["max_depth"].append(profundidade)
        resultados["accuracy_treino"].append(accuracy_treino)
        resultados["accuracy_teste"].append(accuracy_teste)
        resultados["macro_f1_treino"].append(macro_f1_treino)
        resultados["macro_f1_teste"].append(macro_f1_teste)

        print(
            f"max_depth={profundidade} | "
            f"acc_treino={accuracy_treino:.4f} | "
            f"acc_teste={accuracy_teste:.4f} | "
            f"f1_treino={macro_f1_treino:.4f} | "
            f"f1_teste={macro_f1_teste:.4f}"
        )

    dataframe_resultados = pd.DataFrame(resultados)

    caminho_accuracy = (
        Path(caminho_saida)
        / "grafico_overfitting_xgboost_accuracy.png"
    )

    caminho_macro_f1 = (
        Path(caminho_saida)
        / "grafico_overfitting_xgboost_macro_f1.png"
    )

    plt.figure(figsize=(10, 6))
    plt.plot(
        dataframe_resultados["max_depth"],
        dataframe_resultados["accuracy_treino"],
        marker="o",
        label="Treino",
    )
    plt.plot(
        dataframe_resultados["max_depth"],
        dataframe_resultados["accuracy_teste"],
        marker="o",
        label="Teste",
    )
    plt.title("Overfitting - XGBoost (Accuracy)")
    plt.xlabel("max_depth")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        caminho_accuracy,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(
        dataframe_resultados["max_depth"],
        dataframe_resultados["macro_f1_treino"],
        marker="o",
        label="Treino",
    )
    plt.plot(
        dataframe_resultados["max_depth"],
        dataframe_resultados["macro_f1_teste"],
        marker="o",
        label="Teste",
    )
    plt.title("Overfitting - XGBoost (Macro F1)")
    plt.xlabel("max_depth")
    plt.ylabel("Macro F1")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        caminho_macro_f1,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    if caminho_log is not None:
        registrar_analise_overfitting_xgboost(
            caminho_log=caminho_log,
            dataframe_resultados=dataframe_resultados,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            tree_method=tree_method,
            random_state=random_state,
            n_jobs=n_jobs,
        )

    print("\nGráficos salvos com sucesso em:")
    print(f"- {caminho_accuracy}")
    print(f"- {caminho_macro_f1}")