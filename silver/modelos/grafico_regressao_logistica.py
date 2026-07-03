# ======================================================================================
# grafico_regressao_logistica.py
# ======================================================================================
# Responsabilidade:
# - Avaliar possível overfitting da Regressão Logística
# - Testar diferentes valores de C
# - Gerar gráficos de Accuracy e Macro F1-score
# - Registrar os resultados em arquivo .txt
# ======================================================================================

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


def registrar_analise_overfitting_regressao_logistica(
    caminho_log: Path,
    dataframe_resultados: pd.DataFrame,
    max_iter: int,
    solver: str,
    penalty: str,
    random_state: int,
    n_jobs: int,
) -> None:
    """
    Registra no arquivo .txt os resultados da análise de overfitting
    da Regressão Logística para diferentes valores de C.
    """

    caminho_log.parent.mkdir(parents=True, exist_ok=True)

    data_execucao = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(caminho_log, "a", encoding="utf-8") as arquivo:
        arquivo.write("\n")
        arquivo.write("=" * 80)
        arquivo.write("\n")
        arquivo.write("ANÁLISE DE OVERFITTING - REGRESSÃO LOGÍSTICA\n")
        arquivo.write("=" * 80)
        arquivo.write("\n")

        arquivo.write(f"Data/hora da execução: {data_execucao}\n\n")

        arquivo.write("Parâmetros fixos utilizados:\n")
        arquivo.write(f"- MAX_ITER_REGRESSAO_LOGISTICA: {max_iter}\n")
        arquivo.write(f"- SOLVER_REGRESSAO_LOGISTICA: {solver}\n")
        arquivo.write(f"- PENALTY_REGRESSAO_LOGISTICA: {penalty}\n")
        arquivo.write(f"- RANDOM_STATE: {random_state}\n")
        arquivo.write(f"- N_JOBS_REGRESSAO_LOGISTICA: {n_jobs}\n")
        arquivo.write("- class_weight: balanced\n\n")

        arquivo.write("Resultados por valor de C testado:\n\n")

        for _, linha in dataframe_resultados.iterrows():
            arquivo.write("-" * 80)
            arquivo.write("\n")

            arquivo.write(
                f"C_REGRESSAO_LOGISTICA: {linha['C']}\n"
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


def gerar_grafico_overfitting_regressao_logistica(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    caminho_saida: str,
    caminho_log: Path | None = None,
    max_iter: int = 1000,
    solver: str = "saga",
    penalty: str = "l2",
    random_state: int = 42,
    n_jobs: int = -1,
) -> None:
    """
    Gera gráficos para analisar possível overfitting da Regressão Logística
    comparando desempenho de treino e teste em diferentes valores de C.

    O parâmetro C controla a força da regularização:
    - C menor: regularização mais forte
    - C maior: regularização mais fraca
    """

    Path(caminho_saida).mkdir(parents=True, exist_ok=True)

    valores_c = [
        0.001,
        0.01,
        0.1,
        1.0,
        10.0,
        100.0,
    ]

    resultados = {
        "C": [],
        "accuracy_treino": [],
        "accuracy_teste": [],
        "macro_f1_treino": [],
        "macro_f1_teste": [],
    }

    print("\n" + "=" * 80)
    print("ANÁLISE DE OVERFITTING - REGRESSÃO LOGÍSTICA")
    print("=" * 80)

    for valor_c in valores_c:
        modelo = LogisticRegression(
            max_iter=max_iter,
            C=valor_c,
            solver=solver,
            penalty=penalty,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=n_jobs,
        )

        modelo.fit(X_train, y_train)

        y_pred_train = modelo.predict(X_train)
        y_pred_test = modelo.predict(X_test)

        accuracy_treino = accuracy_score(
            y_train,
            y_pred_train,
        )

        accuracy_teste = accuracy_score(
            y_test,
            y_pred_test,
        )

        macro_f1_treino = f1_score(
            y_train,
            y_pred_train,
            average="macro",
            zero_division=0,
        )

        macro_f1_teste = f1_score(
            y_test,
            y_pred_test,
            average="macro",
            zero_division=0,
        )

        resultados["C"].append(valor_c)
        resultados["accuracy_treino"].append(accuracy_treino)
        resultados["accuracy_teste"].append(accuracy_teste)
        resultados["macro_f1_treino"].append(macro_f1_treino)
        resultados["macro_f1_teste"].append(macro_f1_teste)

        print(
            f"C={valor_c} | "
            f"acc_treino={accuracy_treino:.4f} | "
            f"acc_teste={accuracy_teste:.4f} | "
            f"f1_treino={macro_f1_treino:.4f} | "
            f"f1_teste={macro_f1_teste:.4f}"
        )

    dataframe_resultados = pd.DataFrame(resultados)

    caminho_accuracy = (
        Path(caminho_saida)
        / "grafico_overfitting_regressao_logistica_accuracy.png"
    )

    caminho_macro_f1 = (
        Path(caminho_saida)
        / "grafico_overfitting_regressao_logistica_macro_f1.png"
    )

    plt.figure(figsize=(10, 6))
    plt.plot(
        dataframe_resultados["C"],
        dataframe_resultados["accuracy_treino"],
        marker="o",
        label="Treino",
    )
    plt.plot(
        dataframe_resultados["C"],
        dataframe_resultados["accuracy_teste"],
        marker="o",
        label="Teste",
    )
    plt.xscale("log")
    plt.title("Overfitting - Regressão Logística (Accuracy)")
    plt.xlabel("C")
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
        dataframe_resultados["C"],
        dataframe_resultados["macro_f1_treino"],
        marker="o",
        label="Treino",
    )
    plt.plot(
        dataframe_resultados["C"],
        dataframe_resultados["macro_f1_teste"],
        marker="o",
        label="Teste",
    )
    plt.xscale("log")
    plt.title("Overfitting - Regressão Logística (Macro F1)")
    plt.xlabel("C")
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
        registrar_analise_overfitting_regressao_logistica(
            caminho_log=caminho_log,
            dataframe_resultados=dataframe_resultados,
            max_iter=max_iter,
            solver=solver,
            penalty=penalty,
            random_state=random_state,
            n_jobs=n_jobs,
        )

    print("\nGráficos salvos com sucesso em:")
    print(f"- {caminho_accuracy}")
    print(f"- {caminho_macro_f1}")