# ======================================================================================
# grafico_regressao_logistica.py
# ======================================================================================
# Responsabilidade:
# - Avaliar possível overfitting da Regressão Logística via SGDClassifier
# - Testar diferentes valores de C
# - Converter C em alpha para o SGDClassifier
# - Gerar gráficos de Accuracy e Macro F1-score
# - Registrar os resultados em arquivo .txt
# ======================================================================================

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


def calcular_alpha_sgd(
    C: float,
    quantidade_linhas_treino: int,
) -> float:
    """
    Converte o parâmetro C em alpha para o SGDClassifier.

    No LogisticRegression tradicional, C controla a regularização de forma inversa.
    No SGDClassifier, alpha controla diretamente a regularização.
    """

    if C <= 0:
        return 0.0001

    alpha = 1 / (C * quantidade_linhas_treino)

    return alpha


def registrar_analise_overfitting_regressao_logistica(
    caminho_log: Path,
    dataframe_resultados: pd.DataFrame,
    max_iter: int,
    solver: str,
    l1_ratio: float,
    random_state: int,
) -> None:
    """
    Registra no arquivo .txt os resultados da análise de overfitting
    da Regressão Logística via SGDClassifier.
    """

    caminho_log.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data_execucao = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(caminho_log, "a", encoding="utf-8") as arquivo:
        arquivo.write("\n")
        arquivo.write("=" * 80)
        arquivo.write("\n")
        arquivo.write("ANÁLISE DE OVERFITTING - REGRESSÃO LOGÍSTICA VIA SGD\n")
        arquivo.write("=" * 80)
        arquivo.write("\n")

        arquivo.write(f"Data/hora da execução: {data_execucao}\n\n")

        arquivo.write("Parâmetros fixos utilizados:\n")
        arquivo.write(f"- MAX_ITER_REGRESSAO_LOGISTICA: {max_iter}\n")
        arquivo.write(f"- SOLVER_REGRESSAO_LOGISTICA: {solver}\n")
        arquivo.write("- loss: log_loss\n")
        arquivo.write("- penalty: elasticnet\n")
        arquivo.write(f"- L1_RATIO_REGRESSAO_LOGISTICA: {l1_ratio}\n")
        arquivo.write(f"- RANDOM_STATE: {random_state}\n")
        arquivo.write("- class_weight: balanced\n")
        arquivo.write("- n_jobs: -1\n\n")

        arquivo.write("Resultados por valor de C testado:\n\n")

        for _, linha in dataframe_resultados.iterrows():
            arquivo.write("-" * 80)
            arquivo.write("\n")

            arquivo.write(f"C_REGRESSAO_LOGISTICA: {linha['C']}\n")
            arquivo.write(f"ALPHA_CALCULADO_SGD: {linha['alpha']}\n")

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
    max_iter: int = 100,
    solver: str = "sgd",
    l1_ratio: float = 0.0,
    random_state: int = 42,
) -> None:
    """
    Gera gráficos para analisar possível overfitting da Regressão Logística
    usando SGDClassifier.

    A análise testa diferentes valores de C e converte cada valor em alpha.
    """

    Path(caminho_saida).mkdir(
        parents=True,
        exist_ok=True,
    )

    valores_c = [
        0.01,
        0.1,
        1.0,
        10.0,
    ]

    resultados = {
        "C": [],
        "alpha": [],
        "accuracy_treino": [],
        "accuracy_teste": [],
        "macro_f1_treino": [],
        "macro_f1_teste": [],
    }

    print("\n" + "=" * 80)
    print("ANÁLISE DE OVERFITTING - REGRESSÃO LOGÍSTICA VIA SGD")
    print("=" * 80)

    for valor_c in valores_c:
        alpha = calcular_alpha_sgd(
            C=valor_c,
            quantidade_linhas_treino=X_train.shape[0],
        )

        modelo = SGDClassifier(
            loss="log_loss",
            penalty="elasticnet",
            alpha=alpha,
            l1_ratio=l1_ratio,
            max_iter=max_iter,
            tol=1e-3,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )

        modelo.fit(
            X_train,
            y_train,
        )

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
        resultados["alpha"].append(alpha)
        resultados["accuracy_treino"].append(accuracy_treino)
        resultados["accuracy_teste"].append(accuracy_teste)
        resultados["macro_f1_treino"].append(macro_f1_treino)
        resultados["macro_f1_teste"].append(macro_f1_teste)

        print(
            f"C={valor_c} | "
            f"alpha={alpha:.8f} | "
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
    plt.title("Overfitting - Regressão Logística via SGD (Accuracy)")
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
    plt.title("Overfitting - Regressão Logística via SGD (Macro F1)")
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
            l1_ratio=l1_ratio,
            random_state=random_state,
        )

    print("\nGráficos salvos com sucesso em:")
    print(f"- {caminho_accuracy}")
    print(f"- {caminho_macro_f1}")