# ======================================================================================
# matriz_confusao.py
# ======================================================================================
# Responsabilidade:
# - Gerar matriz de confusão absoluta
# - Gerar matriz de confusão normalizada
# - Salvar os gráficos em arquivos .png
# ======================================================================================

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix

def gerar_matriz_confusao(
    y_real: pd.Series,
    y_predito,
    caminho_saida: str,
    nome_arquivo: str,
    titulo: str,
    labels: list[str],
    normalizar: bool = False,
) -> None:
    """
    Gera e salva matriz de confusão.

    Quando normalizar=True, a matriz mostra a proporção de acertos e erros
    por classe real.
    """

    print("\n" + "=" * 80)
    print(f"GERAÇÃO DA MATRIZ DE CONFUSÃO - {titulo}")
    print("=" * 80)

    Path(caminho_saida).mkdir(parents=True, exist_ok=True)

    if normalizar:
        matriz = confusion_matrix(
            y_real,
            y_predito,
            labels=labels,
            normalize="true",
        )

        formato_valores = ".2f"
    else:
        matriz = confusion_matrix(
            y_real,
            y_predito,
            labels=labels,
        )

        formato_valores = "d"

    print("\nLabels utilizadas:")
    for label in labels:
        print(f"- {label}")

    print("\nMatriz calculada:")
    print(matriz)

    display = ConfusionMatrixDisplay(
        confusion_matrix=matriz,
        display_labels=labels,
    )

    display.plot(
        values_format=formato_valores,
        xticks_rotation=45,
    )

    plt.title(titulo)
    plt.tight_layout()

    caminho_arquivo = Path(caminho_saida) / nome_arquivo
    plt.savefig(caminho_arquivo, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nMatriz de confusão salva em: {caminho_arquivo}")

def gerar_matrizes_confusao_arvore_decisao(
    y_real,
    y_predito,
    caminho_saida: str,
) -> None:
    """
    Gera as matrizes de confusão da Árvore de Decisão:
    - absoluta
    - normalizada
    """

    Path(caminho_saida).mkdir(
        parents=True,
        exist_ok=True,
    )

    labels = [
        "sem_sinal_identificado",
        "risco_baixo",
        "risco_moderado",
        "risco_elevado",
    ]

    ordem_classes = [
        "sem_sinal_identificado",
        "risco_baixo",
        "risco_moderado",
        "risco_elevado",
    ]

    # ==================================================================================
    # Matriz de confusão absoluta
    # ==================================================================================
    matriz_absoluta = confusion_matrix(
        y_true=y_real,
        y_pred=y_predito,
        labels=ordem_classes,
    )

    display_absoluta = ConfusionMatrixDisplay(
        confusion_matrix=matriz_absoluta,
        display_labels=ordem_classes,
    )

    fig, ax = plt.subplots(
        figsize=(12, 9),
    )

    display_absoluta.plot(
        ax=ax,
        values_format="d",
        cmap="viridis",
        colorbar=True,
    )

    ax.set_title(
        "Matriz de Confusão - Árvore de Decisão",
        fontsize=18,
    )

    ax.set_xlabel(
        "Classe prevista",
        fontsize=14,
    )

    ax.set_ylabel(
        "Classe real",
        fontsize=14,
    )

    plt.xticks(
        rotation=45,
        ha="right",
    )

    plt.tight_layout()

    caminho_matriz_absoluta = (
        Path(caminho_saida) / "matriz_confusao_arvore_decisao_absoluta.png"
    )

    plt.savefig(
        caminho_matriz_absoluta,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # ==================================================================================
    # Matriz de confusão normalizada
    # ==================================================================================
    matriz_normalizada = confusion_matrix(
        y_true=y_real,
        y_pred=y_predito,
        labels=ordem_classes,
        normalize="true",
    )

    display_normalizada = ConfusionMatrixDisplay(
        confusion_matrix=matriz_normalizada,
        display_labels=ordem_classes,
    )

    fig, ax = plt.subplots(
        figsize=(12, 9),
    )

    display_normalizada.plot(
        ax=ax,
        values_format=".2f",
        cmap="viridis",
        colorbar=True,
    )

    ax.set_title(
        "Matriz de Confusão Normalizada - Árvore de Decisão",
        fontsize=18,
    )

    ax.set_xlabel(
        "Classe prevista",
        fontsize=14,
    )

    ax.set_ylabel(
        "Classe real",
        fontsize=14,
    )

    plt.xticks(
        rotation=45,
        ha="right",
    )

    plt.tight_layout()

    caminho_matriz_normalizada = (
        Path(caminho_saida) / "matriz_confusao_arvore_decisao_normalizada.png"
    )

    plt.savefig(
        caminho_matriz_normalizada,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print("\nMatrizes de confusão da Árvore de Decisão salvas com sucesso:")
    print(f"- {caminho_matriz_absoluta}")
    print(f"- {caminho_matriz_normalizada}")