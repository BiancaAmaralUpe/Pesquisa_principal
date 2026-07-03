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
    y_real: pd.Series,
    y_predito,
    caminho_saida: str,
) -> None:
    """
    Gera as matrizes de confusão da Árvore de Decisão:
    - absoluta
    - normalizada
    """

    labels = [
        "sem_sinal_identificado",
        "risco_baixo",
        "risco_moderado",
        "risco_elevado",
    ]

    gerar_matriz_confusao(
        y_real=y_real,
        y_predito=y_predito,
        caminho_saida=caminho_saida,
        nome_arquivo="matriz_confusao_arvore_decisao_absoluta.png",
        titulo="Matriz de Confusão - Árvore de Decisão",
        labels=labels,
        normalizar=False,
    )

    gerar_matriz_confusao(
        y_real=y_real,
        y_predito=y_predito,
        caminho_saida=caminho_saida,
        nome_arquivo="matriz_confusao_arvore_decisao_normalizada.png",
        titulo="Matriz de Confusão Normalizada - Árvore de Decisão",
        labels=labels,
        normalizar=True,
    )