# ======================================================================================
# matriz_confusao_mlp_modelo.py
# ======================================================================================
# Responsabilidade:
# - Gerar matriz de confusão absoluta e normalizada para o modelo MLP
# ======================================================================================

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix


def gerar_matriz_confusao_mlp(
    y_real,
    y_predito,
    classes_target: list[str],
    caminho_saida: str,
    nome_arquivo: str,
    titulo: str,
    normalizar: str | None = None,
) -> None:
    """
    Gera uma matriz de confusão do modelo MLP.

    Os valores reais e preditos são numéricos.
    Os nomes das classes são utilizados somente para exibição.
    """

    pasta_saida = Path(caminho_saida)

    pasta_saida.mkdir(
        parents=True,
        exist_ok=True,
    )

    labels_numericos = list(
        range(len(classes_target))
    )

    print("\n" + "=" * 80)
    print(f"GERAÇÃO DA MATRIZ DE CONFUSÃO - {titulo}")
    print("=" * 80)

    print("\nMapeamento utilizado:")

    for codigo, classe in enumerate(classes_target):
        print(f"- {codigo}: {classe}")

    matriz = confusion_matrix(
        y_true=y_real,
        y_pred=y_predito,
        labels=labels_numericos,
        normalize=normalizar,
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matriz,
        display_labels=classes_target,
    )

    display.plot(
        values_format=".2f" if normalizar else "d",
    )

    plt.title(titulo)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    caminho_arquivo = pasta_saida / nome_arquivo

    plt.savefig(
        caminho_arquivo,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(f"\nMatriz salva em: {caminho_arquivo}")

def gerar_matrizes_confusao_mlp(
    y_real,
    y_predito,
    classes_target: list[str],
    caminho_saida: str,
) -> None:
    """
    Gera as matrizes de confusão absoluta e normalizada do MLP.
    """

    gerar_matriz_confusao_mlp(
        y_real=y_real,
        y_predito=y_predito,
        classes_target=classes_target,
        caminho_saida=caminho_saida,
        nome_arquivo="matriz_confusao_mlp_absoluta.png",
        titulo="Matriz de Confusão - MLP",
        normalizar=None,
    )

    gerar_matriz_confusao_mlp(
        y_real=y_real,
        y_predito=y_predito,
        classes_target=classes_target,
        caminho_saida=caminho_saida,
        nome_arquivo="matriz_confusao_mlp_normalizada.png",
        titulo="Matriz de Confusão Normalizada - MLP",
        normalizar="true",
    )