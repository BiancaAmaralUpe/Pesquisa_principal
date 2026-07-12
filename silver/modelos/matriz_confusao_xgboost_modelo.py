# ======================================================================================
# matriz_confusao_xgboost_modelo.py
# ======================================================================================
# Responsabilidade:
# - Gerar matriz de confusão absoluta do XGBoost
# - Gerar matriz de confusão normalizada do XGBoost
# - Salvar os gráficos em arquivos .png
# ======================================================================================

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix


def gerar_matriz_confusao_xgboost(
    y_real: pd.Series,
    y_predito,
    caminho_saida: str,
    nome_arquivo: str,
    titulo: str,
    classes_target: list[str],
    normalizar: bool = False,
) -> None:
    """
    Gera e salva a matriz de confusão do XGBoost.

    A matriz é calculada com labels numéricos, mas exibe
    os nomes textuais das classes.
    """

    print("\n" + "=" * 80)
    print(f"GERAÇÃO DA MATRIZ DE CONFUSÃO - {titulo}")
    print("=" * 80)

    caminho_saida_path = Path(caminho_saida)

    caminho_saida_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    y_real_numpy = np.asarray(
        y_real,
        dtype=np.int64,
    ).reshape(-1)

    y_predito_numpy = np.asarray(
        y_predito,
        dtype=np.int64,
    ).reshape(-1)

    if len(y_real_numpy) != len(y_predito_numpy):
        raise ValueError(
            "y_real e y_predito possuem quantidades "
            "diferentes de registros."
        )

    labels_numericos = list(
        range(len(classes_target))
    )

    print("\nMapeamento utilizado:")

    for codigo, nome_classe in enumerate(classes_target):
        print(
            f"- {codigo}: {nome_classe}"
        )

    matriz = confusion_matrix(
        y_true=y_real_numpy,
        y_pred=y_predito_numpy,
        labels=labels_numericos,
        normalize="true" if normalizar else None,
    )

    formato_valores = ".2f" if normalizar else "d"

    print("\nMatriz calculada:")
    print(matriz)

    figura, eixo = plt.subplots(
        figsize=(10, 8),
    )

    visualizacao = ConfusionMatrixDisplay(
        confusion_matrix=matriz,
        display_labels=classes_target,
    )

    visualizacao.plot(
        ax=eixo,
        values_format=formato_valores,
        xticks_rotation=45,
        colorbar=False,
    )

    eixo.set_title(
        titulo
    )

    figura.tight_layout()

    caminho_arquivo = (
        caminho_saida_path
        / nome_arquivo
    )

    figura.savefig(
        caminho_arquivo,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figura
    )

    print(
        f"\nMatriz de confusão salva em: "
        f"{caminho_arquivo}"
    )


def gerar_matrizes_confusao_xgboost(
    y_real: pd.Series,
    y_predito,
    classes_target: list[str],
    caminho_saida: str,
) -> None:
    """
    Gera as matrizes de confusão do XGBoost:
    - absoluta;
    - normalizada.
    """

    gerar_matriz_confusao_xgboost(
        y_real=y_real,
        y_predito=y_predito,
        caminho_saida=caminho_saida,
        nome_arquivo="matriz_confusao_xgboost_absoluta.png",
        titulo="Matriz de Confusão - XGBoost",
        classes_target=classes_target,
        normalizar=False,
    )

    gerar_matriz_confusao_xgboost(
        y_real=y_real,
        y_predito=y_predito,
        caminho_saida=caminho_saida,
        nome_arquivo="matriz_confusao_xgboost_normalizada.png",
        titulo="Matriz de Confusão Normalizada - XGBoost",
        classes_target=classes_target,
        normalizar=True,
    )