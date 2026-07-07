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

def gerar_matriz_confusao_absoluta(
    y_real,
    y_predito,
    caminho_saida: Path,
    classes_reais: list[str],
    classes_previstas: list[str],
) -> None:
    """
    Gera a matriz de confusão absoluta usando tabela cruzada.

    Eixo Y: classe real
    Eixo X: classe prevista
    """

    matriz_confusao = pd.crosstab(
        pd.Series(y_real, name="Classe real"),
        pd.Series(y_predito, name="Classe prevista"),
    )

    matriz_confusao = matriz_confusao.reindex(
        index=classes_reais,
        columns=classes_previstas,
        fill_value=0,
    )

    figura, eixo = plt.subplots(
        figsize=(12, 8),
    )

    imagem = eixo.imshow(
        matriz_confusao.values,
        aspect="auto",
    )

    figura.colorbar(
        imagem,
        ax=eixo,
    )

    eixo.set_title(
        "Matriz de Confusão - Árvore de Decisão",
        fontsize=16,
    )

    eixo.set_xlabel(
        "Classe prevista",
        fontsize=12,
    )

    eixo.set_ylabel(
        "Classe real",
        fontsize=12,
    )

    eixo.set_xticks(
        range(len(classes_previstas)),
    )

    eixo.set_xticklabels(
        classes_previstas,
        rotation=45,
        ha="right",
    )

    eixo.set_yticks(
        range(len(classes_reais)),
    )

    eixo.set_yticklabels(
        classes_reais,
    )

    for linha in range(len(classes_reais)):
        for coluna in range(len(classes_previstas)):
            valor = matriz_confusao.iloc[linha, coluna]

            eixo.text(
                coluna,
                linha,
                str(valor),
                ha="center",
                va="center",
                fontsize=10,
            )

    plt.tight_layout()

    plt.savefig(
        caminho_saida / "matriz_confusao_arvore_decisao_absoluta.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def gerar_matriz_confusao_normalizada(
    y_real,
    y_predito,
    caminho_saida: Path,
    classes_reais: list[str],
    classes_previstas: list[str],
) -> None:
    """
    Gera a matriz de confusão normalizada por linha.

    Cada linha soma aproximadamente 1.0.
    """

    matriz_confusao = pd.crosstab(
        pd.Series(y_real, name="Classe real"),
        pd.Series(y_predito, name="Classe prevista"),
    )

    matriz_confusao = matriz_confusao.reindex(
        index=classes_reais,
        columns=classes_previstas,
        fill_value=0,
    )

    matriz_normalizada = matriz_confusao.div(
        matriz_confusao.sum(axis=1),
        axis=0,
    ).fillna(0)

    figura, eixo = plt.subplots(
        figsize=(12, 8),
    )

    imagem = eixo.imshow(
        matriz_normalizada.values,
        aspect="auto",
    )

    figura.colorbar(
        imagem,
        ax=eixo,
    )

    eixo.set_title(
        "Matriz de Confusão Normalizada - Árvore de Decisão",
        fontsize=16,
    )

    eixo.set_xlabel(
        "Classe prevista",
        fontsize=12,
    )

    eixo.set_ylabel(
        "Classe real",
        fontsize=12,
    )

    eixo.set_xticks(
        range(len(classes_previstas)),
    )

    eixo.set_xticklabels(
        classes_previstas,
        rotation=45,
        ha="right",
    )

    eixo.set_yticks(
        range(len(classes_reais)),
    )

    eixo.set_yticklabels(
        classes_reais,
    )

    for linha in range(len(classes_reais)):
        for coluna in range(len(classes_previstas)):
            percentual = matriz_normalizada.iloc[linha, coluna]
            quantidade = matriz_confusao.iloc[linha, coluna]

            eixo.text(
                coluna,
                linha,
                f"{percentual:.2f}\n({quantidade})",
                ha="center",
                va="center",
                fontsize=10,
            )

    plt.tight_layout()

    plt.savefig(
        caminho_saida / "matriz_confusao_arvore_decisao_normalizada.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def gerar_matrizes_confusao_arvore_decisao(
    y_real,
    y_predito,
    caminho_saida: str,
) -> None:
    """
    Gera as matrizes de confusão absoluta e normalizada
    para o modelo de Árvore de Decisão.
    """

    caminho_saida = Path(caminho_saida)

    caminho_saida.mkdir(
        parents=True,
        exist_ok=True,
    )

    classes_reais = [
        "sem_sinal_identificado",
        "risco_baixo",
        "risco_moderado",
        "risco_elevado",
    ]

    classes_previstas = [
        "sem_sinal_identificado",
        "risco_baixo",
        "risco_moderado",
        "risco_elevado",
    ]

    gerar_matriz_confusao_absoluta(
        y_real=y_real,
        y_predito=y_predito,
        caminho_saida=caminho_saida,
        classes_reais=classes_reais,
        classes_previstas=classes_previstas,
    )

    gerar_matriz_confusao_normalizada(
        y_real=y_real,
        y_predito=y_predito,
        caminho_saida=caminho_saida,
        classes_reais=classes_reais,
        classes_previstas=classes_previstas,
    )

    print("\nMatrizes de confusão da Árvore de Decisão salvas com sucesso.")