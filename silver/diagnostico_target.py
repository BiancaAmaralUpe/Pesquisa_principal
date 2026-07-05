# ======================================================================================
# diagnostico_target_graficos.py
# ======================================================================================
# Responsabilidade:
# - Gerar gráficos de cruzamento entre colunas suspeitas e o target
# - Ajudar a identificar possível vazamento de informação
# - Verificar se o target está coerente com variáveis de violência e letalidade
# ======================================================================================

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


COLUNA_TARGET_DIAGNOSTICO = "classificacao_sinais_risco_feminicidio"


COLUNAS_DIAGNOSTICO_TARGET = [
    "especie_violacao",
    "motivacao",
    "agravantes_unificados",
    "indicador_letalidade",
    "subtipo_letalidade",
    "tipo_violencia_normalizado",
    "risco_ameaca_morte",
    "risco_violencia_fisica_grave",
    "risco_escalada_agressoes",
    "risco_controle_extremo",
    "risco_separacao_termino",
    "risco_violencia_sexual_associada",
    "qtd_sinais_risco_feminicidio",
    "componente_fisico",
    "componente_psicologico",
    "componente_sexual",
    "componente_patrimonial",
    "componente_moral",
    "componente_letal",
]


ORDEM_CLASSES_TARGET = [
    "sem_sinal_identificado",
    "risco_baixo",
    "risco_moderado",
    "risco_elevado",
]


def preparar_dataframe_diagnostico(
    dataframe: pd.DataFrame,
    coluna_analise: str,
    coluna_target: str,
) -> pd.DataFrame:
    """
    Prepara dataframe temporário para diagnóstico.
    """

    dataframe_temporario = dataframe[
        [
            coluna_analise,
            coluna_target,
        ]
    ].copy()

    dataframe_temporario[coluna_analise] = (
        dataframe_temporario[coluna_analise]
        .fillna("nao_informado")
        .astype(str)
    )

    dataframe_temporario[coluna_target] = (
        dataframe_temporario[coluna_target]
        .fillna("nao_informado")
        .astype(str)
    )

    return dataframe_temporario


def gerar_matrizes_target(
    dataframe: pd.DataFrame,
    coluna_analise: str,
    coluna_target: str,
    minimo_registros_categoria: int = 100,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Gera:
    - matriz absoluta
    - matriz percentual por linha
    - total por categoria

    Também remove categorias com poucos registros para evitar interpretação distorcida.
    """

    dataframe_temporario = preparar_dataframe_diagnostico(
        dataframe=dataframe,
        coluna_analise=coluna_analise,
        coluna_target=coluna_target,
    )

    total_por_categoria = dataframe_temporario[coluna_analise].value_counts()

    categorias_validas = total_por_categoria[
        total_por_categoria >= minimo_registros_categoria
    ].index.tolist()

    dataframe_temporario = dataframe_temporario[
        dataframe_temporario[coluna_analise].isin(categorias_validas)
    ]

    matriz_absoluta = pd.crosstab(
        dataframe_temporario[coluna_analise],
        dataframe_temporario[coluna_target],
        dropna=False,
    )

    for classe_target in ORDEM_CLASSES_TARGET:
        if classe_target not in matriz_absoluta.columns:
            matriz_absoluta[classe_target] = 0

    matriz_absoluta = matriz_absoluta[ORDEM_CLASSES_TARGET]

    matriz_percentual = pd.crosstab(
        dataframe_temporario[coluna_analise],
        dataframe_temporario[coluna_target],
        normalize="index",
        dropna=False,
    ) * 100

    for classe_target in ORDEM_CLASSES_TARGET:
        if classe_target not in matriz_percentual.columns:
            matriz_percentual[classe_target] = 0

    matriz_percentual = matriz_percentual[ORDEM_CLASSES_TARGET]

    total_por_categoria = matriz_absoluta.sum(axis=1)

    if "risco_elevado" in matriz_percentual.columns:
        coluna_ordenacao = matriz_percentual["risco_elevado"]
    else:
        coluna_ordenacao = matriz_percentual.sum(axis=1)

    ordem_categorias = coluna_ordenacao.sort_values(
        ascending=False
    ).index.tolist()

    matriz_absoluta = matriz_absoluta.loc[ordem_categorias]
    matriz_percentual = matriz_percentual.loc[ordem_categorias]
    total_por_categoria = total_por_categoria.loc[ordem_categorias]

    return matriz_absoluta, matriz_percentual.round(2), total_por_categoria


def limitar_top_categorias(
    matriz_absoluta: pd.DataFrame,
    matriz_percentual: pd.DataFrame,
    total_por_categoria: pd.Series,
    top_n_categorias: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Limita a quantidade de categorias exibidas no gráfico.
    """

    categorias_selecionadas = matriz_percentual.head(
        top_n_categorias
    ).index.tolist()

    return (
        matriz_absoluta.loc[categorias_selecionadas],
        matriz_percentual.loc[categorias_selecionadas],
        total_por_categoria.loc[categorias_selecionadas],
    )


def gerar_heatmap_percentual_com_quantidade(
    matriz_absoluta: pd.DataFrame,
    matriz_percentual: pd.DataFrame,
    total_por_categoria: pd.Series,
    coluna_analise: str,
    caminho_saida: Path,
) -> None:
    """
    Gera heatmap percentual com anotação de percentual e quantidade absoluta.
    """

    quantidade_linhas = len(matriz_percentual)

    altura_figura = max(
        6,
        quantidade_linhas * 0.55,
    )

    largura_figura = 14

    plt.figure(
        figsize=(
            largura_figura,
            altura_figura,
        )
    )

    imagem = plt.imshow(
        matriz_percentual.values,
        aspect="auto",
    )

    plt.colorbar(
        imagem,
        label="Percentual por categoria (%)",
    )

    labels_linhas = [
        f"{indice} (n={int(total_por_categoria.loc[indice])})"
        for indice in matriz_percentual.index
    ]

    plt.xticks(
        ticks=range(len(matriz_percentual.columns)),
        labels=matriz_percentual.columns,
        rotation=45,
        ha="right",
    )

    plt.yticks(
        ticks=range(len(matriz_percentual.index)),
        labels=labels_linhas,
    )

    plt.title(
        f"Distribuição do target por {coluna_analise}"
    )

    plt.xlabel("Classificação de risco")
    plt.ylabel(coluna_analise)

    for linha in range(matriz_percentual.shape[0]):
        for coluna in range(matriz_percentual.shape[1]):
            percentual = matriz_percentual.iloc[linha, coluna]
            quantidade = matriz_absoluta.iloc[linha, coluna]

            texto = f"{percentual:.1f}%\n({quantidade})"

            plt.text(
                coluna,
                linha,
                texto,
                ha="center",
                va="center",
                fontsize=8,
            )

    plt.tight_layout()

    nome_arquivo = f"heatmap_{coluna_analise}_x_target_percentual_quantidade.png"

    caminho_arquivo = caminho_saida / nome_arquivo

    plt.savefig(
        caminho_arquivo,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Gráfico salvo em: {caminho_arquivo}")


def gerar_grafico_ranking_risco_elevado(
    matriz_percentual: pd.DataFrame,
    total_por_categoria: pd.Series,
    coluna_analise: str,
    caminho_saida: Path,
) -> None:
    """
    Gera gráfico de barras ordenado pela taxa de risco_elevado.
    """

    if "risco_elevado" not in matriz_percentual.columns:
        return

    serie_risco_elevado = matriz_percentual["risco_elevado"].sort_values(
        ascending=True
    )

    labels = [
        f"{indice} (n={int(total_por_categoria.loc[indice])})"
        for indice in serie_risco_elevado.index
    ]

    altura_figura = max(
        6,
        len(serie_risco_elevado) * 0.5,
    )

    plt.figure(
        figsize=(
            12,
            altura_figura,
        )
    )

    plt.barh(
        labels,
        serie_risco_elevado.values,
    )

    plt.title(
        f"Taxa de risco elevado por {coluna_analise}"
    )

    plt.xlabel("Percentual de risco_elevado (%)")
    plt.ylabel(coluna_analise)

    for indice, valor in enumerate(serie_risco_elevado.values):
        plt.text(
            valor,
            indice,
            f" {valor:.1f}%",
            va="center",
            fontsize=8,
        )

    plt.tight_layout()

    nome_arquivo = f"ranking_risco_elevado_{coluna_analise}.png"

    caminho_arquivo = caminho_saida / nome_arquivo

    plt.savefig(
        caminho_arquivo,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Gráfico salvo em: {caminho_arquivo}")


def gerar_grafico_ranking_risco_moderado_ou_elevado(
    matriz_percentual: pd.DataFrame,
    total_por_categoria: pd.Series,
    coluna_analise: str,
    caminho_saida: Path,
) -> None:
    """
    Gera gráfico de barras ordenado pela soma de risco_moderado + risco_elevado.
    """

    colunas_necessarias = [
        "risco_moderado",
        "risco_elevado",
    ]

    for coluna in colunas_necessarias:
        if coluna not in matriz_percentual.columns:
            return

    serie_risco = (
        matriz_percentual["risco_moderado"]
        + matriz_percentual["risco_elevado"]
    ).sort_values(
        ascending=True
    )

    labels = [
        f"{indice} (n={int(total_por_categoria.loc[indice])})"
        for indice in serie_risco.index
    ]

    altura_figura = max(
        6,
        len(serie_risco) * 0.5,
    )

    plt.figure(
        figsize=(
            12,
            altura_figura,
        )
    )

    plt.barh(
        labels,
        serie_risco.values,
    )

    plt.title(
        f"Taxa de risco moderado ou elevado por {coluna_analise}"
    )

    plt.xlabel("Percentual de risco_moderado + risco_elevado (%)")
    plt.ylabel(coluna_analise)

    for indice, valor in enumerate(serie_risco.values):
        plt.text(
            valor,
            indice,
            f" {valor:.1f}%",
            va="center",
            fontsize=8,
        )

    plt.tight_layout()

    nome_arquivo = f"ranking_risco_moderado_ou_elevado_{coluna_analise}.png"

    caminho_arquivo = caminho_saida / nome_arquivo

    plt.savefig(
        caminho_arquivo,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Gráfico salvo em: {caminho_arquivo}")


def gerar_graficos_diagnostico_target(
    dataframe: pd.DataFrame,
    caminho_saida: Path,
    coluna_target: str = COLUNA_TARGET_DIAGNOSTICO,
    colunas_diagnostico: list[str] | None = None,
    top_n_categorias: int = 15,
    minimo_registros_categoria: int = 100,
) -> None:
    """
    Gera gráficos de diagnóstico entre várias colunas e o target.
    """

    if colunas_diagnostico is None:
        colunas_diagnostico = COLUNAS_DIAGNOSTICO_TARGET

    caminho_saida.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\n" + "=" * 80)
    print("GERAÇÃO DE GRÁFICOS DE DIAGNÓSTICO DO TARGET")
    print("=" * 80)

    if coluna_target not in dataframe.columns:
        raise ValueError(
            f"A coluna target '{coluna_target}' não existe no dataframe."
        )

    for coluna_analise in colunas_diagnostico:
        if coluna_analise not in dataframe.columns:
            print(f"\nColuna não encontrada na base: {coluna_analise}")
            continue

        print("\n" + "-" * 80)
        print(f"Gerando gráficos para: {coluna_analise}")
        print("-" * 80)

        matriz_absoluta, matriz_percentual, total_por_categoria = gerar_matrizes_target(
            dataframe=dataframe,
            coluna_analise=coluna_analise,
            coluna_target=coluna_target,
            minimo_registros_categoria=minimo_registros_categoria,
        )

        if matriz_percentual.empty:
            print(
                f"Nenhuma categoria com pelo menos "
                f"{minimo_registros_categoria} registros para {coluna_analise}."
            )
            continue

        matriz_absoluta, matriz_percentual, total_por_categoria = limitar_top_categorias(
            matriz_absoluta=matriz_absoluta,
            matriz_percentual=matriz_percentual,
            total_por_categoria=total_por_categoria,
            top_n_categorias=top_n_categorias,
        )

        gerar_heatmap_percentual_com_quantidade(
            matriz_absoluta=matriz_absoluta,
            matriz_percentual=matriz_percentual,
            total_por_categoria=total_por_categoria,
            coluna_analise=coluna_analise,
            caminho_saida=caminho_saida,
        )

        gerar_grafico_ranking_risco_elevado(
            matriz_percentual=matriz_percentual,
            total_por_categoria=total_por_categoria,
            coluna_analise=coluna_analise,
            caminho_saida=caminho_saida,
        )

        gerar_grafico_ranking_risco_moderado_ou_elevado(
            matriz_percentual=matriz_percentual,
            total_por_categoria=total_por_categoria,
            coluna_analise=coluna_analise,
            caminho_saida=caminho_saida,
        )

    print("\n" + "=" * 80)
    print("GRÁFICOS DE DIAGNÓSTICO DO TARGET FINALIZADOS")
    print("=" * 80)