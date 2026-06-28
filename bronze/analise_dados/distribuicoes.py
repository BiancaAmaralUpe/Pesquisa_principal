# ======================================================================================
# distribuicoes.py
# ======================================================================================
# Responsabilidade:
# - Gerar gráficos de distribuição das principais variáveis
# - Apoiar análise visual da base analítica
# - Salvar gráficos em arquivos .png
# ======================================================================================

from IPython.core import latex_symbols
from Pesquisa_principal.bronze.analise_dados import frequencia
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def garantir_pasta_saida(caminho_saida: str) -> None:
    """
    Garante que a pasta de saída exista.
    """

    Path(caminho_saida).mkdir(parents=True, exist_ok=True)


def gerar_grafico_barras(
    dataframe: pd.DataFrame,
    coluna: str,
    caminho_saida: str,
    nome_arquivo: str,
    titulo: str,
    top_n: int | None = None,
    incluir_na: bool = False,
    ordenar_indice: bool = False,
) -> None:
    """
    Gera gráfico de barras para uma coluna categórica.
    """

    if coluna not in dataframe.columns:
        print(f"[AVISO] Coluna não encontrada: {coluna}")
        return

    garantir_pasta_saida(caminho_saida)

    serie = dataframe[coluna]

    if not incluir_na:
        serie = serie.dropna()

    frequencia = serie.value_counts(dropna=incluir_na)
    if ordenar_indice:
        frequencia = frequencia.sort_index()

    if top_n is not None:
        frequencia = frequencia.head(top_n)

    if frequencia.empty:
        print(f"[AVISO] Sem dados para gerar gráfico da coluna: {coluna}")
        return

    plt.figure(figsize=(12, 6))
    ax = frequencia.plot(kind="bar")

    for indice, valor in enumerate(frequencia):
        ax.text(
            indice,
            valor,
            str(valor),
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylim(0, frequencia.max() * 1.12)

    plt.title(titulo)
    plt.xlabel(coluna)
    plt.ylabel("Quantidade")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    caminho_arquivo = Path(caminho_saida) / nome_arquivo
    plt.savefig(caminho_arquivo, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Gráfico salvo em: {caminho_arquivo}")

def gerar_grafico_histograma(
    dataframe: pd.DataFrame,
    coluna: str,
    caminho_saida: str,
    nome_arquivo: str,
    titulo: str,
    bins: int = 10,
) -> None:
    """
    Gera histograma para coluna numérica.
    """

    if coluna not in dataframe.columns:
        print(f"[AVISO] Coluna não encontrada: {coluna}")
        return

    garantir_pasta_saida(caminho_saida)

    serie = dataframe[coluna].dropna()

    if serie.empty:
        print(f"[AVISO] Sem dados para gerar histograma da coluna: {coluna}")
        return

    plt.figure(figsize=(10, 6))
    plt.hist(serie, bins=bins)
    plt.title(titulo)
    plt.xlabel(coluna)
    plt.ylabel("Frequência")
    plt.tight_layout()

    caminho_arquivo = Path(caminho_saida) / nome_arquivo
    plt.savefig(caminho_arquivo, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Gráfico salvo em: {caminho_arquivo}")

def gerar_mapa_calor_cruzamento(
    dataframe: pd.DataFrame,
    coluna_linha: str,
    coluna_coluna: str,
    caminho_saida: str,
    nome_arquivo: str,
    titulo: str,
    normalizar: bool = True,
) -> None:
    """
    Gera mapa de calor a partir do cruzamento entre duas variáveis categóricas.
    """

    if coluna_linha not in dataframe.columns:
        print(f"[AVISO] Coluna não encontrada: {coluna_linha}")
        return

    if coluna_coluna not in dataframe.columns:
        print(f"[AVISO] Coluna não encontrada: {coluna_coluna}")
        return

    garantir_pasta_saida(caminho_saida)

    if normalizar:
        tabela_cruzada = pd.crosstab(
            dataframe[coluna_linha],
            dataframe[coluna_coluna],
            normalize="index",
        ) * 100
    else:
        tabela_cruzada = pd.crosstab(
            dataframe[coluna_linha],
            dataframe[coluna_coluna],
        )

    if tabela_cruzada.empty:
        print(
            f"[AVISO] Sem dados para gerar mapa de calor: "
            f"{coluna_linha} x {coluna_coluna}"
        )
        return

    plt.figure(figsize=(12, 7))
    imagem = plt.imshow(tabela_cruzada, aspect="auto")

    plt.title(titulo)
    plt.xlabel(coluna_coluna)
    plt.ylabel(coluna_linha)

    plt.xticks(
        ticks=range(len(tabela_cruzada.columns)),
        labels=tabela_cruzada.columns,
        rotation=45,
        ha="right",
    )

    plt.yticks(
        ticks=range(len(tabela_cruzada.index)),
        labels=tabela_cruzada.index,
    )

    plt.colorbar(imagem)

    for i in range(len(tabela_cruzada.index)):
        for j in range(len(tabela_cruzada.columns)):
            valor = tabela_cruzada.iloc[i, j]

            if normalizar:
                texto = f"{valor:.1f}%"
            else:
                texto = f"{int(valor)}"

            plt.text(
                j,
                i,
                texto,
                ha="center",
                va="center",
                fontsize=8,
            )

    plt.tight_layout()

    caminho_arquivo = Path(caminho_saida) / nome_arquivo
    plt.savefig(caminho_arquivo, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Mapa de calor salvo em: {caminho_arquivo}")

def gerar_graficos_distribuicao(
    dataframe: pd.DataFrame,
    caminho_saida: str,
) -> None:
    """
    Gera o conjunto principal de gráficos de distribuição.
    """

    print("\n" + "=" * 80)
    print("GERAÇÃO DE GRÁFICOS DE DISTRIBUIÇÃO")
    print("=" * 80)

    # Target
    gerar_grafico_barras(
        dataframe=dataframe,
        coluna="classificacao_sinais_risco_feminicidio",
        caminho_saida=caminho_saida,
        nome_arquivo="distribuicao_target_classificacao_sinais_risco_feminicidio.png",
        titulo="Distribuição do Target - Classificação dos Sinais de Risco",
        top_n=None,
        incluir_na=False,
    )

    # Tipo de violência normalizado
    gerar_grafico_barras(
        dataframe=dataframe,
        coluna="tipo_violencia_normalizado",
        caminho_saida=caminho_saida,
        nome_arquivo="distribuicao_tipo_violencia_normalizado.png",
        titulo="Distribuição - Tipo de Violência Normalizado",
        top_n=None,
        incluir_na=False,
    )

    # Quantidade de sinais
    gerar_grafico_barras(
        dataframe=dataframe,
        coluna="qtd_sinais_risco_feminicidio",
        caminho_saida=caminho_saida,
        nome_arquivo="distribuicao_qtd_sinais_risco_feminicidio.png",
        titulo="Distribuição - Quantidade de Sinais de Risco de Feminicídio",
        top_n=None,
        incluir_na=False,
        ordenar_indice=True,
    )

    # Componentes de violência
    componentes = [
        "componente_fisico",
        "componente_psicologico",
        "componente_sexual",
        "componente_letal",
    ]

    for coluna in componentes:
        gerar_grafico_barras(
            dataframe=dataframe,
            coluna=coluna,
            caminho_saida=caminho_saida,
            nome_arquivo=f"distribuicao_{coluna}.png",
            titulo=f"Distribuição - {coluna}",
            top_n=None,
            incluir_na=False,
        )

    # Variáveis categóricas importantes
    colunas_top = [
        "tipo_violacao",
        "especie_violacao",
        "relacao_vitima_suspeito",
    ]

    for coluna in colunas_top:
        gerar_grafico_barras(
            dataframe=dataframe,
            coluna=coluna,
            caminho_saida=caminho_saida,
            nome_arquivo=f"top10_{coluna}.png",
            titulo=f"Top 10 categorias - {coluna}",
            top_n=10,
            incluir_na=False,
        )
        
    # Mapas de calor
    gerar_mapa_calor_cruzamento(
        dataframe=dataframe,
        coluna_linha="classificacao_sinais_risco_feminicidio",
        coluna_coluna="tipo_violencia_normalizado",
        caminho_saida=caminho_saida,
        nome_arquivo="mapa_calor_target_tipo_violencia.png",
        titulo="Mapa de Calor - Target x Tipo de Violência Normalizado",
        normalizar=True,
    )

    gerar_mapa_calor_cruzamento(
        dataframe=dataframe,
        coluna_linha="classificacao_sinais_risco_feminicidio",
        coluna_coluna="indicador_letalidade",
        caminho_saida=caminho_saida,
        nome_arquivo="mapa_calor_target_indicador_letalidade.png",
        titulo="Mapa de Calor - Target x Indicador de Letalidade",
        normalizar=True,
    )

    gerar_mapa_calor_cruzamento(
        dataframe=dataframe,
        coluna_linha="classificacao_sinais_risco_feminicidio",
        coluna_coluna="componente_letal",
        caminho_saida=caminho_saida,
        nome_arquivo="mapa_calor_target_componente_letal.png",
        titulo="Mapa de Calor - Target x Componente Letal",
        normalizar=True,
    )

    print("\nGráficos de distribuição gerados com sucesso.")