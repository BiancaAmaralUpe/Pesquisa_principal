# ======================================================================================
# analise_exploratoria.py
# ======================================================================================
# Responsabilidade:
# - Gerar análises exploratórias da base limpa
# - Analisar cruzamentos entre variáveis principais
# - Apoiar interpretações antes de score/modelagem
# ======================================================================================

import pandas as pd
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 1000)
pd.set_option("display.max_colwidth", None)

def imprimir_secao(titulo: str) -> None:
    """
    Imprime uma seção formatada no terminal/output.
    """

    print("\n" + "=" * 80)
    print(titulo)
    print("=" * 80)

def gerar_tabela_cruzada(
    dataframe: pd.DataFrame,
    coluna_linha: str,
    coluna_coluna: str,
    normalizar: bool = False,
) -> None:
    """
    Gera uma tabela cruzada entre duas variáveis categóricas.
    """

    if coluna_linha not in dataframe.columns:
        print(f"[AVISO] Coluna deletada: {coluna_linha}")
        return

    if coluna_coluna not in dataframe.columns:
        print(f"[AVISO] Coluna deletada: {coluna_coluna}")
        return

    print("\n" + "-" * 80)
    print(f"Cruzamento: {coluna_linha} x {coluna_coluna}")
    print("-" * 80)

    if normalizar:
        tabela = pd.crosstab(
            dataframe[coluna_linha],
            dataframe[coluna_coluna],
            normalize="index",
            dropna=False,
        ) * 100

        print(tabela.round(2).to_string())

    else:
        tabela = pd.crosstab(
            dataframe[coluna_linha],
            dataframe[coluna_coluna],
            dropna=False,
        )

        print(tabela.to_string())

def analisar_tipo_violacao_por_denuncia_emergencial(dataframe: pd.DataFrame) -> None:
    """
    Analisa a relação entre tipo de violação e denúncia emergencial.
    """

    gerar_tabela_cruzada(
        dataframe=dataframe,
        coluna_linha="tipo_violacao",
        coluna_coluna="denuncia_emergencial",
        normalizar=True,
    )


def analisar_tipo_violacao_por_relacao_vitima_suspeito(dataframe: pd.DataFrame) -> None:
    """
    Analisa a relação entre tipo de violação e relação vítima-suspeito.
    """

    gerar_tabela_cruzada(
        dataframe=dataframe,
        coluna_linha="tipo_violacao",
        coluna_coluna="relacao_vitima_suspeito",
        normalizar=False,
    )


def analisar_tipo_violacao_por_cenario(dataframe: pd.DataFrame) -> None:
    """
    Analisa a relação entre tipo de violação e cenário da violação.
    """

    gerar_tabela_cruzada(
        dataframe=dataframe,
        coluna_linha="tipo_violacao",
        coluna_coluna="cenario_violacao",
        normalizar=False,
    )


def analisar_denuncia_emergencial_por_faixa_etaria_vitima(dataframe: pd.DataFrame) -> None:
    """
    Analisa a relação entre denúncia emergencial e faixa etária da vítima.
    """

    gerar_tabela_cruzada(
        dataframe=dataframe,
        coluna_linha="faixa_etaria_vitima",
        coluna_coluna="denuncia_emergencial",
        normalizar=True,
    )


def analisar_uf_por_tipo_violacao(dataframe: pd.DataFrame) -> None:
    """
    Analisa a relação entre UF e tipo de violação.
    """

    gerar_tabela_cruzada(
        dataframe=dataframe,
        coluna_linha="uf",
        coluna_coluna="tipo_violacao",
        normalizar=True,
    )


def analisar_faixa_etaria_suspeito_por_tipo_violacao(dataframe: pd.DataFrame) -> None:
    """
    Analisa a relação entre faixa etária do suspeito e tipo de violação.
    """

    gerar_tabela_cruzada(
        dataframe=dataframe,
        coluna_linha="faixa_etaria_suspeito",
        coluna_coluna="tipo_violacao",
        normalizar=True,
    )


def analisar_info_suspeito_nao_informada(dataframe: pd.DataFrame) -> None:
    """
    Investiga a categoria 'info_suspeito_nao_informada'
    em relação às principais variáveis.
    """

    imprimir_secao("ANÁLISE DA INFORMAÇÃO DO SUSPEITO NÃO INFORMADA")

    if "faixa_etaria_suspeito" not in dataframe.columns:
        print("[AVISO] Coluna 'faixa_etaria_suspeito' não encontrada.")
        return

    dataframe_info_nao_informada = dataframe[
        dataframe["faixa_etaria_suspeito"] == "info_suspeito_nao_informada"
    ]

    total_base = len(dataframe)
    total_info_nao_informada = len(dataframe_info_nao_informada)

    percentual = (total_info_nao_informada / total_base) * 100

    print(
        f"Total de registros com info_suspeito_nao_informada: "
        f"{total_info_nao_informada} ({percentual:.2f}%)"
    )

    print("\nDistribuição por tipo de violação:")
    print(
        dataframe_info_nao_informada["tipo_violacao"]
        .value_counts(dropna=False)
        .head(10)
    )

    print("\nDistribuição por relação vítima-suspeito:")
    print(
        dataframe_info_nao_informada["relacao_vitima_suspeito"]
        .value_counts(dropna=False)
        .head(10)
    )


def executar_analise_exploratoria(dataframe: pd.DataFrame) -> None:
    """
    Executa análises exploratórias baseadas em cruzamentos.
    """

    imprimir_secao("INÍCIO DA ANÁLISE EXPLORATÓRIA")

    analisar_tipo_violacao_por_denuncia_emergencial(dataframe)
    analisar_tipo_violacao_por_relacao_vitima_suspeito(dataframe)
    analisar_tipo_violacao_por_cenario(dataframe)
    analisar_denuncia_emergencial_por_faixa_etaria_vitima(dataframe)
    analisar_faixa_etaria_suspeito_por_tipo_violacao(dataframe)
    analisar_info_suspeito_nao_informada(dataframe)

    imprimir_secao("FIM DA ANÁLISE EXPLORATÓRIA")