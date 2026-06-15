# ======================================================================================
# analise_exploratoria.py
# ======================================================================================
# Responsabilidade:
# - Gerar análises exploratórias da base analítica
# - Analisar cruzamentos entre variáveis principais
# - Investigar relações entre violência, letalidade, agravantes e grupo vulnerável
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

    Quando normalizar=True, apresenta percentuais por linha.
    """

    if dataframe.empty:
        print("[AVISO] DataFrame vazio. Cruzamento não executado.")
        return

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
        return

    tabela = pd.crosstab(
        dataframe[coluna_linha],
        dataframe[coluna_coluna],
        dropna=False,
    )

    print(tabela.to_string())

def analisar_tipo_violacao_por_denuncia_emergencial(dataframe: pd.DataFrame) -> None:
    """
    Analisa a relação entre tipo de violação original e denúncia emergencial.
    """

    gerar_tabela_cruzada(
        dataframe=dataframe,
        coluna_linha="tipo_violacao",
        coluna_coluna="denuncia_emergencial",
        normalizar=True,
    )

def analisar_tipo_violacao_por_relacao_vitima_suspeito(dataframe: pd.DataFrame) -> None:
    """
    Analisa a relação entre tipo de violação original e relação vítima-suspeito.
    """

    gerar_tabela_cruzada(
        dataframe=dataframe,
        coluna_linha="tipo_violacao",
        coluna_coluna="relacao_vitima_suspeito",
        normalizar=False,
    )

def analisar_tipo_violacao_por_cenario(dataframe: pd.DataFrame) -> None:
    """
    Analisa a relação entre tipo de violação original e cenário da violação.
    """

    gerar_tabela_cruzada(
        dataframe=dataframe,
        coluna_linha="tipo_violacao",
        coluna_coluna="cenario_violacao",
        normalizar=False,
    )

def analisar_tipo_violencia_normalizado_por_denuncia_emergencial(
    dataframe: pd.DataFrame,
) -> None:
    """
    Analisa a relação entre tipo de violência normalizado e denúncia emergencial.
    """

    gerar_tabela_cruzada(
        dataframe=dataframe,
        coluna_linha="tipo_violencia_normalizado",
        coluna_coluna="denuncia_emergencial",
        normalizar=True,
    )

def analisar_tipo_violencia_normalizado_por_cenario(
    dataframe: pd.DataFrame,
) -> None:
    """
    Analisa a relação entre tipo de violência normalizado e cenário da violação.
    """

    gerar_tabela_cruzada(
        dataframe=dataframe,
        coluna_linha="tipo_violencia_normalizado",
        coluna_coluna="cenario_violacao",
        normalizar=True,
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

def analisar_faixa_etaria_suspeito_por_tipo_violacao(dataframe: pd.DataFrame) -> None:
    """
    Analisa a relação entre faixa etária do suspeito e tipo de violação original.
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

    if dataframe.empty:
        print("[AVISO] DataFrame vazio. Análise não executada.")
        return

    coluna_suspeito = "faixa_etaria_suspeito"
    valor_nao_informado = "info_suspeito_nao_informada"

    if coluna_suspeito not in dataframe.columns:
        print(f"[AVISO] Coluna deletada: {coluna_suspeito}")
        return

    dataframe_info_nao_informada = dataframe[
        dataframe[coluna_suspeito] == valor_nao_informado
    ]

    total_base = len(dataframe)
    total_info_nao_informada = len(dataframe_info_nao_informada)

    percentual = (total_info_nao_informada / total_base) * 100

    print(
        f"Total de registros com {valor_nao_informado}: "
        f"{total_info_nao_informada} ({percentual:.2f}%)"
    )

    if total_info_nao_informada == 0:
        print("Nenhum registro encontrado com informação do suspeito não informada.")
        return

    if "tipo_violacao" in dataframe_info_nao_informada.columns:
        print("\nDistribuição por tipo de violação:")
        print(
            dataframe_info_nao_informada["tipo_violacao"]
            .value_counts(dropna=False)
            .to_string()
        )
    else:
        print("[AVISO] Coluna deletada: tipo_violacao")

    if "tipo_violencia_normalizado" in dataframe_info_nao_informada.columns:
        print("\nDistribuição por tipo de violência normalizado:")
        print(
            dataframe_info_nao_informada["tipo_violencia_normalizado"]
            .value_counts(dropna=False)
            .to_string()
        )
    else:
        print("[AVISO] Coluna deletada: tipo_violencia_normalizado")

    if "relacao_vitima_suspeito" in dataframe_info_nao_informada.columns:
        print("\nDistribuição por relação vítima-suspeito:")
        print(
            dataframe_info_nao_informada["relacao_vitima_suspeito"]
            .value_counts(dropna=False)
            .to_string()
        )
    else:
        print("[AVISO] Coluna deletada: relacao_vitima_suspeito")

def analisar_agravantes_por_indicador_letalidade(
    dataframe: pd.DataFrame,
) -> None:
    """
    Analisa a relação entre agravantes unificados e indicador de letalidade.
    """

    gerar_tabela_cruzada(
        dataframe=dataframe,
        coluna_linha="agravantes_unificados",
        coluna_coluna="indicador_letalidade",
        normalizar=True,
    )

def analisar_agravantes_por_subtipo_letalidade(
    dataframe: pd.DataFrame,
) -> None:
    """
    Analisa a relação entre agravantes unificados e subtipo de letalidade.
    """

    gerar_tabela_cruzada(
        dataframe=dataframe,
        coluna_linha="agravantes_unificados",
        coluna_coluna="subtipo_letalidade",
        normalizar=True,
    )

def analisar_tipo_violencia_por_grupo_vulneravel(
    dataframe: pd.DataFrame,
) -> None:
    """
    Analisa a relação entre grupo vulnerável e tipo de violência normalizado.
    """

    gerar_tabela_cruzada(
        dataframe=dataframe,
        coluna_linha="grupo_vulneravel",
        coluna_coluna="tipo_violencia_normalizado",
        normalizar=True,
    )

def analisar_especies_nao_mapeadas(
    dataframe: pd.DataFrame,
) -> None:
    """
    Lista as espécies de violação classificadas como outras_violencias.
    """

    imprimir_secao("ESPÉCIES CLASSIFICADAS COMO OUTRAS VIOLÊNCIAS")

    if dataframe.empty:
        print("[AVISO] DataFrame vazio. Análise não executada.")
        return

    if "tipo_violencia_normalizado" not in dataframe.columns:
        print("[AVISO] Coluna deletada: tipo_violencia_normalizado")
        return

    if "especie_violacao" not in dataframe.columns:
        print("[AVISO] Coluna deletada: especie_violacao")
        return

    dataframe_outras = dataframe[
        dataframe["tipo_violencia_normalizado"] == "outras_violencias"
    ]

    if dataframe_outras.empty:
        print("Nenhuma espécie classificada como outras_violencias.")
        return

    frequencia = (
        dataframe_outras["especie_violacao"]
        .value_counts(dropna=False)
        .to_frame("quantidade")
    )

    frequencia["percentual (%)"] = (
        frequencia["quantidade"] / len(dataframe_outras)
    ) * 100

    print(frequencia.round(2).to_string())

def analisar_grupo_vulneravel_por_relacao_vitima_suspeito(
    dataframe: pd.DataFrame,
) -> None:
    """
    Analisa grupo vulnerável por relação vítima-suspeito.
    """

    gerar_tabela_cruzada(
        dataframe=dataframe,
        coluna_linha="grupo_vulneravel",
        coluna_coluna="relacao_vitima_suspeito",
        normalizar=True,
    )

def analisar_grupo_vulneravel_por_cenario_violacao(
    dataframe: pd.DataFrame,
) -> None:
    """
    Analisa grupo vulnerável por cenário da violação.
    """

    gerar_tabela_cruzada(
        dataframe=dataframe,
        coluna_linha="grupo_vulneravel",
        coluna_coluna="cenario_violacao",
        normalizar=True,
    )

def analisar_grupo_vulneravel_por_indicador_letalidade(
    dataframe: pd.DataFrame,
) -> None:
    """
    Analisa grupo vulnerável por indicador de letalidade.
    """

    gerar_tabela_cruzada(
        dataframe=dataframe,
        coluna_linha="grupo_vulneravel",
        coluna_coluna="indicador_letalidade",
        normalizar=True,
    )

def analisar_suspeito_faixa_etaria_informada_por_tipo_violencia(
    dataframe: pd.DataFrame,
) -> None:
    """
    Analisa se a faixa etária do suspeito foi informada por tipo de violência normalizado.
    """

    gerar_tabela_cruzada(
        dataframe=dataframe,
        coluna_linha="suspeito_faixa_etaria_informada",
        coluna_coluna="tipo_violencia_normalizado",
        normalizar=True,
    )

def executar_analise_exploratoria(dataframe: pd.DataFrame) -> None:
    """
    Executa análises exploratórias baseadas em cruzamentos.
    """

    imprimir_secao("INÍCIO DA ANÁLISE EXPLORATÓRIA")

    analisar_tipo_violacao_por_denuncia_emergencial(dataframe)
    analisar_tipo_violacao_por_relacao_vitima_suspeito(dataframe)
    analisar_tipo_violacao_por_cenario(dataframe)

    analisar_tipo_violencia_normalizado_por_denuncia_emergencial(dataframe)
    analisar_tipo_violencia_normalizado_por_cenario(dataframe)
    analisar_especies_nao_mapeadas(dataframe)

    analisar_denuncia_emergencial_por_faixa_etaria_vitima(dataframe)
    analisar_faixa_etaria_suspeito_por_tipo_violacao(dataframe)
    analisar_info_suspeito_nao_informada(dataframe)

    analisar_agravantes_por_indicador_letalidade(dataframe)
    analisar_agravantes_por_subtipo_letalidade(dataframe)

    analisar_tipo_violencia_por_grupo_vulneravel(dataframe)
    analisar_grupo_vulneravel_por_relacao_vitima_suspeito(dataframe)
    analisar_grupo_vulneravel_por_cenario_violacao(dataframe)
    analisar_grupo_vulneravel_por_indicador_letalidade(dataframe)

    analisar_suspeito_faixa_etaria_informada_por_tipo_violencia(dataframe)

    imprimir_secao("FIM DA ANÁLISE EXPLORATÓRIA")