# ======================================================================================
# preparacao_base_analitica.py
# ======================================================================================
# Responsabilidade:
# - Preparar a base analítica a partir da base limpa
# - Remover variáveis fora do recorte metodológico
# - Remover registros com ruídos analíticos
# - Remover variáveis com potencial de enviesamento para modelagem
# ======================================================================================

import pandas as pd

from Pesquisa_principal.constants import (
    COLUNA_FAIXA_ETARIA_SUSPEITO,
    COLUNAS_ENVIESAMENTO_REMOVER,
    COLUNAS_GEOGRAFICAS_TEMPORAIS_REMOVER,
    COLUNAS_RUIDO_ANALITICO,
    VALOR_INFO_SUSPEITO_NAO_INFORMADA,
)

def imprimir_secao(titulo: str) -> None:
    """
    Imprime uma seção formatada no terminal/output.
    """

    print("\n" + "=" * 80)
    print(titulo)
    print("=" * 80)


def validar_dataframe(dataframe: pd.DataFrame) -> bool:
    """
    Verifica se o DataFrame possui dados para preparação analítica.
    """

    if dataframe.empty:
        print("[AVISO] DataFrame vazio. Preparação analítica não executada.")
        return False

    if dataframe.shape[1] == 0:
        print("[AVISO] DataFrame sem colunas. Preparação analítica não executada.")
        return False

    return True


def remover_info_suspeito_nao_informada(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Remove registros em que a faixa etária do suspeito não foi informada.

    Regra aplicada:
    - Remove linhas onde faixa_etaria_suspeito == 'info_suspeito_nao_informada'
    """

    dataframe = dataframe.copy()

    imprimir_secao("REMOÇÃO DE INFORMAÇÃO DO SUSPEITO NÃO INFORMADA")

    if not validar_dataframe(dataframe):
        return dataframe

    coluna = COLUNA_FAIXA_ETARIA_SUSPEITO
    valor_ruido = VALOR_INFO_SUSPEITO_NAO_INFORMADA

    if coluna not in dataframe.columns:
        print(f"[AVISO] Coluna deletada: {coluna}")
        return dataframe

    qtd_antes = len(dataframe)

    qtd_removidos = (
        dataframe[coluna] == valor_ruido
    ).sum()

    dataframe = dataframe[
        dataframe[coluna] != valor_ruido
    ].copy()

    qtd_depois = len(dataframe)

    print(f"Coluna analisada: {coluna}")
    print(f"Valor removido: {valor_ruido}")
    print(f"Registros antes: {qtd_antes}")
    print(f"Registros removidos: {qtd_removidos}")
    print(f"Registros depois: {qtd_depois}")

    return dataframe


def remover_colunas_geograficas_temporais(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Remove variáveis geográficas e temporais que não serão usadas na modelagem.

    Justificativa:
    - O objetivo atual do estudo não é análise espacial.
    - Variáveis geográficas podem elevar risco de overfitting.
    - Variáveis temporais derivadas não fazem parte do recorte metodológico atual.
    """

    dataframe = dataframe.copy()

    imprimir_secao("REMOÇÃO DE VARIÁVEIS GEOGRÁFICAS E TEMPORAIS")

    if not validar_dataframe(dataframe):
        return dataframe

    colunas_existentes = [
        coluna for coluna in COLUNAS_GEOGRAFICAS_TEMPORAIS_REMOVER
        if coluna in dataframe.columns
    ]

    colunas_deletadas = [
        coluna for coluna in COLUNAS_GEOGRAFICAS_TEMPORAIS_REMOVER
        if coluna not in dataframe.columns
    ]

    if not colunas_existentes:
        print("Nenhuma variável geográfica/temporal encontrada para remoção.")
        return dataframe

    qtd_colunas_antes = dataframe.shape[1]

    dataframe = dataframe.drop(columns=colunas_existentes)

    print("Colunas removidas:")
    for coluna in colunas_existentes:
        print(f"- {coluna}")

    if colunas_deletadas:
        print("\nColunas já deletadas ou ausentes:")
        for coluna in colunas_deletadas:
            print(f"- {coluna}")

    print(f"\nQuantidade de colunas antes: {qtd_colunas_antes}")
    print(f"Quantidade de colunas depois: {dataframe.shape[1]}")

    return dataframe


def remover_registros_com_ruido(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Remove registros com valores ausentes em colunas essenciais.
    """

    dataframe = dataframe.copy()

    imprimir_secao("REMOÇÃO DE REGISTROS COM RUÍDO")

    if not validar_dataframe(dataframe):
        return dataframe

    qtd_antes = len(dataframe)

    for coluna in COLUNAS_RUIDO_ANALITICO:
        if coluna not in dataframe.columns:
            print(f"- {coluna}: coluna deletada.")
            continue

        qtd_removidos = dataframe[coluna].isna().sum()

        dataframe = dataframe.dropna(subset=[coluna]).copy()

        print(f"- {coluna}: {qtd_removidos} registros removidos por <NA>")

    qtd_depois = len(dataframe)

    print(f"\nRegistros antes: {qtd_antes}")
    print(f"Registros depois: {qtd_depois}")
    print(f"Registros removidos: {qtd_antes - qtd_depois}")

    return dataframe


def remover_variaveis_enviesamento(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Remove variáveis sensíveis com potencial de enviesamento na modelagem.

    Observação:
    - A remoção é aplicada na base analítica/modelagem.
    - A camada bronze tratada deve preservar os dados rastreáveis.
    """

    dataframe = dataframe.copy()

    imprimir_secao("REMOÇÃO DE VARIÁVEIS COM POTENCIAL DE ENVIESAMENTO")

    if not validar_dataframe(dataframe):
        return dataframe

    colunas_existentes = [
        coluna for coluna in COLUNAS_ENVIESAMENTO_REMOVER
        if coluna in dataframe.columns
    ]

    colunas_deletadas = [
        coluna for coluna in COLUNAS_ENVIESAMENTO_REMOVER
        if coluna not in dataframe.columns
    ]

    if not colunas_existentes:
        print("Nenhuma variável sensível encontrada para remoção.")
        return dataframe

    qtd_colunas_antes = dataframe.shape[1]

    dataframe = dataframe.drop(columns=colunas_existentes)

    print("Colunas removidas:")
    for coluna in colunas_existentes:
        print(f"- {coluna}")

    if colunas_deletadas:
        print("\nColunas já deletadas ou ausentes:")
        for coluna in colunas_deletadas:
            print(f"- {coluna}")

    print(f"\nQuantidade de colunas antes: {qtd_colunas_antes}")
    print(f"Quantidade de colunas depois: {dataframe.shape[1]}")

    return dataframe


def preparar_base_analitica(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Executa a preparação da base analítica.
    """

    imprimir_secao("INÍCIO DA PREPARAÇÃO DA BASE ANALÍTICA")

    dataframe_analise = dataframe.copy()

    dataframe_analise = remover_colunas_geograficas_temporais(dataframe_analise)
    dataframe_analise = remover_variaveis_enviesamento(dataframe_analise)
    dataframe_analise = remover_registros_com_ruido(dataframe_analise)

    imprimir_secao("FIM DA PREPARAÇÃO DA BASE ANALÍTICA")

    print(f"Linhas finais: {dataframe_analise.shape[0]}")
    print(f"Colunas finais: {dataframe_analise.shape[1]}")

    return dataframe_analise