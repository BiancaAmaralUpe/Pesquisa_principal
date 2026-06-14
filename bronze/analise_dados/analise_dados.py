# ======================================================================================
# analise_dados.py
# ======================================================================================
# Responsabilidade:
# - Remover ruídos analíticos identificados para preparação de treino/modelagem
# ======================================================================================

import pandas as pd

def remover_info_suspeito_nao_informada(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Remove registros em que a faixa etária do suspeito não foi informada.

    Regra aplicada:
    - Remove linhas onde faixa_etaria_suspeito == 'info_suspeito_nao_informada'
    """

    dataframe = dataframe.copy()

    qtd_antes = len(dataframe)

    coluna = "faixa_etaria_suspeito"
    valor_ruido = "info_suspeito_nao_informada"

    if coluna not in dataframe.columns:
        print(f"Coluna não encontrada: {coluna}")
        return dataframe

    qtd_removidos = (
        dataframe[coluna] == valor_ruido
    ).sum()

    dataframe = dataframe[
        dataframe[coluna] != valor_ruido
    ].copy()

    qtd_depois = len(dataframe)

    print("\n" + "=" * 80)
    print("REMOÇÃO DE RUÍDO ANALÍTICO")
    print("=" * 80)

    print(f"Coluna analisada: {coluna}")
    print(f"Valor removido: {valor_ruido}")
    print(f"Registros antes: {qtd_antes}")
    print(f"Registros removidos: {qtd_removidos}")
    print(f"Registros depois: {qtd_depois}")

    return dataframe
import pandas as pd


def remover_colunas_geograficas(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Remove variáveis geográficas que não serão usadas na modelagem.

    Justificativa:
    - O objetivo do estudo não é análise por estado, município, cidade ou bairro.
    - Essas variáveis podem elevar risco de overfitting.
    - Também podem introduzir enviesamento regional no modelo.
    """

    dataframe = dataframe.copy()

    colunas_remover = [
    "uf",
    "municipio",
    "estado",
    "cidade",
    "bairro",
    "data_denuncia_ano",
    "data_denuncia_mes",
    "data_denuncia_dia",
    "canal_atendimento",
]
    colunas_existentes = [
        coluna for coluna in colunas_remover
        if coluna in dataframe.columns
    ]

    print("\n" + "=" * 80)
    print("REMOÇÃO DE VARIÁVEIS GEOGRÁFICAS")
    print("=" * 80)

    if not colunas_existentes:
        print("Nenhuma variável geográfica encontrada para remoção.")
        return dataframe

    dataframe = dataframe.drop(columns=colunas_existentes)

    print("Colunas removidas:")
    for coluna in colunas_existentes:
        print(f"- {coluna}")

    print(f"Quantidade de colunas após remoção: {dataframe.shape[1]}")

    return dataframe

def remover_registros_com_ruido(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Remove registros com valores ausentes em colunas essenciais.
    """

    dataframe = dataframe.copy()

    colunas_com_ruido = [
        "cenario_violacao",
        "sexo_vitima",
        "sexo_suspeito",
        "relacao_vitima_suspeito",
    ]

    print("\n" + "=" * 80)
    print("REMOÇÃO DE REGISTROS COM RUÍDO")
    print("=" * 80)

    qtd_antes = len(dataframe)

    for coluna in colunas_com_ruido:
        if coluna not in dataframe.columns:
            print(f"- {coluna}: coluna deletada.")
            continue

        qtd_removidos = dataframe[coluna].isna().sum()
        dataframe = dataframe.dropna(subset=[coluna])

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
    Remove variáveis sensíveis com potencial de enviesamento.
    """

    dataframe = dataframe.copy()

    colunas_remover = [
        "raca_cor_suspeito",
        "raca_cor_vitima",
    ]

    colunas_existentes = [
        coluna for coluna in colunas_remover
        if coluna in dataframe.columns
    ]

    print("\n" + "=" * 80)
    print("REMOÇÃO DE VARIÁVEIS COM POTENCIAL DE ENVIESAMENTO")
    print("=" * 80)

    if not colunas_existentes:
        print("Nenhuma variável sensível encontrada para remoção.")
        return dataframe

    dataframe = dataframe.drop(columns=colunas_existentes)

    print("Colunas removidas:")
    for coluna in colunas_existentes:
        print(f"- {coluna}")

    print(f"Quantidade de colunas após remoção: {dataframe.shape[1]}")

    return dataframe
    