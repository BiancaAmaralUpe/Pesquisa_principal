# ======================================================================================
# preparacao_dados.py
# ======================================================================================
# Responsabilidade:
# - Preparar a base analítica para etapas futuras de modelagem
# - Separar variável alvo e variáveis explicativas
# - Remover colunas que causam vazamento de informação
# - Preencher nulos com marcadores informativos
# - Identificar colunas categóricas
# - Validar estrutura final antes de encoding/normalização
# ======================================================================================

import pandas as pd

from Pesquisa_principal.bronze.analise_dados.definir_metodologia import validar_dataframe
from Pesquisa_principal.bronze.analise_dados.analise_completude import imprimir_secao
from Pesquisa_principal.constants import COLUNAS_REMOVER_BASE_TREINO, MARCADORES_NULOS_MODELAGEM

def validar_coluna_alvo(
    dataframe: pd.DataFrame,
    coluna_alvo: str,
) -> None:
    """
    Valida se a coluna alvo existe no DataFrame.
    """

    imprimir_secao("VALIDAÇÃO DA COLUNA ALVO")

    if coluna_alvo not in dataframe.columns:
        raise ValueError(f"Coluna alvo não encontrada: {coluna_alvo}")

    print(f"Coluna alvo encontrada: {coluna_alvo}")

def validar_base_pre_modelagem(
    dataframe: pd.DataFrame,
) -> None:
    """
    Valida a estrutura da base antes da modelagem.
    """

    imprimir_secao("VALIDAÇÃO DA BASE PRÉ-MODELAGEM")

    print(f"Linhas: {dataframe.shape[0]}")
    print(f"Colunas: {dataframe.shape[1]}")
    print(f"Total de valores nulos: {dataframe.isna().sum().sum()}")

    print("\nTipos de dados:")
    print(dataframe.dtypes)

def remover_colunas_base_treino(
    dataframe: pd.DataFrame,
    coluna_alvo: str,
) -> pd.DataFrame:
    """
    Remove colunas que não devem entrar como variáveis explicativas na base de treino.
    """

    dataframe = dataframe.copy()

    imprimir_secao("REMOÇÃO DE COLUNAS DA BASE DE TREINO")

    colunas_para_remover = [
        coluna
        for coluna in COLUNAS_REMOVER_BASE_TREINO
        if coluna in dataframe.columns and coluna != coluna_alvo
    ]

    if not colunas_para_remover:
        print("Nenhuma coluna removida da base de treino.")
        return dataframe

    dataframe = dataframe.drop(columns=colunas_para_remover)

    print("Colunas removidas do X para evitar vazamento ou redundância:")
    for coluna in colunas_para_remover:
        print(f"- {coluna}")

    print(f"\nQuantidade de colunas após remoção: {dataframe.shape[1]}")

    return dataframe

def preencher_nulos_com_marcadores(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Preenche valores nulos com marcadores informativos para modelagem.
    """

    dataframe = dataframe.copy()

    imprimir_secao("PREENCHIMENTO DE NULOS COM MARCADORES")

    for coluna, marcador in MARCADORES_NULOS_MODELAGEM.items():
        if coluna not in dataframe.columns:
            print(f"- {coluna}: Coluna deletada.")
            continue

        qtd_nulos = dataframe[coluna].isna().sum()

        dataframe[coluna] = dataframe[coluna].fillna(marcador)

        print(f"- {coluna}: {qtd_nulos} nulos preenchidos com '{marcador}'")

    print(f"\nTotal de nulos após preenchimento: {dataframe.isna().sum().sum()}")

    return dataframe

def identificar_colunas_categoricas(
    dataframe: pd.DataFrame,
) -> list[str]:
    """
    Identifica colunas categóricas da base.
    """

    colunas_categoricas = dataframe.select_dtypes(
        include=["object", "string"]
    ).columns.tolist()

    imprimir_secao("COLUNAS CATEGÓRICAS IDENTIFICADAS")

    for coluna in colunas_categoricas:
        print(f"- {coluna}")

    return colunas_categoricas

def separar_variavel_alvo(
    dataframe: pd.DataFrame,
    coluna_alvo: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separa a base em variáveis explicativas X e variável alvo y.
    """

    if coluna_alvo not in dataframe.columns:
        raise ValueError(f"Coluna alvo não encontrada: {coluna_alvo}")

    X = dataframe.drop(columns=[coluna_alvo])
    y = dataframe[coluna_alvo]

    imprimir_secao("SEPARAÇÃO DA VARIÁVEL ALVO")

    print(f"Coluna alvo: {coluna_alvo}")
    print(f"Formato de X: {X.shape}")
    print(f"Formato de y: {y.shape}")

    return X, y

def preparar_dados_modelagem(
    dataframe: pd.DataFrame,
    coluna_alvo: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Executa a preparação inicial dos dados para modelagem.
    """

    imprimir_secao("INÍCIO DA PREPARAÇÃO DOS DADOS PARA MODELAGEM")

    if not validar_dataframe(dataframe):
        return dataframe.copy(), pd.Series(dtype="object")

    validar_coluna_alvo(
        dataframe=dataframe,
        coluna_alvo=coluna_alvo,
    )

    validar_base_pre_modelagem(dataframe)

    dataframe_modelagem = remover_colunas_base_treino(
        dataframe=dataframe,
        coluna_alvo=coluna_alvo,
    )

    dataframe_modelagem = preencher_nulos_com_marcadores(dataframe_modelagem)

    identificar_colunas_categoricas(dataframe_modelagem)

    X, y = separar_variavel_alvo(
        dataframe=dataframe_modelagem,
        coluna_alvo=coluna_alvo,
    )

    imprimir_secao("FIM DA PREPARAÇÃO DOS DADOS PARA MODELAGEM")

    return X, y