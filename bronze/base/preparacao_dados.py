# ======================================================================================
# preparacao_dados.py
# ======================================================================================
# Responsabilidade:
# - Preparar a base analítica para etapas futuras de modelagem
# - Separar variável alvo e variáveis explicativas
# - Identificar colunas categóricas
# - Validar estrutura final antes de encoding/normalização
# ======================================================================================

import pandas as pd

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

    print("\n" + "=" * 80)
    print("SEPARAÇÃO DA VARIÁVEL ALVO")
    print("=" * 80)

    print(f"Coluna alvo: {coluna_alvo}")
    print(f"Formato de X: {X.shape}")
    print(f"Formato de y: {y.shape}")

    return X, y


def identificar_colunas_categoricas(
    dataframe: pd.DataFrame,
) -> list[str]:
    """
    Identifica colunas categóricas da base.
    """

    colunas_categoricas = dataframe.select_dtypes(
        include=["object", "string"]
    ).columns.tolist()

    print("\n" + "=" * 80)
    print("COLUNAS CATEGÓRICAS IDENTIFICADAS")
    print("=" * 80)

    for coluna in colunas_categoricas:
        print(f"- {coluna}")

    return colunas_categoricas


def validar_base_pre_modelagem(
    dataframe: pd.DataFrame,
) -> None:
    """
    Valida a estrutura da base antes da modelagem.
    """

    print("\n" + "=" * 80)
    print("VALIDAÇÃO DA BASE PRÉ-MODELAGEM")
    print("=" * 80)

    print(f"Linhas: {dataframe.shape[0]}")
    print(f"Colunas: {dataframe.shape[1]}")
    print(f"Total de valores nulos: {dataframe.isna().sum().sum()}")

    print("\nTipos de dados:")
    print(dataframe.dtypes)

def preencher_nulos_com_marcadores(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Preenche valores nulos com marcadores informativos para modelagem.
    """

    dataframe = dataframe.copy()

    marcadores = {
        "motivacao": "motivacao_nao_informada",
        "agravantes": "agravantes_nao_informado",
        "agravantes_policiais": "agravantes_policiais_nao_informado",
        "grau_instrucao_vitima": "instrucao_vitima_nao_informada",
        "faixa_renda_vitima": "renda_vitima_nao_informada",
        "pais_origem_vitima": "pais_origem_vitima_nao_informado",
        "nacionalidade_vitima": "nacionalidade_vitima_nao_informada",
        "deficiencia_vitima": "deficiencia_vitima_nao_informada",
        "grau_instrucao_suspeito": "instrucao_suspeito_nao_informada",
        "faixa_renda_suspeito": "renda_suspeito_nao_informada",
        "pais_origem_suspeito": "pais_origem_suspeito_nao_informado",
        "nacionalidade_suspeito": "nacionalidade_suspeito_nao_informada",
        "deficiencia_suspeito": "deficiencia_suspeito_nao_informada",
    }

    print("\n" + "=" * 80)
    print("PREENCHIMENTO DE NULOS COM MARCADORES")
    print("=" * 80)

    for coluna, marcador in marcadores.items():
        if coluna not in dataframe.columns:
            print(f"- {coluna}: coluna não encontrada.")
            continue

        qtd_nulos = dataframe[coluna].isna().sum()

        dataframe[coluna] = dataframe[coluna].fillna(marcador)

        print(f"- {coluna}: {qtd_nulos} nulos preenchidos com '{marcador}'")

    print(f"\nTotal de nulos após preenchimento: {dataframe.isna().sum().sum()}")

    return dataframe

def preparar_dados_modelagem(
    dataframe: pd.DataFrame,
    coluna_alvo: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Executa a preparação inicial dos dados para modelagem.
    """
    validar_base_pre_modelagem(dataframe)
    dataframe = preencher_nulos_com_marcadores(dataframe)
    identificar_colunas_categoricas(dataframe)

    X, y = separar_variavel_alvo(
        dataframe=dataframe,
        coluna_alvo=coluna_alvo,
    )

    return X, y

