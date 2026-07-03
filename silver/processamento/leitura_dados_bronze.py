# ======================================================================================
# leitura_dados_bronze.py
# ======================================================================================
# Responsabilidade:
# - Ler a base de modelagem gerada pela camada Bronze
# - Validar se o arquivo existe
# - Validar se a coluna alvo existe
# - Exibir informações iniciais da base carregada na Silver
# ======================================================================================

from pathlib import Path

import pandas as pd

def validar_arquivo_entrada(
    caminho_entrada: Path,
) -> None:
    """
    Valida se o arquivo de entrada existe.
    """

    if not caminho_entrada.exists():
        raise FileNotFoundError(
            f"Arquivo de entrada não encontrado: {caminho_entrada}"
        )

    if caminho_entrada.suffix.lower() != ".csv":
        print(
            "[AVISO] O arquivo de entrada não possui extensão .csv: "
            f"{caminho_entrada}"
        )

def validar_coluna_alvo(
    dataframe: pd.DataFrame,
    coluna_alvo: str,
) -> None:
    """
    Valida se a coluna alvo existe na base.
    """

    if coluna_alvo not in dataframe.columns:
        raise ValueError(
            f"Coluna alvo não encontrada na base Silver: {coluna_alvo}"
        )

def ler_base_modelagem_bronze(
    caminho_entrada: Path,
    coluna_alvo: str,
) -> pd.DataFrame:
    """
    Lê a base de treino/teste gerada pela camada Bronze.
    """

    print("\n" + "=" * 80)
    print("LEITURA DA BASE GERADA PELA CAMADA BRONZE")
    print("=" * 80)

    print(f"Arquivo de entrada: {caminho_entrada}")

    validar_arquivo_entrada(caminho_entrada)

    dataframe = pd.read_csv(caminho_entrada)

    if dataframe.empty:
        raise ValueError("A base carregada está vazia.")

    validar_coluna_alvo(
        dataframe=dataframe,
        coluna_alvo=coluna_alvo,
    )

    print("\nBase carregada com sucesso.")
    print(f"Linhas: {dataframe.shape[0]}")
    print(f"Colunas: {dataframe.shape[1]}")

    print("\nColuna alvo encontrada:")
    print(coluna_alvo)

    print("\nTotal de valores nulos na base:")
    print(dataframe.isna().sum().sum())

    print("\nTipos de dados carregados:")
    print(dataframe.dtypes)

    print("\nDistribuição da coluna alvo:")
    print(dataframe[coluna_alvo].value_counts(dropna=False))

    print("\nDistribuição percentual da coluna alvo:")
    distribuicao_percentual = dataframe[coluna_alvo].value_counts(
        normalize=True,
        dropna=False,
    ) * 100

    print(distribuicao_percentual.round(2))

    return dataframe