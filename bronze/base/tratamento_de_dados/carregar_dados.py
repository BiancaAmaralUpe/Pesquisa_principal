# base/carregar_dados.py

from pathlib import Path

import pandas as pd

def carregar_csv(caminho_csv: str, separador: str = ",") -> pd.DataFrame:
    """
    Carrega um arquivo CSV.
    """
    caminho_csv = Path(caminho_csv)

    if not caminho_csv.exists():
        raise FileNotFoundError(f"Arquivo CSV não encontrado: {caminho_csv}")

    print(f"Lendo arquivo CSV: {caminho_csv}")

    dataframe = pd.read_csv(caminho_csv, sep=separador)

    print(f"CSV carregado com sucesso. Linhas e colunas: {dataframe.shape}")

    return dataframe

def diagnostico_inicial(dataframe: pd.DataFrame) -> None:
    """
    Exibe um diagnóstico inicial simples da base.
    """

    print("=" * 80)
    print("DIAGNÓSTICO INICIAL DA BASE")
    print("=" * 80)

    print("\nQuantidade de linhas e colunas:")
    print(dataframe.shape)

    print("\nTipos das colunas:")
    print(dataframe.dtypes)

    print("\nValores nulos por coluna:")
    print(dataframe.isna().sum())

    print("\nTotal de valores nulos:")
    print(dataframe.isna().sum().sum())

    print("\nPrimeiras linhas:")
    print(dataframe.head())