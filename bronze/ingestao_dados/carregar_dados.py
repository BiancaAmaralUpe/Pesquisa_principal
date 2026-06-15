# ======================================================================================
# carregar_dados.py
# ======================================================================================
# Responsabilidade:
# - Carregar arquivos CSV
# - Validar existência do arquivo
# - Retornar DataFrame para a pipeline
# ======================================================================================

from pathlib import Path

import pandas as pd

from Pesquisa_principal.constants import VALORES_NULOS_PADRAO


def carregar_csv(
    caminho_csv: str | Path,
    separador: str = ",",
) -> pd.DataFrame:
    """
    Carrega um arquivo CSV.
    """

    caminho_csv = Path(caminho_csv)

    if not caminho_csv.exists():
        raise FileNotFoundError(f"Arquivo CSV não encontrado: {caminho_csv}")

    if not caminho_csv.is_file():
        raise IsADirectoryError(f"O caminho informado não é um arquivo: {caminho_csv}")

    print(f"Lendo arquivo CSV: {caminho_csv}")

    dataframe = pd.read_csv(
        caminho_csv,
        sep=separador,
        low_memory=False,
        na_values=VALORES_NULOS_PADRAO,
        keep_default_na=True,
    )

    print(f"CSV carregado com sucesso. Linhas e colunas: {dataframe.shape}")

    return dataframe