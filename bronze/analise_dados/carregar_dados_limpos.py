# ======================================================================================
# carregar_dados_limpos.py
# ======================================================================================
# Responsabilidade:
# - Carregar a base limpa gerada pela camada bronze
# ======================================================================================

from pathlib import Path

import pandas as pd

def carregar_dados_limpos(caminho_csv: str | Path) -> pd.DataFrame:
    """
    Carrega a base limpa gerada pela etapa de limpeza da camada bronze.
    """

    caminho_csv = Path(caminho_csv)

    if not caminho_csv.exists():
        raise FileNotFoundError(
            f"Arquivo limpo não encontrado: {caminho_csv}. "
            "Execute primeiro a pipeline bronze."
        )

    print(f"Carregando base limpa: {caminho_csv}")

    dataframe = pd.read_csv(caminho_csv, low_memory=False)

    print(f"Base limpa carregada com sucesso. Linhas e colunas: {dataframe.shape}")

    return dataframe