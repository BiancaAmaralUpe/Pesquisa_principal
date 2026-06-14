# ======================================================================================
# tratativa_nome_colunas.py
# ======================================================================================
# Responsabilidade:
# - Padronizar os nomes das colunas
# - Converter nomes para snake_case
# - Remover acentos, espaços, caracteres especiais e conectivos
# ======================================================================================

import re
import unicodedata

import pandas as pd


CONECTIVOS_REMOVER = {
    "de",
    "da",
    "do",
    "das",
    "dos",
}


def remover_acentos(texto: str) -> str:
    """
    Remove acentos de uma string.
    """

    texto = unicodedata.normalize("NFKD", texto)
    texto = texto.encode("ascii", "ignore").decode("utf-8")

    return texto


def tratar_nome_coluna(nome_coluna: str) -> str:
    """
    Padroniza o nome de uma coluna para snake_case, removendo conectivos.

    Exemplos:
    'Data da denúncia - Ano' -> 'data_denuncia_ano'
    'Faixa de renda da vítima' -> 'faixa_renda_vitima'
    'Relação Vítima x Suspeito' -> 'relacao_vitima_suspeito'
    """

    nome_coluna = str(nome_coluna).strip()

    nome_coluna = remover_acentos(nome_coluna)

    nome_coluna = nome_coluna.lower()

    # Tratativas específicas antes da limpeza geral
    nome_coluna = nome_coluna.replace("\\", "_")
    nome_coluna = nome_coluna.replace("/", "_")
    nome_coluna = nome_coluna.replace("-", "_")
    nome_coluna = nome_coluna.replace(" x ", "_")

    # Remove caracteres especiais e transforma em underline
    nome_coluna = re.sub(r"[^a-z0-9_]+", "_", nome_coluna)

    # Remove underlines duplicados
    nome_coluna = re.sub(r"_+", "_", nome_coluna)

    # Remove underline no início/fim
    nome_coluna = nome_coluna.strip("_")

    # Remove conectivos
    partes = nome_coluna.split("_")
    partes = [
        parte
        for parte in partes
        if parte not in CONECTIVOS_REMOVER
    ]

    nome_coluna = "_".join(partes)

    return nome_coluna


def tratar_nomes_colunas(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica a padronização dos nomes das colunas do DataFrame.
    """

    dataframe = dataframe.copy()

    colunas_antes = list(dataframe.columns)

    dataframe.columns = [
        tratar_nome_coluna(coluna)
        for coluna in dataframe.columns
    ]

    colunas_depois = list(dataframe.columns)

    print("Nomes das colunas tratados com sucesso.")

    print("\nMapeamento das colunas:")
    for coluna_antiga, coluna_nova in zip(colunas_antes, colunas_depois):
        print(f"- {coluna_antiga} -> {coluna_nova}")

    return dataframe