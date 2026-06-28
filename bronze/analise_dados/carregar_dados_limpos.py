# ======================================================================================
# carregar_dados_limpos.py
# ======================================================================================
# Responsabilidade:
# - Carregar a base limpa gerada pela camada bronze
# - Validar existência do arquivo
# - Padronizar leitura de valores ausentes
# - Retornar um DataFrame pronto para análise
# ======================================================================================
# imports das libs 
import pandas as pd
from pathlib import Path

# imports das variaveis constants.
from Pesquisa_principal.constants import VALORES_NULOS_PADRAO

def validar_caminho_arquivo(caminho_csv: str | Path) -> Path:
    """
    Valida se o caminho informado existe e aponta para um arquivo.
    """

    caminho_csv = Path(caminho_csv)

    if not caminho_csv.exists():
        raise FileNotFoundError(
            f"Arquivo limpo não encontrado: {caminho_csv}. "
            "Execute primeiro a pipeline bronze."
        )

    if not caminho_csv.is_file():
        raise IsADirectoryError(
            f"O caminho informado não é um arquivo: {caminho_csv}"
        )

    if caminho_csv.suffix.lower() != ".csv":
        print(
            f"[AVISO] O arquivo informado não possui extensão .csv: "
            f"{caminho_csv.name}"
        )

    return caminho_csv

def ler_csv_com_fallback_encoding(
    caminho_csv: Path,
    separador: str = ",",
) -> pd.DataFrame:
    """
    Lê um arquivo CSV tentando diferentes encodings.
    """

    encodings_testados = [
        "utf-8",
        "utf-8-sig",
        "latin1",
    ]

    ultimo_erro: Exception | None = None

    for encoding in encodings_testados:
        try:
            dataframe = pd.read_csv(
                caminho_csv,
                sep=separador,
                encoding=encoding,
                low_memory=False,
                na_values=VALORES_NULOS_PADRAO,
                keep_default_na=True,
            )

            print(f"Arquivo lido com encoding: {encoding}")

            return dataframe

        except UnicodeDecodeError as erro:
            ultimo_erro = erro

    raise UnicodeDecodeError(
        "encoding",
        b"",
        0,
        1,
        f"Não foi possível ler o arquivo com os encodings testados. "
        f"Último erro: {ultimo_erro}",
    )

def validar_dataframe_carregado(dataframe: pd.DataFrame) -> None:
    """
    Valida se o DataFrame carregado possui dados.
    """

    if dataframe.empty:
        raise ValueError(
            "A base limpa foi carregada, mas está vazia."
        )

    if dataframe.shape[1] == 0:
        raise ValueError(
            "A base limpa foi carregada, mas não possui colunas."
        )

def exibir_resumo_carga(
    dataframe: pd.DataFrame,
    caminho_csv: Path,
) -> None:
    """
    Exibe um resumo simples da base carregada.
    """

    print("\n" + "=" * 80)
    print("CARGA DA BASE LIMPA")
    print("=" * 80)

    print(f"Arquivo carregado: {caminho_csv}")
    print(f"Quantidade de linhas: {dataframe.shape[0]}")
    print(f"Quantidade de colunas: {dataframe.shape[1]}")
    print(f"Total de valores nulos: {dataframe.isna().sum().sum()}")

    colunas_duplicadas = dataframe.columns[dataframe.columns.duplicated()].tolist()

    if colunas_duplicadas:
        print("\n[AVISO] Colunas duplicadas encontradas:")
        for coluna in colunas_duplicadas:
            print(f"- {coluna}")
    else:
        print("Nenhuma coluna duplicada encontrada.")

def carregar_dados_limpos(
    caminho_csv: str | Path,
    separador: str = ",",
) -> pd.DataFrame:
    """
    Carrega a base limpa gerada pela etapa de limpeza da camada bronze.
    """

    caminho_csv = validar_caminho_arquivo(caminho_csv)

    print(f"Carregando base limpa: {caminho_csv}")

    dataframe = ler_csv_com_fallback_encoding(
        caminho_csv=caminho_csv,
        separador=separador,
    )

    validar_dataframe_carregado(dataframe)

    exibir_resumo_carga(
        dataframe=dataframe,
        caminho_csv=caminho_csv,
    )

    return dataframe