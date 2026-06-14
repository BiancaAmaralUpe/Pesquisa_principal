# ======================================================================================
# transformacao_dado_xlsx_csv.py
# ======================================================================================
# Responsabilidade:
# - Ler um arquivo Excel (.xlsx)
# - Converter para CSV
# - Salvar o CSV em uma pasta de saída
# ======================================================================================

from pathlib import Path

import pandas as pd

# pyrefly: ignore [missing-import]
from Pesquisa_principal.constants import ARQUIVO_XLSX_FEMINICIDIO, ARQUIVO_CSV_FEMINICIDIO

def encontrar_linha_cabecalho(dataframe: pd.DataFrame) -> int:
    """
    Encontra a linha que contém o cabeçalho real da base.

    A base possui linhas iniciais acima do cabeçalho.
    Procuramos uma linha que contenha 'Data da denúncia - Ano'.
    """

    for indice, linha in dataframe.iterrows():
        valores_linha = linha.astype(str).str.strip().tolist()

        if "Data da denúncia - Ano" in valores_linha:
            return indice

    raise ValueError(
        "Não foi possível encontrar a linha de cabeçalho com 'Data da denúncia - Ano'."
    )


def ajustar_cabecalho_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Ajusta o cabeçalho do DataFrame usando a linha real de nomes das colunas.
    """

    indice_cabecalho = encontrar_linha_cabecalho(dataframe)

    novas_colunas = (
        dataframe.iloc[indice_cabecalho]
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )

    dataframe = dataframe.iloc[indice_cabecalho + 1:].copy()
    dataframe.columns = novas_colunas

    # Remove o nome herdado do índice da linha usada como cabeçalho.
    dataframe.columns.name = None

    dataframe = dataframe.reset_index(drop=True)

    return dataframe


def transformar_xlsx_em_csv(
    caminho_xlsx: str | Path,
    caminho_csv: str | Path,
    nome_aba: int | str = 0,
) -> pd.DataFrame:
    """
    Lê um arquivo Excel, ajusta o cabeçalho real e salva uma versão em CSV.
    """

    caminho_xlsx = Path(caminho_xlsx)
    caminho_csv = Path(caminho_csv)

    if not caminho_xlsx.exists():
        raise FileNotFoundError(f"Arquivo Excel não encontrado: {caminho_xlsx}")

    print(f"Lendo arquivo Excel: {caminho_xlsx}")

    dataframe_bruto = pd.read_excel(
        caminho_xlsx,
        sheet_name=nome_aba,
        header=None,
    )

    dataframe = ajustar_cabecalho_dataframe(dataframe_bruto)

    caminho_csv.parent.mkdir(parents=True, exist_ok=True)

    dataframe.to_csv(caminho_csv, index=False, encoding="utf-8")

    print(f"CSV salvo com sucesso em: {caminho_csv}")
    print(f"Linhas e colunas: {dataframe.shape}")

    return dataframe


def carregar_csv_existente(caminho_csv: str | Path) -> pd.DataFrame:
    """
    Carrega o CSV já existente.
    """

    caminho_csv = Path(caminho_csv)

    if not caminho_csv.exists():
        raise FileNotFoundError(f"Arquivo CSV não encontrado: {caminho_csv}")

    print(f"CSV já existe. Carregando arquivo: {caminho_csv}")

    dataframe = pd.read_csv(caminho_csv, low_memory=False)

    print(f"CSV carregado com sucesso. Linhas e colunas: {dataframe.shape}")

    return dataframe


def csv_esta_com_cabecalho_invalido(caminho_csv: str | Path) -> bool:
    """
    Verifica se o CSV parece ter sido gerado com cabeçalho errado.

    Critérios:
    - Colunas começando com 'Unnamed'
    - Primeira coluna com texto estranho herdado da linha superior do Excel
    - Ausência da coluna esperada 'Data da denúncia - Ano'
    """

    caminho_csv = Path(caminho_csv)

    if not caminho_csv.exists():
        return True

    try:
        amostra = pd.read_csv(caminho_csv, nrows=5, low_memory=False)
    except Exception as erro:
        print(f"Não foi possível validar o CSV existente: {erro}")
        return True

    colunas = [str(coluna).strip() for coluna in amostra.columns]

    tem_coluna_unnamed = any(coluna.startswith("Unnamed") for coluna in colunas)

    primeira_coluna = colunas[0] if colunas else ""

    primeira_coluna_invalida = (
        "Violência Doméstica e Familiar Contra a Mulher ouViolência Contra a Mulher"
        in primeira_coluna
    )

    coluna_principal_ausente = "Data da denúncia - Ano" not in colunas

    if tem_coluna_unnamed:
        print("CSV inválido detectado: existem colunas 'Unnamed'.")
        return True

    if primeira_coluna_invalida:
        print("CSV inválido detectado: cabeçalho parece estar deslocado.")
        return True

    if coluna_principal_ausente:
        print("CSV inválido detectado: coluna 'Data da denúncia - Ano' não encontrada.")
        return True

    return False


def csv_esta_desatualizado(caminho_xlsx: str | Path, caminho_csv: str | Path) -> bool:
    """
    Verifica se o XLSX foi modificado depois do CSV.

    Se o XLSX for mais recente que o CSV, o CSV deve ser recriado.
    """

    caminho_xlsx = Path(caminho_xlsx)
    caminho_csv = Path(caminho_csv)

    if not caminho_csv.exists():
        return True

    if not caminho_xlsx.exists():
        raise FileNotFoundError(f"Arquivo Excel não encontrado: {caminho_xlsx}")

    data_modificacao_xlsx = caminho_xlsx.stat().st_mtime
    data_modificacao_csv = caminho_csv.stat().st_mtime

    return data_modificacao_xlsx > data_modificacao_csv


def deve_reprocessar_csv(
    caminho_xlsx: str | Path,
    caminho_csv: str | Path,
    forcar_reprocessamento: bool = False,
) -> bool:
    """
    Decide se o CSV precisa ser recriado.
    """

    caminho_csv = Path(caminho_csv)

    if forcar_reprocessamento:
        print("Reprocessamento forçado ativado.")
        return True

    if not caminho_csv.exists():
        print("CSV ainda não existe. Será criado a partir do XLSX.")
        return True

    if csv_esta_desatualizado(caminho_xlsx, caminho_csv):
        print("CSV desatualizado: o XLSX é mais recente. O CSV será recriado.")
        return True

    if csv_esta_com_cabecalho_invalido(caminho_csv):
        print("CSV com estrutura inválida. O CSV será recriado.")
        return True

    return False


def transformar_base_feminicidio(forcar_reprocessamento: bool = False) -> pd.DataFrame:
    """
    Carrega a base de feminicídio.

    Regra:
    - Se o CSV não existir, cria a partir do XLSX.
    - Se o CSV estiver desatualizado, recria.
    - Se o CSV estiver com cabeçalho inválido, recria.
    - Se forcar_reprocessamento=True, recria.
    - Caso contrário, carrega o CSV existente.
    """

    precisa_reprocessar = deve_reprocessar_csv(
        caminho_xlsx=ARQUIVO_XLSX_FEMINICIDIO,
        caminho_csv=ARQUIVO_CSV_FEMINICIDIO,
        forcar_reprocessamento=forcar_reprocessamento,
    )

    if precisa_reprocessar:
        return transformar_xlsx_em_csv(
            caminho_xlsx=ARQUIVO_XLSX_FEMINICIDIO,
            caminho_csv=ARQUIVO_CSV_FEMINICIDIO,
        )

    return carregar_csv_existente(ARQUIVO_CSV_FEMINICIDIO)