# ======================================================================================
# limpeza_dados.py
# ======================================================================================
# Responsabilidade:
# - Remover linhas totalmente vazias
# - Padronizar nomes das colunas
# - Padronizar textos
# - Tratar valores textuais equivalentes a nulo
# - Gerar relatório simples da limpeza
# ======================================================================================

import pandas as pd
from pathlib import Path
# pyrefly: ignore [missing-import]
from Pesquisa_principal.bronze.base.tratamento_de_dados.tratativa_nome_colunas import tratar_nomes_colunas

def validar_cabecalho(dataframe: pd.DataFrame) -> None:
    """
    Valida se o cabeçalho da base está correto.

    Se existirem colunas 'Unnamed', a base provavelmente foi carregada
    com o cabeçalho errado. Nesse caso, a correção deve acontecer na
    transformação XLSX -> CSV, e não removendo as colunas na limpeza.
    """

    colunas_unnamed = [
        coluna for coluna in dataframe.columns
        if str(coluna).startswith("Unnamed")
    ]

    if colunas_unnamed:
        raise ValueError(
            "Cabeçalho inválido detectado. Existem colunas 'Unnamed'. "
            "Reprocesse o CSV a partir do XLSX antes da limpeza. "
            f"Colunas problemáticas: {colunas_unnamed}"
        )

    if "Data da denúncia - Ano" not in dataframe.columns:
        raise ValueError(
            "Cabeçalho inválido detectado. A coluna esperada "
            "'Data da denúncia - Ano' não foi encontrada."
        )

    print("Cabeçalho validado com sucesso.")


def remover_linhas_totalmente_vazias(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Remove linhas em que todos os valores são nulos.
    """

    qtd_antes = len(dataframe)

    dataframe = dataframe.dropna(how="all").copy()

    qtd_depois = len(dataframe)
    qtd_removidas = qtd_antes - qtd_depois

    print(f"Linhas totalmente vazias removidas: {qtd_removidas}")

    return dataframe


def padronizar_textos(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Padroniza colunas de texto:
    - Remove espaços no início e no fim
    - Remove múltiplos espaços internos
    """

    dataframe = dataframe.copy()

    colunas_texto = dataframe.select_dtypes(include=["object", "string"]).columns

    for coluna in colunas_texto:
        dataframe[coluna] = (
            dataframe[coluna]
            .astype("string")
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
        )

    print(f"Colunas textuais padronizadas: {len(colunas_texto)}")

    return dataframe


def converter_textos_nulos_para_na(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Converte textos que representam ausência de informação para pd.NA.

    Exemplos tratados:
    - ""
    - " "
    - "NULL"
    - "null"
    - "NAN"
    - "nan"
    - "NaN"
    - "None"
    - "none"
    - "N/D"
    - "ND"
    - "NA"
    - "N.A"
    - "Não informado"
    - "Nao informado"
    """

    dataframe = dataframe.copy()

    colunas_texto = dataframe.select_dtypes(include=["object", "string"]).columns

    valores_nulos_textuais = {
        "",
        "null",
        "nan",
        "none",
        "n/d",
        "nd",
        "na",
        "n.a",
        "não informado",
        "nao informado",
    }

    total_convertidos = 0

    for coluna in colunas_texto:
        serie_original = dataframe[coluna]

        mascara_nulos_textuais = (
            serie_original
            .astype("string")
            .str.strip()
            .str.lower()
            .isin(valores_nulos_textuais)
        )

        qtd_convertidos_coluna = int(mascara_nulos_textuais.sum())
        total_convertidos += qtd_convertidos_coluna

        dataframe.loc[mascara_nulos_textuais, coluna] = pd.NA

        if qtd_convertidos_coluna > 0:
            print(
                f"Coluna '{coluna}': "
                f"{qtd_convertidos_coluna} valores textuais nulos convertidos para pd.NA."
            )

    print(f"Total de valores NULL/NAN/N/D/vazios convertidos para pd.NA: {total_convertidos}")

    return dataframe

def tratar_nulos_colunas_criticas(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Trata valores nulos em colunas críticas da base.

    A coluna 'faixa_etaria_suspeito' possui alto percentual de ausência.
    Por isso, os valores ausentes são preenchidos com uma categoria explícita,
    indicando que a informação do suspeito não foi informada.
    """

    dataframe = dataframe.copy()

    if "faixa_etaria_suspeito" in dataframe.columns:
        qtd_nulos = dataframe["faixa_etaria_suspeito"].isna().sum()

        dataframe["faixa_etaria_suspeito"] = dataframe["faixa_etaria_suspeito"].fillna(
            "info_suspeito_nao_informada"
        )

        print(
            "Coluna 'faixa_etaria_suspeito': "
            f"{qtd_nulos} valores nulos preenchidos com "
            "'info_suspeito_nao_informada'."
        )

    if "faixa_etaria_vitima" in dataframe.columns:
        qtd_nulos = dataframe["faixa_etaria_vitima"].isna().sum()

        dataframe["faixa_etaria_vitima"] = dataframe["faixa_etaria_vitima"].fillna(
            "info_vitima_nao_informada"
        )

        print(
            "Coluna 'faixa_etaria_vitima': "
            f"{qtd_nulos} valores nulos preenchidos com "
            "'info_vitima_nao_informada'."
        )

    if "municipio" in dataframe.columns:
        qtd_nulos = dataframe["municipio"].isna().sum()

        dataframe["municipio"] = dataframe["municipio"].fillna(
            "municipio_nao_informado"
        )

        print(
            "Coluna 'municipio': "
            f"{qtd_nulos} valores nulos preenchidos com "
            "'municipio_nao_informado'."
        )

    return dataframe

def relatorio_limpeza(dataframe: pd.DataFrame) -> None:
    """
    Exibe um relatório simples após a limpeza.
    """

    print("\n" + "=" * 80)
    print("RELATÓRIO APÓS LIMPEZA")
    print("=" * 80)

    print("\nQuantidade de linhas e colunas:")
    print(dataframe.shape)

    print("\nColunas da base após limpeza:")
    for indice, coluna in enumerate(dataframe.columns, start=1):
        print(f"{indice:02d}. {coluna}")

    print("\nTotal de valores nulos:")
    print(dataframe.isna().sum().sum())

    print("\nValores nulos por coluna:")
    print(dataframe.isna().sum())

    print("\nPrévia dos dados limpos:")
    print(dataframe.head())


def limpar_dados_bronze(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Executa a limpeza inicial da camada bronze.

    Fluxo:
    1. Valida o cabeçalho
    2. Remove linhas totalmente vazias
    3. Trata nomes das colunas
    4. Padroniza textos
    5. Converte textos nulos para pd.NA
    6. Trata nulos críticos com categorias explícitas
    7. Exibe relatório da limpeza
    """

    print("\n" + "=" * 80)
    print("INÍCIO DA LIMPEZA DOS DADOS - CAMADA BRONZE")
    print("=" * 80)

    validar_cabecalho(dataframe)

    dataframe = remover_linhas_totalmente_vazias(dataframe)
    dataframe = tratar_nomes_colunas(dataframe)
    dataframe = padronizar_textos(dataframe)
    dataframe = converter_textos_nulos_para_na(dataframe)
    dataframe = tratar_nulos_colunas_criticas(dataframe)

    relatorio_limpeza(dataframe)

    print("\n" + "=" * 80)
    print("FIM DA LIMPEZA DOS DADOS - CAMADA BRONZE")
    print("=" * 80)

    return dataframe

def salvar_dataframe_limpo(
    dataframe: pd.DataFrame,
    caminho_saida: str | Path,
) -> None:
    """
    Salva o DataFrame limpo em formato CSV.
    """

    caminho_saida = Path(caminho_saida)

    caminho_saida.parent.mkdir(parents=True, exist_ok=True)

    dataframe.to_csv(
        caminho_saida,
        index=False,
        encoding="utf-8",
    )

    print(f"DataFrame limpo salvo com sucesso em: {caminho_saida}")