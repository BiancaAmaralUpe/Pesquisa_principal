# ======================================================================================
# informacoes_iniciais.py
# ======================================================================================
# Responsabilidade:
# - Exibir informações iniciais da base
# - Organizar o diagnóstico em formato de relatório textual
# ======================================================================================

import pandas as pd

def imprimir_titulo(titulo: str) -> None:
    """
    Imprime um título principal formatado.
    """

    print("=" * 80)
    print(titulo)
    print("=" * 80)


def imprimir_secao(numero: int, titulo: str) -> None:
    """
    Imprime uma seção numerada do relatório.
    """

    print("\n" + "-" * 80)
    print(f"[{numero}] {titulo}")
    print("-" * 80)


def diagnostico_inicial(dataframe: pd.DataFrame) -> None:
    """
    Exibe um diagnóstico inicial organizado da base de dados.
    """

    imprimir_titulo("PIPELINE BRONZE - DIAGNÓSTICO INICIAL DA BASE")

    imprimir_secao(1, "DIMENSÕES DA BASE")
    print(f"Quantidade de linhas: {dataframe.shape[0]}")
    print(f"Quantidade de colunas: {dataframe.shape[1]}")

    imprimir_secao(2, "COLUNAS DA BASE")
    for indice, coluna in enumerate(dataframe.columns, start=1):
        print(f"{indice:02d}. {coluna}")

    imprimir_secao(3, "TIPOS DE DADOS")
    print(dataframe.dtypes)

    imprimir_secao(4, "VALORES AUSENTES POR COLUNA")
    nulos_por_coluna = dataframe.isna().sum()
    print(nulos_por_coluna)

    imprimir_secao(5, "RESUMO DE VALORES AUSENTES")
    total_nulos = dataframe.isna().sum().sum()
    colunas_com_nulos = nulos_por_coluna[nulos_por_coluna > 0]

    print(f"Total de valores nulos: {total_nulos}")
    print(f"Quantidade de colunas com nulos: {len(colunas_com_nulos)}")

    if len(colunas_com_nulos) > 0:
        print("\nColunas com valores nulos:")
        for coluna, quantidade in colunas_com_nulos.items():
            percentual = (quantidade / len(dataframe)) * 100
            print(f"- {coluna}: {quantidade} nulos ({percentual:.2f}%)")
    else:
        print("Nenhuma coluna possui valores nulos.")

    imprimir_secao(6, "VALORES TEXTUAIS PROBLEMÁTICOS")
    colunas_texto = dataframe.select_dtypes(include=["object"])

    strings_vazias = (
        colunas_texto
        .map(lambda valor: isinstance(valor, str) and valor.strip() == "")
        .sum()
        .sum()
    )

    strings_nan_none = (
        colunas_texto
        .map(
            lambda valor: isinstance(valor, str)
            and valor.strip().lower() in ["nan", "none"]
        )
        .sum()
        .sum()
    )

    print(f"Strings vazias: {strings_vazias}")
    print(f"Strings com 'nan' ou 'none' como texto: {strings_nan_none}")

    imprimir_secao(7, "PRÉVIA DOS DADOS")
    print(dataframe.head())

    imprimir_secao(8, "RESUMO FINAL DO DIAGNÓSTICO")

    print("Diagnóstico inicial concluído.")
    print("A base foi carregada e inspecionada com sucesso.")

    if total_nulos > 0:
        print("Observação: existem valores ausentes que deverão ser tratados na próxima camada.")
    else:
        print("Observação: não foram encontrados valores ausentes.")