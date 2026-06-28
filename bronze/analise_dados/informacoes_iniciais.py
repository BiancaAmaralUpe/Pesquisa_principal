# ======================================================================================
# informacoes_iniciais.py
# ======================================================================================
# Responsabilidade:
# - Exibir informações iniciais da base
# - Organizar o diagnóstico em formato de relatório textual
# - Identificar problemas básicos antes da limpeza dos dados
# ======================================================================================
# imports das libs 
import pandas as pd

# imports das variaveis constants.
from Pesquisa_principal.constants import VALORES_TEXTUAIS_PROBLEMATICOS

def imprimir_titulo(titulo: str) -> None:
    """
    Imprime um título principal formatado.
    """

    print("\n" + "=" * 80)
    print(titulo)
    print("=" * 80)

def imprimir_secao(numero: int, titulo: str) -> None:
    """
    Imprime uma seção numerada do relatório.
    """

    print("\n" + "-" * 80)
    print(f"[{numero}] {titulo}")
    print("-" * 80)

def validar_dataframe(dataframe: pd.DataFrame) -> bool:
    """
    Verifica se o DataFrame possui dados para diagnóstico.
    """

    if dataframe.empty:
        print("[AVISO] DataFrame vazio. Diagnóstico inicial não executado.")
        return False

    if dataframe.shape[1] == 0:
        print("[AVISO] DataFrame sem colunas. Diagnóstico inicial não executado.")
        return False

    return True

def diagnosticar_dimensoes(dataframe: pd.DataFrame) -> None:
    """
    Exibe quantidade de linhas e colunas da base.
    """

    imprimir_secao(1, "DIMENSÕES DA BASE")

    print(f"Quantidade de linhas: {dataframe.shape[0]}")
    print(f"Quantidade de colunas: {dataframe.shape[1]}")

def diagnosticar_colunas(dataframe: pd.DataFrame) -> None:
    """
    Lista as colunas da base.
    """

    imprimir_secao(2, "COLUNAS DA BASE")

    for indice, coluna in enumerate(dataframe.columns, start=1):
        print(f"{indice:02d}. {coluna}")

def diagnosticar_colunas_duplicadas(dataframe: pd.DataFrame) -> None:
    """
    Verifica se existem colunas duplicadas.
    """

    imprimir_secao(3, "COLUNAS DUPLICADAS")

    colunas_duplicadas = dataframe.columns[
        dataframe.columns.duplicated()
    ].tolist()

    if not colunas_duplicadas:
        print("Nenhuma coluna duplicada encontrada.")
        return

    print("Colunas duplicadas encontradas:")
    for coluna in colunas_duplicadas:
        print(f"- {coluna}")

def diagnosticar_tipos_dados(dataframe: pd.DataFrame) -> None:
    """
    Exibe os tipos de dados por coluna.
    """

    imprimir_secao(4, "TIPOS DE DADOS")

    print(dataframe.dtypes.to_string())

def diagnosticar_valores_ausentes(dataframe: pd.DataFrame) -> None:
    """
    Exibe quantidade e percentual de valores ausentes por coluna.
    """

    imprimir_secao(5, "VALORES AUSENTES POR COLUNA")

    total_linhas = len(dataframe)

    resumo_nulos = pd.DataFrame({
        "qtd_nulos": dataframe.isna().sum(),
        "percentual_nulos": (dataframe.isna().sum() / total_linhas) * 100,
    })

    resumo_nulos = resumo_nulos.sort_values(
        by="percentual_nulos",
        ascending=False,
    )

    print(resumo_nulos.round(2).to_string())

def diagnosticar_resumo_ausentes(dataframe: pd.DataFrame) -> None:
    """
    Exibe resumo geral de valores ausentes.
    """

    imprimir_secao(6, "RESUMO DE VALORES AUSENTES")

    total_linhas = len(dataframe)
    nulos_por_coluna = dataframe.isna().sum()
    total_nulos = nulos_por_coluna.sum()
    colunas_com_nulos = nulos_por_coluna[nulos_por_coluna > 0]

    print(f"Total de valores nulos: {total_nulos}")
    print(f"Quantidade de colunas com nulos: {len(colunas_com_nulos)}")

    if len(colunas_com_nulos) == 0:
        print("Nenhuma coluna possui valores nulos.")
        return

    print("\nColunas com valores nulos:")

    for coluna, quantidade in colunas_com_nulos.sort_values(ascending=False).items():
        percentual = (quantidade / total_linhas) * 100
        print(f"- {coluna}: {quantidade} nulos ({percentual:.2f}%)")

def diagnosticar_valores_textuais_problematicos(dataframe: pd.DataFrame) -> None:
    """
    Identifica valores textuais que podem representar ausência de informação.
    """

    imprimir_secao(7, "VALORES TEXTUAIS PROBLEMÁTICOS")

    colunas_texto = dataframe.select_dtypes(include=["object"])

    if colunas_texto.empty:
        print("Nenhuma coluna textual encontrada.")
        return

    total_problematicos_geral = 0

    for coluna in colunas_texto.columns:
        serie_texto = (
            colunas_texto[coluna]
            .dropna()
            .astype(str)
            .str.strip()
            .str.lower()
        )

        quantidade_problematicos = serie_texto.isin(
            VALORES_TEXTUAIS_PROBLEMATICOS
        ).sum()

        if quantidade_problematicos > 0:
            total_problematicos_geral += quantidade_problematicos
            print(f"- {coluna}: {quantidade_problematicos} valores problemáticos")

    if total_problematicos_geral == 0:
        print("Nenhum valor textual problemático encontrado.")
    else:
        print(f"\nTotal geral de valores textuais problemáticos: {total_problematicos_geral}")

def diagnosticar_uso_memoria(dataframe: pd.DataFrame) -> None:
    """
    Exibe o uso aproximado de memória do DataFrame.
    """

    imprimir_secao(8, "USO DE MEMÓRIA")

    memoria_mb = dataframe.memory_usage(deep=True).sum() / (1024 ** 2)

    print(f"Uso aproximado de memória: {memoria_mb:.2f} MB")

def exibir_previa_dados(dataframe: pd.DataFrame, qtd_linhas: int = 5) -> None:
    """
    Exibe uma prévia dos dados.
    """

    imprimir_secao(9, "PRÉVIA DOS DADOS")

    print(dataframe.head(qtd_linhas).to_string())

def exibir_resumo_final(dataframe: pd.DataFrame) -> None:
    """
    Exibe um resumo final do diagnóstico inicial.
    """

    imprimir_secao(10, "RESUMO FINAL DO DIAGNÓSTICO")

    total_nulos = dataframe.isna().sum().sum()
    colunas_duplicadas = dataframe.columns[
        dataframe.columns.duplicated()
    ].tolist()

    print("Diagnóstico inicial concluído.")
    print("A base foi carregada e inspecionada com sucesso.")

    if total_nulos > 0:
        print(
            "Observação: existem valores ausentes que deverão ser avaliados "
            "nas etapas de limpeza, completude e preparação dos dados."
        )
    else:
        print("Observação: não foram encontrados valores ausentes.")

    if colunas_duplicadas:
        print(
            "Observação: existem colunas duplicadas que devem ser tratadas "
            "antes das próximas etapas."
        )

def diagnostico_inicial(dataframe: pd.DataFrame) -> None:
    """
    Exibe um diagnóstico inicial organizado da base de dados.
    """

    imprimir_titulo("PIPELINE BRONZE - DIAGNÓSTICO INICIAL DA BASE")

    if not validar_dataframe(dataframe):
        return

    diagnosticar_dimensoes(dataframe)
    diagnosticar_colunas(dataframe)
    diagnosticar_colunas_duplicadas(dataframe)
    diagnosticar_tipos_dados(dataframe)
    diagnosticar_valores_ausentes(dataframe)
    diagnosticar_resumo_ausentes(dataframe)
    diagnosticar_valores_textuais_problematicos(dataframe)
    diagnosticar_uso_memoria(dataframe)
    exibir_previa_dados(dataframe)
    exibir_resumo_final(dataframe)