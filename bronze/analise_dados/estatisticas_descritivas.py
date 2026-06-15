# ======================================================================================
# estatisticas_descritivas.py
# ======================================================================================
# Responsabilidade:
# - Gerar estatísticas descritivas da base limpa
# - Avaliar volume de registros, colunas e tipos de dados
# - Investigar valores nulos por coluna
# - Investigar frequência das principais categorias
# - Apoiar decisões iniciais sobre qualidade dos dados
# ======================================================================================
import pandas as pd
from Pesquisa_principal.constants import CATEGORIAS_NAO_INFORMADAS, COLUNAS_CATEGORICAS_INTERESSE, LIMITE_PERCENTUAL_NULOS_CRITICO

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
    Valida se o DataFrame possui dados para geração das estatísticas.
    """

    if dataframe.empty:
        print("[AVISO] DataFrame vazio. Estatísticas descritivas não executadas.")
        return False

    if dataframe.shape[1] == 0:
        print("[AVISO] DataFrame sem colunas. Estatísticas descritivas não executadas.")
        return False

    return True

def estatistica_dimensoes(dataframe: pd.DataFrame) -> None:
    """
    Exibe quantidade de linhas e colunas da base.
    """

    imprimir_secao(1, "DIMENSÕES DA BASE LIMPA")

    print(f"Quantidade de linhas: {dataframe.shape[0]}")
    print(f"Quantidade de colunas: {dataframe.shape[1]}")

def estatistica_colunas_duplicadas(dataframe: pd.DataFrame) -> None:
    """
    Verifica se existem colunas duplicadas na base.
    """

    imprimir_secao(2, "COLUNAS DUPLICADAS")

    colunas_duplicadas = dataframe.columns[
        dataframe.columns.duplicated()
    ].tolist()

    if not colunas_duplicadas:
        print("Nenhuma coluna duplicada encontrada.")
        return

    print("Colunas duplicadas encontradas:")
    for coluna in colunas_duplicadas:
        print(f"- {coluna}")

def estatistica_tipos_dados(dataframe: pd.DataFrame) -> None:
    """
    Exibe os tipos de dados por coluna.
    """

    imprimir_secao(3, "TIPOS DE DADOS")

    print(dataframe.dtypes)

def estatistica_valores_nulos(dataframe: pd.DataFrame) -> None:
    """
    Exibe quantidade e percentual de valores nulos por coluna.
    """

    imprimir_secao(4, "VALORES NULOS POR COLUNA")

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

def estatistica_colunas_criticas_nulos(
    dataframe: pd.DataFrame,
    limite_percentual: float = LIMITE_PERCENTUAL_NULOS_CRITICO,
) -> None:
    """
    Lista colunas com percentual de nulos acima de um limite definido.
    """

    imprimir_secao(5, "COLUNAS COM ALTO PERCENTUAL DE NULOS")

    total_linhas = len(dataframe)

    percentual_nulos = (dataframe.isna().sum() / total_linhas) * 100

    colunas_criticas = percentual_nulos[
        percentual_nulos >= limite_percentual
    ].sort_values(ascending=False)

    if colunas_criticas.empty:
        print(f"Nenhuma coluna possui {limite_percentual:.2f}% ou mais de nulos.")
        return

    print(f"Colunas com {limite_percentual:.2f}% ou mais de nulos:\n")

    for coluna, percentual in colunas_criticas.items():
        qtd_nulos = dataframe[coluna].isna().sum()
        print(f"- {coluna}: {qtd_nulos} nulos ({percentual:.2f}%)")

def estatistica_valores_unicos(dataframe: pd.DataFrame) -> None:
    """
    Exibe a quantidade de valores únicos por coluna.
    """

    imprimir_secao(6, "QUANTIDADE DE VALORES ÚNICOS POR COLUNA")

    valores_unicos = dataframe.nunique(dropna=True).sort_values(ascending=False)

    print(valores_unicos.to_string())

def estatistica_descritiva_numerica(dataframe: pd.DataFrame) -> None:
    """
    Exibe estatísticas descritivas para colunas numéricas.
    """

    imprimir_secao(7, "ESTATÍSTICAS DE VARIÁVEIS NUMÉRICAS")

    colunas_numericas = dataframe.select_dtypes(include=["number"])

    if colunas_numericas.empty:
        print("Nenhuma coluna numérica encontrada.")
        return

    print(
        colunas_numericas
        .describe()
        .transpose()
        .round(2)
        .to_string()
    )

def estatistica_frequencia_coluna(
    dataframe: pd.DataFrame,
    coluna: str,
) -> None:
    """
    Exibe todas as categorias de uma coluna.
    """

    if coluna not in dataframe.columns:
        print(f"[AVISO] Coluna '{coluna}' não encontrada.")
        return

    frequencia = dataframe[coluna].value_counts(dropna=False)

    percentual = (
        dataframe[coluna]
        .value_counts(dropna=False, normalize=True)
        .mul(100)
        .round(2)
    )

    resultado = pd.DataFrame({
        "quantidade": frequencia,
        "percentual (%)": percentual,
    })

    print(f"\nValores da coluna '{coluna}':")
    print(resultado.to_string())

def estatistica_frequencias_principais(dataframe: pd.DataFrame) -> None:
    """
    Exibe frequências das principais colunas categóricas do estudo.
    """

    imprimir_secao(8, "FREQUÊNCIAS DAS PRINCIPAIS VARIÁVEIS CATEGÓRICAS")

    for coluna in COLUNAS_CATEGORICAS_INTERESSE:
        estatistica_frequencia_coluna(
            dataframe=dataframe,
            coluna=coluna,
        )

def estatistica_info_nao_informada(dataframe: pd.DataFrame) -> None:
    """
    Investiga categorias criadas para representar informação não informada.
    """

    imprimir_secao(9, "INVESTIGAÇÃO DE INFORMAÇÕES NÃO INFORMADAS")

    categorias_investigar = {
        "faixa_etaria_suspeito": "info_suspeito_nao_informada",
        "faixa_etaria_vitima": "info_vitima_nao_informada",
    }

    total_linhas = len(dataframe)

    for coluna, categoria in categorias_investigar.items():
        if coluna not in dataframe.columns:
            print(f"- {coluna}: coluna não encontrada.")
            continue

        qtd = (dataframe[coluna] == categoria).sum()
        percentual = (qtd / total_linhas) * 100

        print(
            f"- {coluna}: {qtd} registros "
            f"({percentual:.2f}%) com '{categoria}'"
        )

def gerar_estatisticas_descritivas(dataframe: pd.DataFrame) -> None:
    """
    Executa o relatório completo de estatísticas descritivas.
    """

    imprimir_titulo("ESTATÍSTICAS DESCRITIVAS - BASE FEMINICÍDIO")

    if not validar_dataframe(dataframe):
        return

    estatistica_dimensoes(dataframe)
    estatistica_colunas_duplicadas(dataframe)
    estatistica_tipos_dados(dataframe)
    estatistica_valores_nulos(dataframe)
    estatistica_colunas_criticas_nulos(
        dataframe=dataframe,
        limite_percentual=LIMITE_PERCENTUAL_NULOS_CRITICO,
    )
    estatistica_valores_unicos(dataframe)
    estatistica_descritiva_numerica(dataframe)
    estatistica_frequencias_principais(dataframe)
    estatistica_info_nao_informada(dataframe)

    imprimir_titulo("FIM DAS ESTATÍSTICAS DESCRITIVAS")