# ======================================================================================
# estatisticas_descritivas.py
# ======================================================================================
# Responsabilidade:
# - Gerar estatísticas descritivas da base limpa
# - Avaliar volume de registros, colunas, tipos de dados
# - Investigar valores nulos por coluna
# - Investigar frequência das principais categorias
# - Apoiar decisões sobre qualidade dos dados
# ======================================================================================

import pandas as pd


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


def estatistica_dimensoes(dataframe: pd.DataFrame) -> None:
    """
    Exibe quantidade de linhas e colunas da base.
    """

    imprimir_secao(1, "DIMENSÕES DA BASE LIMPA")

    print(f"Quantidade de linhas: {dataframe.shape[0]}")
    print(f"Quantidade de colunas: {dataframe.shape[1]}")


def estatistica_tipos_dados(dataframe: pd.DataFrame) -> None:
    """
    Exibe os tipos de dados por coluna.
    """

    imprimir_secao(2, "TIPOS DE DADOS")

    print(dataframe.dtypes)


def estatistica_valores_nulos(dataframe: pd.DataFrame) -> None:
    """
    Exibe quantidade e percentual de valores nulos por coluna.
    """

    imprimir_secao(3, "VALORES NULOS POR COLUNA")

    total_linhas = len(dataframe)

    resumo_nulos = pd.DataFrame({
        "qtd_nulos": dataframe.isna().sum(),
        "percentual_nulos": (dataframe.isna().sum() / total_linhas) * 100,
    })

    resumo_nulos = resumo_nulos.sort_values(
        by="percentual_nulos",
        ascending=False,
    )

    print(resumo_nulos)


def estatistica_colunas_criticas_nulos(
    dataframe: pd.DataFrame,
    limite_percentual: float = 30.0,
) -> None:
    """
    Lista colunas com percentual de nulos acima de um limite definido.
    """

    imprimir_secao(4, "COLUNAS COM ALTO PERCENTUAL DE NULOS")

    total_linhas = len(dataframe)

    percentual_nulos = (dataframe.isna().sum() / total_linhas) * 100

    colunas_criticas = percentual_nulos[
        percentual_nulos >= limite_percentual
    ].sort_values(ascending=False)

    if colunas_criticas.empty:
        print(f"Nenhuma coluna possui {limite_percentual}% ou mais de nulos.")
        return

    print(f"Colunas com {limite_percentual}% ou mais de nulos:\n")

    for coluna, percentual in colunas_criticas.items():
        qtd_nulos = dataframe[coluna].isna().sum()
        print(f"- {coluna}: {qtd_nulos} nulos ({percentual:.2f}%)")


def estatistica_valores_unicos(dataframe: pd.DataFrame) -> None:
    """
    Exibe a quantidade de valores únicos por coluna.
    """

    imprimir_secao(5, "QUANTIDADE DE VALORES ÚNICOS POR COLUNA")

    valores_unicos = dataframe.nunique(dropna=True).sort_values(ascending=False)

    print(valores_unicos)


def estatistica_frequencia_coluna(
    dataframe: pd.DataFrame,
    coluna: str,
    top_n: int = 10,
) -> None:
    """
    Exibe as categorias mais frequentes de uma coluna.
    """

    if coluna not in dataframe.columns:
        print(f"[AVISO] Coluna '{coluna}' não encontrada.")
        return

    print(f"\nTop {top_n} valores da coluna '{coluna}':")
    print(dataframe[coluna].value_counts(dropna=False).head(top_n))


def estatistica_frequencias_principais(dataframe: pd.DataFrame) -> None:
    """
    Exibe frequências das principais colunas categóricas do estudo.
    """

    imprimir_secao(6, "FREQUÊNCIAS DAS PRINCIPAIS VARIÁVEIS CATEGÓRICAS")

    colunas_interesse = [
        "tipo_violacao",
        "grupo_vulneravel",
        "especie_violacao",
        "canal_atendimento",
        "cenario_violacao",
        "denuncia_emergencial",
        "uf",
        "municipio",
        "sexo_vitima",
        "faixa_etaria_vitima",
        "sexo_suspeito",
        "faixa_etaria_suspeito",
        "relacao_vitima_suspeito",
    ]

    for coluna in colunas_interesse:
        estatistica_frequencia_coluna(
            dataframe=dataframe,
            coluna=coluna,
            top_n=10,
        )


def estatistica_info_nao_informada(dataframe: pd.DataFrame) -> None:
    """
    Investiga categorias criadas para representar informação não informada.
    """

    imprimir_secao(7, "INVESTIGAÇÃO DE INFORMAÇÕES NÃO INFORMADAS")

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

        print(f"- {coluna}: {qtd} registros ({percentual:.2f}%) com '{categoria}'")


def gerar_estatisticas_descritivas(dataframe: pd.DataFrame) -> None:
    """
    Executa o relatório completo de estatísticas descritivas.
    """

    imprimir_titulo("ESTATÍSTICAS DESCRITIVAS - BASE FEMINICÍDIO")
    
    estatistica_dimensoes(dataframe)
    estatistica_tipos_dados(dataframe)
    estatistica_valores_nulos(dataframe)
    estatistica_colunas_criticas_nulos(dataframe, limite_percentual=30.0)
    estatistica_valores_unicos(dataframe)
    estatistica_frequencias_principais(dataframe)
    estatistica_info_nao_informada(dataframe)

    imprimir_titulo("FIM DAS ESTATÍSTICAS DESCRITIVAS")