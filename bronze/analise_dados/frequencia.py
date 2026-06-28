# ======================================================================================
# frequencia.py
# ======================================================================================
# Responsabilidade:
# - Analisar distribuição das variáveis categóricas
# - Identificar categorias mais frequentes
# - Identificar categorias raras
# - Apoiar decisões sobre agrupamento, normalização e modelagem
# ======================================================================================
# imports das libs 
import pandas as pd

# imports das variaveis constants.
from Pesquisa_principal.constants import COLUNAS_FREQUENCIA, LIMITE_PERCENTUAL_CATEGORIA_RARA

def imprimir_secao(titulo: str) -> None:
    """
    Imprime uma seção formatada no terminal/output.
    """

    print("\n" + "=" * 80)
    print(titulo)
    print("=" * 80)

def validar_dataframe(dataframe: pd.DataFrame) -> bool:
    """
    Valida se o DataFrame possui dados para análise de frequência.
    """

    if dataframe.empty:
        print("[AVISO] DataFrame vazio. Análise de frequência não executada.")
        return False

    if dataframe.shape[1] == 0:
        print("[AVISO] DataFrame sem colunas. Análise de frequência não executada.")
        return False

    return True

def calcular_tabela_frequencia(
    dataframe: pd.DataFrame,
    coluna: str,
) -> pd.DataFrame:
    """
    Calcula quantidade e percentual de cada categoria de uma coluna.
    """

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

    return resultado

def analisar_frequencia_coluna(
    dataframe: pd.DataFrame,
    coluna: str,
) -> None:
    """
    Exibe todas as categorias de uma coluna.
    """

    if coluna not in dataframe.columns:
        print(f"[AVISO] Coluna deletada: {coluna}")
        return

    print("\n" + "=" * 80)
    print(f"FREQUÊNCIA - {coluna.upper()}")
    print("=" * 80)

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

    print(resultado.to_string())

def analisar_categorias_raras(
    dataframe: pd.DataFrame,
    coluna: str,
    limite_percentual: float = LIMITE_PERCENTUAL_CATEGORIA_RARA,
) -> None:
    """
    Identifica categorias com percentual abaixo do limite definido.
    """

    if coluna not in dataframe.columns:
        print(f"[AVISO] Coluna deletada: {coluna}")
        return

    resultado = calcular_tabela_frequencia(
        dataframe=dataframe,
        coluna=coluna,
    )

    categorias_raras = resultado[
        resultado["percentual (%)"] < limite_percentual
    ]

    print(f"\nCategorias raras com menos de {limite_percentual:.2f}%:")

    if categorias_raras.empty:
        print("- Nenhuma categoria rara encontrada.")
        return

    print(categorias_raras.to_string())

def analisar_resumo_cardinalidade(
    dataframe: pd.DataFrame,
    coluna: str,
) -> None:
    """
    Exibe resumo de cardinalidade da coluna.
    """

    if coluna not in dataframe.columns:
        print(f"[AVISO] Coluna deletada: {coluna}")
        return

    quantidade_categorias = dataframe[coluna].nunique(dropna=False)

    print(f"\nQuantidade de categorias distintas: {quantidade_categorias}")

def executar_analise_frequencias(
    dataframe: pd.DataFrame,
) -> None:
    """
    Executa análises de frequência das principais variáveis categóricas.
    """

    imprimir_secao("INÍCIO DA ANÁLISE DE FREQUÊNCIAS")

    if not validar_dataframe(dataframe):
        return

    for coluna in COLUNAS_FREQUENCIA:
        if coluna not in dataframe.columns:
            print(f"[AVISO] Coluna deletada: {coluna}")
            continue

        analisar_frequencia_coluna(
            dataframe=dataframe,
            coluna=coluna
        )

        analisar_resumo_cardinalidade(
            dataframe=dataframe,
            coluna=coluna,
        )

        analisar_categorias_raras(
            dataframe=dataframe,
            coluna=coluna,
            limite_percentual=LIMITE_PERCENTUAL_CATEGORIA_RARA,
        )

    imprimir_secao("FIM DA ANÁLISE DE FREQUÊNCIAS")