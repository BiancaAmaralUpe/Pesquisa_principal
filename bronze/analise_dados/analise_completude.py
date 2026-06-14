# ======================================================================================
# analise_completude.py
# ======================================================================================

import pandas as pd

def analisar_agravantes_policiais(
    dataframe: pd.DataFrame,
) -> None:
    """
    Analisa registros que possuem informação de agravantes policiais.
    """

    dados = dataframe.dropna(
        subset=["agravantes_policiais"]
    )

    percentual = (len(dados) / len(dataframe)) * 100

    print("\n" + "=" * 80)
    print("AGRAVANTES POLICIAIS")
    print("=" * 80)

    print(
        f"Registros com informação: "
        f"{len(dados)} ({percentual:.2f}%)"
    )

    print("\nDistribuição:")

    print(
        dados["agravantes_policiais"]
        .value_counts(dropna=False)
    )

def analisar_agravantes(dataframe: pd.DataFrame) -> None:
    """
    Analisa apenas registros que possuem agravantes.
    """

    dados = dataframe.dropna(subset=["agravantes"])

    percentual = (len(dados) / len(dataframe)) * 100

    print("\n" + "=" * 80)
    print("AGRAVANTES")
    print("=" * 80)

    print(
        f"Registros com informação: "
        f"{len(dados)} ({percentual:.2f}%)"
    )

    print("\nDistribuição:")
    print(
        dados["agravantes"]
        .value_counts()
        .head(20)
    )

def executar_analise_completude(
    dataframe: pd.DataFrame,
) -> None:
    """
    Executa todas as análises relacionadas à completude dos dados.
    """

    analisar_agravantes(dataframe)
    analisar_agravantes_policiais(dataframe)