# ======================================================================================
# separacao_target.py
# ======================================================================================
# Responsabilidade:
# - Separar a base de modelagem em X e y
# - Garantir que a coluna alvo não permaneça nas variáveis explicativas
# - Exibir informações da separação para o output da Silver
# ======================================================================================

import pandas as pd


def separar_variaveis_explicativas_e_target(
    dataframe: pd.DataFrame,
    coluna_alvo: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separa a base em variáveis explicativas (X) e variável alvo (y).
    """

    print("\n" + "=" * 80)
    print("SEPARAÇÃO ENTRE VARIÁVEIS EXPLICATIVAS E TARGET")
    print("=" * 80)

    if coluna_alvo not in dataframe.columns:
        raise ValueError(
            f"Coluna alvo não encontrada na base: {coluna_alvo}"
        )

    X = dataframe.drop(columns=[coluna_alvo])
    y = dataframe[coluna_alvo]

    if coluna_alvo in X.columns:
        raise ValueError(
            "Erro de separação: a coluna alvo permaneceu dentro de X."
        )

    print(f"Coluna alvo: {coluna_alvo}")

    print("\nFormato de X:")
    print(X.shape)

    print("\nFormato de y:")
    print(y.shape)

    print("\nColunas de X:")
    for coluna in X.columns:
        print(f"- {coluna}")

    print("\nDistribuição de y:")
    print(y.value_counts(dropna=False))

    print("\nDistribuição percentual de y:")
    print((y.value_counts(normalize=True, dropna=False) * 100).round(2))

    print("\nSeparação concluída com sucesso.")

    return X, y