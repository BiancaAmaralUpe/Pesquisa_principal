# ======================================================================================
# divisao_treino_teste.py
# ======================================================================================
# Responsabilidade:
# - Dividir X e y em treino e teste
# - Preservar a proporção das classes com estratificação
# - Exibir a distribuição do target em treino e teste
# ======================================================================================

import pandas as pd
from sklearn.model_selection import train_test_split

def exibir_distribuicao_target(
    y: pd.Series,
    nome_base: str,
) -> None:
    """
    Exibe a distribuição absoluta e percentual do target.
    """

    print("\n" + "-" * 80)
    print(f"DISTRIBUIÇÃO DO TARGET - {nome_base}")
    print("-" * 80)

    print("\nDistribuição absoluta:")
    print(y.value_counts(dropna=False))

    print("\nDistribuição percentual:")
    print((y.value_counts(normalize=True, dropna=False) * 100).round(2))

def dividir_treino_teste_estratificado(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Divide a base em treino e teste com estratificação pelo target.

    A estratificação é importante porque o target está desbalanceado.
    Assim, treino e teste preservam proporções semelhantes das classes.
    """

    print("\n" + "=" * 80)
    print("DIVISÃO TREINO/TESTE ESTRATIFICADA")
    print("=" * 80)

    print(f"Tamanho do teste: {test_size}")
    print(f"Random state: {random_state}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    print("\nFormato dos dados:")
    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_test: {y_test.shape}")

    exibir_distribuicao_target(
        y=y,
        nome_base="BASE COMPLETA",
    )

    exibir_distribuicao_target(
        y=y_train,
        nome_base="TREINO",
    )

    exibir_distribuicao_target(
        y=y_test,
        nome_base="TESTE",
    )

    print("\nDivisão treino/teste concluída com sucesso.")

    return X_train, X_test, y_train, y_test