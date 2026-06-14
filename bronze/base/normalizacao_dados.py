# ======================================================================================
# normalizacao_dados.py
# ======================================================================================
# Responsabilidade:
# - Aplicar encoding simples nas variáveis categóricas
# - Preparar X e y em formato numérico para modelagem
# ======================================================================================

import pandas as pd


def aplicar_encoding_simples(
    X: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aplica One-Hot Encoding simples usando pd.get_dummies.
    """

    print("\n" + "=" * 80)
    print("ENCODING SIMPLES DAS VARIÁVEIS EXPLICATIVAS")
    print("=" * 80)

    qtd_colunas_antes = X.shape[1]

    X_encoded = pd.get_dummies(
        X,
        drop_first=False,
        dtype=int,
    )

    qtd_colunas_depois = X_encoded.shape[1]

    print(f"Quantidade de colunas antes do encoding: {qtd_colunas_antes}")
    print(f"Quantidade de colunas após encoding: {qtd_colunas_depois}")

    print("\nFormato final de X após encoding:")
    print(X_encoded.shape)

    return X_encoded


def codificar_variavel_alvo(
    y: pd.Series,
) -> pd.Series:
    """
    Codifica a variável alvo em valores numéricos simples.
    """

    print("\n" + "=" * 80)
    print("CODIFICAÇÃO DA VARIÁVEL ALVO")
    print("=" * 80)

    categorias = sorted(y.unique())

    mapeamento = {
        categoria: codigo
        for codigo, categoria in enumerate(categorias)
    }

    y_encoded = y.map(mapeamento)

    print("Mapeamento da variável alvo:")

    for categoria, codigo in mapeamento.items():
        print(f"- {codigo}: {categoria}")

    print("\nFormato final de y após encoding:")
    print(y_encoded.shape)

    return y_encoded


def executar_normalizacao_dados(
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Executa a normalização inicial dos dados.

    Nesta etapa:
    - X recebe One-Hot Encoding
    - y recebe codificação numérica
    """

    print("\n" + "=" * 80)
    print("INÍCIO DA NORMALIZAÇÃO DOS DADOS")
    print("=" * 80)

    X_encoded = aplicar_encoding_simples(X)
    y_encoded = codificar_variavel_alvo(y)

    print("\n" + "=" * 80)
    print("FIM DA NORMALIZAÇÃO DOS DADOS")
    print("=" * 80)

    return X_encoded, y_encoded