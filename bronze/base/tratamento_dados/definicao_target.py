# ======================================================================================
# definicao_target.py
# ======================================================================================
# Responsabilidade:
# - Documentar a variável alvo da modelagem
# - Exibir distribuição das classes do target
# - Validar se o target existe na base
# - Registrar colunas removidas para evitar vazamento de informação
# ======================================================================================

import pandas as pd


def definir_target_modelagem(
    dataframe: pd.DataFrame,
    coluna_alvo: str,
    colunas_removidas_por_vazamento: list[str],
) -> None:
    """
    Documenta a definição da variável alvo da modelagem.

    Esta função não altera o DataFrame.
    Ela apenas registra informações metodológicas no output da preparação de modelagem.
    """

    print("\n" + "=" * 80)
    print("DEFINIÇÃO DO TARGET DA MODELAGEM")
    print("=" * 80)

    if coluna_alvo not in dataframe.columns:
        raise ValueError(f"Coluna alvo não encontrada na base: {coluna_alvo}")

    print(f"\nTarget definido: {coluna_alvo}")

    print("\nDescrição metodológica:")
    print(
        "A variável alvo representa a classificação dos registros conforme "
        "a presença e a intensidade de sinais de risco associados ao feminicídio."
    )

    print("\nObjetivo da modelagem:")
    print(
        "O objetivo é treinar modelos capazes de identificar padrões associados "
        "às classes de risco a partir das demais características disponíveis na denúncia."
    )

    print("\nDistribuição do target:")
    distribuicao_target = dataframe[coluna_alvo].value_counts(dropna=False)
    print(distribuicao_target)

    print("\nDistribuição percentual do target:")
    distribuicao_percentual = dataframe[coluna_alvo].value_counts(
        normalize=True,
        dropna=False,
    ) * 100
    print(distribuicao_percentual.round(2))

    quantidade_nulos = dataframe[coluna_alvo].isna().sum()
    print(f"\nQuantidade de nulos no target: {quantidade_nulos}")

    print("\nColunas removidas para evitar vazamento de informação:")
    for coluna in colunas_removidas_por_vazamento:
        if coluna in dataframe.columns:
            print(f"- {coluna}")
        else:
            print(f"- {coluna} [não encontrada na base]")

    print("\nJustificativa:")
    print(
        "As colunas removidas por vazamento foram utilizadas direta ou indiretamente "
        "na construção do target. Mantê-las no conjunto de treino poderia fazer com "
        "que o modelo aprendesse a regra de criação da variável alvo, e não padrões "
        "reais dos dados."
    )