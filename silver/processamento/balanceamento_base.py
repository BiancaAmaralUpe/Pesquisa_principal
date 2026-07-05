# ======================================================================================
# balanceamento_base.py
# ======================================================================================
# Responsabilidade:
# - Aplicar balanceamento somente na base de treino
# - Evitar balanceamento na base de teste
# - Corrigir desbalanceamento do target antes do treinamento dos modelos
# ======================================================================================

import pandas as pd


def exibir_distribuicao_target(
    y: pd.Series,
    titulo: str,
) -> None:
    """
    Exibe a distribuição absoluta e percentual do target.
    """

    print("\n" + "=" * 80)
    print(titulo)
    print("=" * 80)

    distribuicao_absoluta = y.value_counts()

    distribuicao_percentual = (
        y.value_counts(normalize=True) * 100
    ).round(2)

    print("\nDistribuição absoluta:")
    print(distribuicao_absoluta)

    print("\nDistribuição percentual:")
    print(distribuicao_percentual)


def definir_quantidade_alvo_por_classe(
    y_train: pd.Series,
    estrategia: str,
    quantidade_alvo: int | None = None,
) -> int:
    """
    Define a quantidade final desejada para cada classe.

    Estratégias disponíveis:
    - oversampling: aumenta todas as classes até a maior classe
    - undersampling: reduz todas as classes até a menor classe
    - hibrido: ajusta todas as classes para uma quantidade intermediária
    """

    contagem_classes = y_train.value_counts()

    if estrategia == "oversampling":
        return int(contagem_classes.max())

    if estrategia == "undersampling":
        return int(contagem_classes.min())

    if estrategia == "hibrido":
        if quantidade_alvo is not None:
            return int(quantidade_alvo)

        return int(contagem_classes.median())

    raise ValueError(
        "Estratégia de balanceamento inválida. "
        "Use: 'oversampling', 'undersampling' ou 'hibrido'."
    )


def balancear_classe(
    dataframe_treino: pd.DataFrame,
    coluna_target: str,
    classe_target: str,
    quantidade_alvo: int,
    estrategia: str,
    random_state: int,
) -> pd.DataFrame:
    """
    Aplica amostragem em uma única classe do target.
    """

    dataframe_classe = dataframe_treino[
        dataframe_treino[coluna_target] == classe_target
    ]

    quantidade_atual = len(dataframe_classe)

    if quantidade_atual == quantidade_alvo:
        return dataframe_classe

    if quantidade_atual < quantidade_alvo:
        dataframe_balanceado = dataframe_classe.sample(
            n=quantidade_alvo,
            replace=True,
            random_state=random_state,
        )

        return dataframe_balanceado

    if quantidade_atual > quantidade_alvo:
        if estrategia == "oversampling":
            return dataframe_classe

        dataframe_balanceado = dataframe_classe.sample(
            n=quantidade_alvo,
            replace=False,
            random_state=random_state,
        )

        return dataframe_balanceado

    return dataframe_classe


def aplicar_balanceamento_treino(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    estrategia: str,
    random_state: int,
    quantidade_alvo: int | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Aplica balanceamento somente nos dados de treino.

    Importante:
    - A base de teste não deve ser balanceada.
    - O teste precisa continuar representando a distribuição real da base.
    """

    print("\n" + "=" * 80)
    print("BALANCEAMENTO DA BASE DE TREINO")
    print("=" * 80)

    print(f"Estratégia utilizada: {estrategia}")

    exibir_distribuicao_target(
        y=y_train,
        titulo="DISTRIBUIÇÃO DO TARGET ANTES DO BALANCEAMENTO",
    )

    coluna_target_temporaria = "__target_balanceamento__"

    dataframe_treino = X_train.copy()
    dataframe_treino[coluna_target_temporaria] = y_train.values

    quantidade_final_por_classe = definir_quantidade_alvo_por_classe(
        y_train=y_train,
        estrategia=estrategia,
        quantidade_alvo=quantidade_alvo,
    )

    print(
        "\nQuantidade alvo por classe após balanceamento: "
        f"{quantidade_final_por_classe}"
    )

    dataframes_balanceados = []

    classes_target = y_train.value_counts().index.tolist()

    for classe_target in classes_target:
        dataframe_classe_balanceada = balancear_classe(
            dataframe_treino=dataframe_treino,
            coluna_target=coluna_target_temporaria,
            classe_target=classe_target,
            quantidade_alvo=quantidade_final_por_classe,
            estrategia=estrategia,
            random_state=random_state,
        )

        dataframes_balanceados.append(dataframe_classe_balanceada)

    dataframe_balanceado = pd.concat(
        dataframes_balanceados,
        axis=0,
        ignore_index=True,
    )

    dataframe_balanceado = dataframe_balanceado.sample(
        frac=1,
        random_state=random_state,
    ).reset_index(
        drop=True,
    )

    y_train_balanceado = dataframe_balanceado[coluna_target_temporaria].copy()

    X_train_balanceado = dataframe_balanceado.drop(
        columns=[
            coluna_target_temporaria,
        ]
    )

    exibir_distribuicao_target(
        y=y_train_balanceado,
        titulo="DISTRIBUIÇÃO DO TARGET APÓS O BALANCEAMENTO",
    )

    print("\nFormato antes do balanceamento:")
    print(f"X_train: {X_train.shape}")
    print(f"y_train: {y_train.shape}")

    print("\nFormato após o balanceamento:")
    print(f"X_train_balanceado: {X_train_balanceado.shape}")
    print(f"y_train_balanceado: {y_train_balanceado.shape}")

    print("\nBalanceamento da base de treino concluído com sucesso.")

    return X_train_balanceado, y_train_balanceado