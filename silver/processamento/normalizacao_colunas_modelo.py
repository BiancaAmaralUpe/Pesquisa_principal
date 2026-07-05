# ======================================================================================
# normalizacao_colunas_modelo.py
# ======================================================================================
# Responsabilidade:
# - Normalizar nomes das colunas após o One-Hot Encoding
# - Remover caracteres especiais dos nomes das features
# - Evitar erro no LightGBM com caracteres inválidos
# ======================================================================================

import re
import unicodedata

import pandas as pd


def remover_acentos(texto: str) -> str:
    """
    Remove acentos de um texto.
    """

    texto_normalizado = unicodedata.normalize(
        "NFKD",
        texto,
    )

    texto_sem_acento = texto_normalizado.encode(
        "ascii",
        "ignore",
    ).decode(
        "ascii",
    )

    return texto_sem_acento


def limpar_nome_coluna(nome_coluna: str) -> str:
    """
    Limpa o nome de uma coluna para ser compatível com modelos como LightGBM.
    """

    nome_coluna = str(nome_coluna)

    nome_coluna = remover_acentos(nome_coluna)

    nome_coluna = nome_coluna.lower()

    nome_coluna = re.sub(
        pattern=r"[^a-zA-Z0-9_]",
        repl="_",
        string=nome_coluna,
    )

    nome_coluna = re.sub(
        pattern=r"_+",
        repl="_",
        string=nome_coluna,
    )

    nome_coluna = nome_coluna.strip("_")

    if nome_coluna == "":
        nome_coluna = "coluna"

    if nome_coluna[0].isdigit():
        nome_coluna = f"coluna_{nome_coluna}"

    return nome_coluna


def gerar_nomes_colunas_unicos(
    colunas: list[str],
) -> list[str]:
    """
    Gera nomes de colunas limpos e únicos.

    Isso evita que duas colunas diferentes virem o mesmo nome depois da limpeza.
    """

    nomes_finais = []
    contagem_nomes = {}

    for coluna in colunas:
        nome_limpo = limpar_nome_coluna(coluna)

        if nome_limpo not in contagem_nomes:
            contagem_nomes[nome_limpo] = 0
            nomes_finais.append(nome_limpo)
            continue

        contagem_nomes[nome_limpo] += 1

        nome_unico = f"{nome_limpo}_{contagem_nomes[nome_limpo]}"

        nomes_finais.append(nome_unico)

    return nomes_finais


def normalizar_nomes_colunas_modelo(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Normaliza os nomes das colunas de treino e teste.

    Importante:
    - Não altera os valores da base.
    - Só altera os nomes das colunas.
    - Deve ser aplicado depois do encoding.
    """

    print("\n" + "=" * 80)
    print("NORMALIZAÇÃO DOS NOMES DAS COLUNAS PARA OS MODELOS")
    print("=" * 80)

    if list(X_train.columns) != list(X_test.columns):
        raise ValueError(
            "As colunas de X_train e X_test estão diferentes. "
            "Não é seguro aplicar a normalização dos nomes."
        )

    colunas_originais = list(X_train.columns)

    colunas_normalizadas = gerar_nomes_colunas_unicos(
        colunas=colunas_originais,
    )

    X_train_normalizado = X_train.copy()
    X_test_normalizado = X_test.copy()

    X_train_normalizado.columns = colunas_normalizadas
    X_test_normalizado.columns = colunas_normalizadas

    quantidade_colunas = len(colunas_originais)

    quantidade_colunas_alteradas = sum(
        coluna_original != coluna_normalizada
        for coluna_original, coluna_normalizada in zip(
            colunas_originais,
            colunas_normalizadas,
        )
    )

    print(f"Quantidade de colunas: {quantidade_colunas}")
    print(f"Quantidade de nomes alterados: {quantidade_colunas_alteradas}")

    print("\nExemplos de nomes normalizados:")

    exemplos_exibidos = 0

    for coluna_original, coluna_normalizada in zip(
        colunas_originais,
        colunas_normalizadas,
    ):
        if coluna_original != coluna_normalizada:
            print(f"- {coluna_original}  ->  {coluna_normalizada}")
            exemplos_exibidos += 1

        if exemplos_exibidos >= 20:
            break

    print("\nNormalização dos nomes das colunas concluída com sucesso.")

    return X_train_normalizado, X_test_normalizado