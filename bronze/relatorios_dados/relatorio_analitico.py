# ======================================================================================
# relatorio_analitico.py
# ======================================================================================
# Responsabilidade:
# - Documentar decisões da preparação analítica
# - Registrar colunas removidas
# - Registrar motivos metodológicos
# - Comparar a base antes e depois da preparação analítica
# - Validar o estado final da base
# ======================================================================================

import pandas as pd

from Pesquisa_principal.constants import DECISOES_PREPARACAO_ANALITICA


def imprimir_secao(titulo: str) -> None:
    """
    Imprime uma seção formatada no terminal/output.
    """

    print("\n" + "=" * 80)
    print(titulo)
    print("=" * 80)


def validar_dataframes(
    dataframe_original: pd.DataFrame,
    dataframe_final: pd.DataFrame,
) -> bool:
    """
    Valida se os DataFrames possuem estrutura para geração do relatório.
    """

    if dataframe_original.empty:
        print("[AVISO] DataFrame original vazio. Relatório não executado.")
        return False

    if dataframe_original.shape[1] == 0:
        print("[AVISO] DataFrame original sem colunas. Relatório não executado.")
        return False

    if dataframe_final.empty:
        print("[AVISO] DataFrame final vazio. Relatório não executado.")
        return False

    if dataframe_final.shape[1] == 0:
        print("[AVISO] DataFrame final sem colunas. Relatório não executado.")
        return False

    return True


def identificar_colunas_removidas(
    dataframe_original: pd.DataFrame,
    dataframe_final: pd.DataFrame,
    colunas_removidas: list[str] | None = None,
) -> list[str]:
    """
    Identifica colunas removidas durante a preparação analítica.

    Se uma lista for informada manualmente, ela será usada.
    Caso contrário, as colunas removidas serão inferidas pela comparação
    entre a base original e a base final.
    """

    if colunas_removidas is not None:
        return colunas_removidas

    colunas_originais = set(dataframe_original.columns)
    colunas_finais = set(dataframe_final.columns)

    return sorted(list(colunas_originais - colunas_finais))

def relatorio_dimensoes(
    dataframe_original: pd.DataFrame,
    dataframe_final: pd.DataFrame,
) -> None:
    """
    Exibe comparação de dimensões antes e depois da preparação analítica.
    """
    imprimir_secao("COMPARAÇÃO DE DIMENSÕES")

    linhas_antes = dataframe_original.shape[0]
    linhas_depois = dataframe_final.shape[0]

    colunas_originais = set(dataframe_original.columns)
    colunas_finais = set(dataframe_final.columns)

    colunas_removidas = sorted(list(colunas_originais - colunas_finais))
    colunas_criadas = sorted(list(colunas_finais - colunas_originais))

    print(f"Linhas antes: {linhas_antes}")
    print(f"Linhas depois: {linhas_depois}")
    print(f"Linhas removidas: {linhas_antes - linhas_depois}")

    print(f"\nColunas antes: {dataframe_original.shape[1]}")
    print(f"Colunas depois: {dataframe_final.shape[1]}")
    print(f"Colunas removidas: {len(colunas_removidas)}")
    print(f"Colunas criadas: {len(colunas_criadas)}")
    print(
        f"Saldo final de colunas: "
        f"{dataframe_final.shape[1] - dataframe_original.shape[1]}"
    )

    if colunas_criadas:
        print("\nColunas criadas na preparação analítica:")
        for coluna in colunas_criadas:
            print(f"- {coluna}")

def relatorio_colunas_removidas(
    colunas_removidas: list[str],
) -> None:
    """
    Exibe as colunas removidas e seus motivos metodológicos.
    """

    imprimir_secao("COLUNAS REMOVIDAS DA BASE ANALÍTICA")

    if not colunas_removidas:
        print("Nenhuma coluna foi removida.")
        return

    for coluna in colunas_removidas:
        decisao = DECISOES_PREPARACAO_ANALITICA.get(
            coluna,
            {
                "tipo_decisao": "não definido",
                "motivo": "Motivo metodológico não documentado no constants.py.",
            },
        )

        print(f"\n- Coluna: {coluna}")
        print(f"  Tipo de decisão: {decisao['tipo_decisao']}")
        print(f"  Motivo: {decisao['motivo']}")


def relatorio_colunas_mantidas(
    dataframe_final: pd.DataFrame,
) -> None:
    """
    Exibe as colunas mantidas na base final.
    """

    imprimir_secao("COLUNAS MANTIDAS NA BASE ANALÍTICA")

    for indice, coluna in enumerate(dataframe_final.columns, start=1):
        print(f"{indice:02d}. {coluna}")


def relatorio_nulos_restantes(
    dataframe_final: pd.DataFrame,
) -> None:
    """
    Exibe valores nulos restantes na base final.
    """

    imprimir_secao("VALORES NULOS RESTANTES NA BASE ANALÍTICA")

    nulos = dataframe_final.isna().sum()
    nulos = nulos[nulos > 0].sort_values(ascending=False)

    if nulos.empty:
        print("Nenhum valor nulo restante.")
        return

    total_linhas = len(dataframe_final)

    resumo_nulos = pd.DataFrame({
        "qtd_nulos": nulos,
        "percentual_nulos": (nulos / total_linhas) * 100,
    })

    print(resumo_nulos.round(2).to_string())


def relatorio_tipos_dados(
    dataframe_final: pd.DataFrame,
) -> None:
    """
    Exibe os tipos de dados da base analítica final.
    """

    imprimir_secao("TIPOS DE DADOS DA BASE ANALÍTICA")

    print(dataframe_final.dtypes.to_string())


def relatorio_validacao_final(
    dataframe_final: pd.DataFrame,
) -> None:
    """
    Exibe validação final da base analítica.
    """

    imprimir_secao("VALIDAÇÃO FINAL DA BASE ANALÍTICA")

    print(f"Linhas finais: {dataframe_final.shape[0]}")
    print(f"Colunas finais: {dataframe_final.shape[1]}")
    print(f"Total de valores nulos: {dataframe_final.isna().sum().sum()}")

    colunas_duplicadas = dataframe_final.columns[
        dataframe_final.columns.duplicated()
    ].tolist()

    if colunas_duplicadas:
        print("\n[AVISO] Colunas duplicadas encontradas:")
        for coluna in colunas_duplicadas:
            print(f"- {coluna}")
    else:
        print("\nNenhuma coluna duplicada encontrada.")

    print("\nBase preparada para análise/modelagem.")


def relatorio_preparacao_analitica(
    dataframe_original: pd.DataFrame,
    dataframe_final: pd.DataFrame,
    colunas_removidas: list[str] | None = None,
) -> None:
    """
    Gera relatório metodológico da preparação analítica.
    """

    imprimir_secao("RELATÓRIO DA PREPARAÇÃO ANALÍTICA DA BASE")

    if not validar_dataframes(
        dataframe_original=dataframe_original,
        dataframe_final=dataframe_final,
    ):
        return

    colunas_removidas_identificadas = identificar_colunas_removidas(
        dataframe_original=dataframe_original,
        dataframe_final=dataframe_final,
        colunas_removidas=colunas_removidas,
    )

    relatorio_dimensoes(
        dataframe_original=dataframe_original,
        dataframe_final=dataframe_final,
    )

    relatorio_colunas_removidas(
        colunas_removidas=colunas_removidas_identificadas,
    )

    relatorio_colunas_mantidas(
        dataframe_final=dataframe_final,
    )

    relatorio_nulos_restantes(
        dataframe_final=dataframe_final,
    )

    relatorio_tipos_dados(
        dataframe_final=dataframe_final,
    )

    relatorio_validacao_final(
        dataframe_final=dataframe_final,
    )