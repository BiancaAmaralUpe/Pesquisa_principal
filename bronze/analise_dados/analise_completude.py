# ======================================================================================
# analise_completude.py
# ======================================================================================
# Responsabilidade:
# - Avaliar a completude da base de dados
# - Medir percentual de preenchimento e ausência por coluna
# - Identificar variáveis críticas por excesso de valores ausentes
# - Investigar campos com marcadores de informação não informada
# ======================================================================================

import pandas as pd


def imprimir_secao(titulo: str) -> None:
    """
    Imprime uma seção formatada no terminal/output.
    """

    print("\n" + "=" * 80)
    print(titulo)
    print("=" * 80)


def analisar_completude_geral(dataframe: pd.DataFrame) -> None:
    """
    Exibe a completude geral das colunas da base.

    A completude indica o percentual de registros preenchidos
    em cada variável.
    """

    imprimir_secao("ANÁLISE GERAL DE COMPLETUDE")

    total_linhas = len(dataframe)

    resumo = pd.DataFrame({
        "qtd_preenchidos": dataframe.notna().sum(),
        "qtd_nulos": dataframe.isna().sum(),
        "percentual_preenchido": (dataframe.notna().sum() / total_linhas) * 100,
        "percentual_nulos": (dataframe.isna().sum() / total_linhas) * 100,
    })

    resumo = resumo.sort_values(
        by="percentual_nulos",
        ascending=False,
    )

    print(resumo.round(2).to_string())


def classificar_colunas_por_completude(dataframe: pd.DataFrame) -> None:
    """
    Classifica as colunas conforme o percentual de preenchimento.
    """

    imprimir_secao("CLASSIFICAÇÃO DAS COLUNAS POR COMPLETUDE")

    total_linhas = len(dataframe)

    percentual_preenchido = (
        dataframe.notna().sum() / total_linhas
    ) * 100

    alta_completude = percentual_preenchido[percentual_preenchido >= 95]

    media_completude = percentual_preenchido[
        (percentual_preenchido >= 70) & (percentual_preenchido < 95)
    ]

    baixa_completude = percentual_preenchido[percentual_preenchido < 70]

    print("\nVariáveis com alta completude (>= 95%):")
    if alta_completude.empty:
        print("- Nenhuma variável nessa faixa.")
    else:
        for coluna, percentual in alta_completude.sort_values(ascending=False).items():
            print(f"- {coluna}: {percentual:.2f}% preenchido")

    print("\nVariáveis com completude intermediária (70% a 95%):")
    if media_completude.empty:
        print("- Nenhuma variável nessa faixa.")
    else:
        for coluna, percentual in media_completude.sort_values(ascending=False).items():
            print(f"- {coluna}: {percentual:.2f}% preenchido")

    print("\nVariáveis com baixa completude (< 70%):")
    if baixa_completude.empty:
        print("- Nenhuma variável nessa faixa.")
    else:
        for coluna, percentual in baixa_completude.sort_values().items():
            print(f"- {coluna}: {percentual:.2f}% preenchido")


def analisar_valores_nao_informados(dataframe: pd.DataFrame) -> None:
    """
    Analisa categorias usadas para representar informação não informada.
    """

    imprimir_secao("ANÁLISE DE VALORES NÃO INFORMADOS")

    marcadores_nao_informados = [
        "info_suspeito_nao_informada",
        "info_vitima_nao_informada",
        "nao_informado",
        "não informado",
        "N/D",
        "n/d",
        "nan",
        "None",
    ]

    total_linhas = len(dataframe)

    for coluna in dataframe.columns:
        if dataframe[coluna].dtype != "object":
            continue

        serie_texto = dataframe[coluna].astype(str).str.strip()

        quantidade = serie_texto.isin(marcadores_nao_informados).sum()

        if quantidade > 0:
            percentual = (quantidade / total_linhas) * 100
            print(f"- {coluna}: {quantidade} registros ({percentual:.2f}%)")


def analisar_agravantes(dataframe: pd.DataFrame) -> None:
    """
    Analisa apenas registros que possuem agravantes preenchidos.
    """

    coluna = "agravantes"

    if coluna not in dataframe.columns:
        print(f"[AVISO] Coluna não encontrada: {coluna}")
        return

    dados = dataframe.dropna(subset=[coluna])

    percentual = (len(dados) / len(dataframe)) * 100

    imprimir_secao("AGRAVANTES")

    print(
        f"Registros com informação: "
        f"{len(dados)} ({percentual:.2f}%)"
    )

    print("\nDistribuição:")
    print(
        dados[coluna]
        .value_counts(dropna=False)
        .head(20)
    )


def analisar_agravantes_policiais(dataframe: pd.DataFrame) -> None:
    """
    Analisa registros que possuem informação de agravantes policiais.
    """

    coluna = "agravantes_policiais"

    if coluna not in dataframe.columns:
        print(f"[AVISO] Coluna não encontrada: {coluna}")
        return

    dados = dataframe.dropna(subset=[coluna])

    percentual = (len(dados) / len(dataframe)) * 100

    imprimir_secao("AGRAVANTES POLICIAIS")

    print(
        f"Registros com informação: "
        f"{len(dados)} ({percentual:.2f}%)"
    )

    print("\nDistribuição:")
    print(
        dados[coluna]
        .value_counts(dropna=False)
        .head(20)
    )


def executar_analise_completude(dataframe: pd.DataFrame) -> None:
    """
    Executa todas as análises relacionadas à completude dos dados.
    """

    imprimir_secao("INÍCIO DA ANÁLISE DE COMPLETUDE")

    analisar_completude_geral(dataframe)
    classificar_colunas_por_completude(dataframe)
    analisar_valores_nao_informados(dataframe)
    analisar_agravantes(dataframe)
    analisar_agravantes_policiais(dataframe)

    imprimir_secao("FIM DA ANÁLISE DE COMPLETUDE")