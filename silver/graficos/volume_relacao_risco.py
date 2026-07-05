# ======================================================================================
# volume_relacao_risco.py
# ======================================================================================
# Responsabilidade:
# - Analisar o volume de registros relacionados a risco de morte
# - Verificar a distribuição do target dentro desses registros
# - Gerar gráfico de pizza
# - Gerar gráfico de colunas
# - Exibir quantidade real e porcentagem
# ======================================================================================

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ORDEM_CLASSES_TARGET = [
    "sem_sinal_identificado",
    "risco_baixo",
    "risco_moderado",
    "risco_elevado",
]

def identificar_registros_com_relacao_risco(
    dataframe: pd.DataFrame,
    coluna_relacao_risco: str,
    termo_relacao_risco: str,
) -> pd.Series:
    """
    Identifica registros que possuem relação com risco de morte.

    Por padrão, essa verificação procura o termo informado dentro da coluna
    selecionada. Exemplo:
    coluna_relacao_risco='agravantes_unificados'
    termo_relacao_risco='RISCO DE MORTE'
    """

    if coluna_relacao_risco not in dataframe.columns:
        raise ValueError(
            f"A coluna '{coluna_relacao_risco}' não existe no dataframe."
        )

    serie_coluna = (
        dataframe[coluna_relacao_risco]
        .fillna("")
        .astype(str)
        .str.upper()
        .str.strip()
    )

    termo_normalizado = termo_relacao_risco.upper().strip()

    mascara_risco = serie_coluna.str.contains(
        termo_normalizado,
        regex=False,
        na=False,
    )

    return mascara_risco


def preparar_volume_linhas_risco(
    dataframe: pd.DataFrame,
    mascara_risco: pd.Series,
) -> pd.DataFrame:
    """
    Prepara o volume geral de linhas com e sem relação com risco de morte.
    """
    total_linhas = len(dataframe)
    total_com_risco = int(mascara_risco.sum())
    total_sem_risco = int(total_linhas - total_com_risco)

    percentual_com_risco = (
        total_com_risco / total_linhas * 100
        if total_linhas > 0
        else 0
    )

    percentual_sem_risco = (
        total_sem_risco / total_linhas * 100
        if total_linhas > 0
        else 0
    )

    dataframe_volume = pd.DataFrame(
        {
            "categoria": [
                "Com relação com risco de morte",
                "Sem relação com risco de morte",
            ],
            "quantidade": [
                total_com_risco,
                total_sem_risco,
            ],
            "percentual": [
                percentual_com_risco,
                percentual_sem_risco,
            ],
        }
    )

    return dataframe_volume


def preparar_volume_target_com_risco(
    dataframe: pd.DataFrame,
    mascara_risco: pd.Series,
    coluna_target: str,
) -> pd.DataFrame:
    """
    Prepara a distribuição do target somente entre os registros
    relacionados a risco de morte.
    """

    if coluna_target not in dataframe.columns:
        raise ValueError(
            f"A coluna target '{coluna_target}' não existe no dataframe."
        )

    dataframe_risco = dataframe.loc[
        mascara_risco,
        [
            coluna_target,
        ],
    ].copy()

    dataframe_risco[coluna_target] = (
        dataframe_risco[coluna_target]
        .fillna("nao_informado")
        .astype(str)
    )

    total_com_risco = len(dataframe_risco)

    contagem_target = (
        dataframe_risco[coluna_target]
        .value_counts()
        .reindex(
            ORDEM_CLASSES_TARGET,
            fill_value=0,
        )
    )

    percentual_target = (
        contagem_target / total_com_risco * 100
        if total_com_risco > 0
        else contagem_target * 0
    )

    dataframe_target = pd.DataFrame(
        {
            "classe_target": contagem_target.index,
            "quantidade": contagem_target.values,
            "percentual": percentual_target.values,
        }
    )

    dataframe_target["percentual"] = dataframe_target["percentual"].round(2)

    return dataframe_target


def gerar_grafico_pizza_volume_linhas(
    dataframe_volume: pd.DataFrame,
    caminho_saida: Path,
) -> None:
    """
    Gera gráfico de pizza com quantidade de linhas com e sem relação
    com risco de morte.
    """

    caminho_saida.mkdir(
        parents=True,
        exist_ok=True,
    )

    labels = []

    for _, linha in dataframe_volume.iterrows():
        labels.append(
            f"{linha['categoria']}\n"
            f"{int(linha['quantidade'])} registros "
            f"({linha['percentual']:.1f}%)"
        )

    plt.figure(
        figsize=(
            9,
            7,
        )
    )

    plt.pie(
        dataframe_volume["quantidade"],
        labels=labels,
        startangle=90,
    )

    plt.title(
        "Volume de registros com relação com risco de morte"
    )

    plt.tight_layout()

    caminho_arquivo = (
        caminho_saida
        / "pizza_volume_linhas_relacao_risco_morte.png"
    )

    plt.savefig(
        caminho_arquivo,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Gráfico salvo em: {caminho_arquivo}")


def gerar_grafico_colunas_volume_linhas(
    dataframe_volume: pd.DataFrame,
    caminho_saida: Path,
) -> None:
    """
    Gera gráfico de colunas com quantidade de linhas com e sem relação
    com risco de morte.
    """

    caminho_saida.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(
        figsize=(
            10,
            6,
        )
    )

    plt.bar(
        dataframe_volume["categoria"],
        dataframe_volume["quantidade"],
    )

    plt.title(
        "Quantidade de registros com e sem relação com risco de morte"
    )

    plt.xlabel("Categoria")
    plt.ylabel("Quantidade de registros")

    plt.xticks(
        rotation=20,
        ha="right",
    )

    for indice, linha in dataframe_volume.iterrows():
        texto = (
            f"{int(linha['quantidade'])}\n"
            f"{linha['percentual']:.1f}%"
        )

        plt.text(
            indice,
            linha["quantidade"],
            texto,
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()

    caminho_arquivo = (
        caminho_saida
        / "colunas_volume_linhas_relacao_risco_morte.png"
    )

    plt.savefig(
        caminho_arquivo,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Gráfico salvo em: {caminho_arquivo}")


def gerar_grafico_pizza_target_com_risco(
    dataframe_target: pd.DataFrame,
    caminho_saida: Path,
) -> None:
    """
    Gera gráfico de pizza com a distribuição do target entre registros
    relacionados a risco de morte.
    """

    caminho_saida.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe_filtrado = dataframe_target[
        dataframe_target["quantidade"] > 0
    ].copy()

    if dataframe_filtrado.empty:
        print(
            "Não há registros relacionados a risco de morte "
            "para gerar gráfico de pizza do target."
        )
        return

    labels = []

    for _, linha in dataframe_filtrado.iterrows():
        labels.append(
            f"{linha['classe_target']}\n"
            f"{int(linha['quantidade'])} registros "
            f"({linha['percentual']:.1f}%)"
        )

    plt.figure(
        figsize=(
            9,
            7,
        )
    )

    plt.pie(
        dataframe_filtrado["quantidade"],
        labels=labels,
        startangle=90,
    )

    plt.title(
        "Distribuição do target nos registros com risco de morte"
    )

    plt.tight_layout()

    caminho_arquivo = (
        caminho_saida
        / "pizza_target_relacao_risco_morte.png"
    )

    plt.savefig(
        caminho_arquivo,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Gráfico salvo em: {caminho_arquivo}")


def gerar_grafico_colunas_target_com_risco(
    dataframe_target: pd.DataFrame,
    caminho_saida: Path,
) -> None:
    """
    Gera gráfico de colunas com a distribuição do target entre registros
    relacionados a risco de morte.
    """

    caminho_saida.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(
        figsize=(
            11,
            6,
        )
    )

    plt.bar(
        dataframe_target["classe_target"],
        dataframe_target["quantidade"],
    )

    plt.title(
        "Quantidade de targets nos registros com risco de morte"
    )

    plt.xlabel("Classe do target")
    plt.ylabel("Quantidade de registros")

    plt.xticks(
        rotation=25,
        ha="right",
    )

    for indice, linha in dataframe_target.iterrows():
        texto = (
            f"{int(linha['quantidade'])}\n"
            f"{linha['percentual']:.1f}%"
        )

        plt.text(
            indice,
            linha["quantidade"],
            texto,
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()

    caminho_arquivo = (
        caminho_saida
        / "colunas_target_relacao_risco_morte.png"
    )

    plt.savefig(
        caminho_arquivo,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Gráfico salvo em: {caminho_arquivo}")


def exibir_resumo_volume_relacao_risco(
    dataframe_volume: pd.DataFrame,
    dataframe_target: pd.DataFrame,
    coluna_relacao_risco: str,
    termo_relacao_risco: str,
) -> None:
    """
    Exibe no terminal o resumo da análise.
    """

    print("\n" + "=" * 80)
    print("RESUMO - VOLUME COM RELAÇÃO COM RISCO DE MORTE")
    print("=" * 80)

    print(f"Coluna analisada: {coluna_relacao_risco}")
    print(f"Termo procurado: {termo_relacao_risco}")

    print("\nVolume geral de linhas:")
    for _, linha in dataframe_volume.iterrows():
        print(
            f"- {linha['categoria']}: "
            f"{int(linha['quantidade'])} registros "
            f"({linha['percentual']:.2f}%)"
        )

    print("\nDistribuição do target dentro dos registros com risco de morte:")
    for _, linha in dataframe_target.iterrows():
        print(
            f"- {linha['classe_target']}: "
            f"{int(linha['quantidade'])} registros "
            f"({linha['percentual']:.2f}%)"
        )


def gerar_graficos_volume_relacao_risco(
    dataframe: pd.DataFrame,
    caminho_saida: Path,
    coluna_target: str,
    coluna_relacao_risco: str = "agravantes_unificados",
    termo_relacao_risco: str = "RISCO DE MORTE",
) -> None:
    """
    Executa o diagnóstico completo de volume relacionado a risco de morte.

    Saídas geradas:
    - gráfico de pizza com linhas com/sem risco de morte
    - gráfico de colunas com linhas com/sem risco de morte
    - gráfico de pizza com distribuição do target nos registros com risco de morte
    - gráfico de colunas com distribuição do target nos registros com risco de morte
    """

    print("\n" + "=" * 80)
    print("GERAÇÃO DE GRÁFICOS - VOLUME E RELAÇÃO COM RISCO DE MORTE")
    print("=" * 80)

    caminho_saida.mkdir(
        parents=True,
        exist_ok=True,
    )

    mascara_risco = identificar_registros_com_relacao_risco(
        dataframe=dataframe,
        coluna_relacao_risco=coluna_relacao_risco,
        termo_relacao_risco=termo_relacao_risco,
    )

    dataframe_volume = preparar_volume_linhas_risco(
        dataframe=dataframe,
        mascara_risco=mascara_risco,
    )

    dataframe_target = preparar_volume_target_com_risco(
        dataframe=dataframe,
        mascara_risco=mascara_risco,
        coluna_target=coluna_target,
    )

    exibir_resumo_volume_relacao_risco(
        dataframe_volume=dataframe_volume,
        dataframe_target=dataframe_target,
        coluna_relacao_risco=coluna_relacao_risco,
        termo_relacao_risco=termo_relacao_risco,
    )

    gerar_grafico_pizza_volume_linhas(
        dataframe_volume=dataframe_volume,
        caminho_saida=caminho_saida,
    )

    gerar_grafico_colunas_volume_linhas(
        dataframe_volume=dataframe_volume,
        caminho_saida=caminho_saida,
    )

    gerar_grafico_pizza_target_com_risco(
        dataframe_target=dataframe_target,
        caminho_saida=caminho_saida,
    )

    gerar_grafico_colunas_target_com_risco(
        dataframe_target=dataframe_target,
        caminho_saida=caminho_saida,
    )

    print("\n" + "=" * 80)
    print("GRÁFICOS DE VOLUME E RELAÇÃO COM RISCO DE MORTE FINALIZADOS")
    print("=" * 80)