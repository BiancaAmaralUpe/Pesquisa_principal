# ======================================================================================
# balanceamento_base.py
# ======================================================================================
# Responsabilidade:
# - Verificar se a base está balanceada ou desbalanceada
# - Gerar gráficos de distribuição do target
# - Comparar distribuição da base completa, treino e teste
# - Exibir quantidade real e porcentagem por classe
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


def calcular_distribuicao_target(
    y: pd.Series,
    nome_base: str,
) -> pd.DataFrame:
    """
    Calcula a quantidade e o percentual de cada classe do target.
    """

    total_registros = len(y)

    contagem = (
        y.value_counts()
        .reindex(
            ORDEM_CLASSES_TARGET,
            fill_value=0,
        )
    )

    percentual = (
        contagem / total_registros * 100
        if total_registros > 0
        else contagem * 0
    )

    dataframe_distribuicao = pd.DataFrame(
        {
            "base": nome_base,
            "classe_target": contagem.index,
            "quantidade": contagem.values,
            "percentual": percentual.values,
        }
    )

    dataframe_distribuicao["percentual"] = (
        dataframe_distribuicao["percentual"].round(2)
    )

    return dataframe_distribuicao


def exibir_resumo_balanceamento(
    dataframe_distribuicao: pd.DataFrame,
    nome_base: str,
) -> None:
    """
    Exibe no terminal o resumo do balanceamento da base.
    """

    print("\n" + "=" * 80)
    print(f"RESUMO DE BALANCEAMENTO - {nome_base.upper()}")
    print("=" * 80)

    total_registros = dataframe_distribuicao["quantidade"].sum()

    print(f"Total de registros: {total_registros}")

    for _, linha in dataframe_distribuicao.iterrows():
        print(
            f"- {linha['classe_target']}: "
            f"{int(linha['quantidade'])} registros "
            f"({linha['percentual']:.2f}%)"
        )

    maior_classe = dataframe_distribuicao["quantidade"].max()
    menor_classe = dataframe_distribuicao[
        dataframe_distribuicao["quantidade"] > 0
    ]["quantidade"].min()

    razao_desbalanceamento = (
        maior_classe / menor_classe
        if menor_classe > 0
        else 0
    )

    print(
        f"\nRazão de desbalanceamento "
        f"maior classe / menor classe: {razao_desbalanceamento:.2f}"
    )

    if razao_desbalanceamento <= 1.5:
        print("Interpretação: base aproximadamente balanceada.")
    elif razao_desbalanceamento <= 5:
        print("Interpretação: base moderadamente desbalanceada.")
    else:
        print("Interpretação: base fortemente desbalanceada.")


def gerar_grafico_colunas_balanceamento(
    dataframe_distribuicao: pd.DataFrame,
    caminho_saida: Path,
    nome_arquivo: str,
    titulo: str,
) -> None:
    """
    Gera gráfico de colunas com quantidade e porcentagem por classe.
    """

    caminho_saida.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(
        figsize=(
            12,
            6,
        )
    )

    plt.bar(
        dataframe_distribuicao["classe_target"],
        dataframe_distribuicao["quantidade"],
    )

    plt.title(titulo)
    plt.xlabel("Classe do target")
    plt.ylabel("Quantidade de registros")

    plt.xticks(
        rotation=25,
        ha="right",
    )

    for indice, linha in dataframe_distribuicao.iterrows():
        texto = (
            f"{int(linha['quantidade'])}\n"
            f"{linha['percentual']:.2f}%"
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

    caminho_arquivo = caminho_saida / nome_arquivo

    plt.savefig(
        caminho_arquivo,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Gráfico salvo em: {caminho_arquivo}")


def gerar_grafico_pizza_balanceamento(
    dataframe_distribuicao: pd.DataFrame,
    caminho_saida: Path,
    nome_arquivo: str,
    titulo: str,
) -> None:
    """
    Gera gráfico de pizza com quantidade e porcentagem por classe.
    """

    caminho_saida.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe_filtrado = dataframe_distribuicao[
        dataframe_distribuicao["quantidade"] > 0
    ].copy()

    labels = []

    for _, linha in dataframe_filtrado.iterrows():
        labels.append(
            f"{linha['classe_target']}\n"
            f"{int(linha['quantidade'])} registros "
            f"({linha['percentual']:.2f}%)"
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

    plt.title(titulo)

    plt.tight_layout()

    caminho_arquivo = caminho_saida / nome_arquivo

    plt.savefig(
        caminho_arquivo,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Gráfico salvo em: {caminho_arquivo}")


def gerar_grafico_comparativo_percentual(
    dataframe_distribuicoes: pd.DataFrame,
    caminho_saida: Path,
) -> None:
    """
    Gera gráfico comparativo percentual entre base completa, treino e teste.
    """

    caminho_saida.mkdir(
        parents=True,
        exist_ok=True,
    )

    tabela_percentual = dataframe_distribuicoes.pivot(
        index="classe_target",
        columns="base",
        values="percentual",
    )

    tabela_percentual = tabela_percentual.reindex(
        ORDEM_CLASSES_TARGET
    )

    plt.figure(
        figsize=(
            12,
            6,
        )
    )

    ax = tabela_percentual.plot(
        kind="bar",
        figsize=(
            12,
            6,
        ),
    )

    plt.title("Comparativo percentual do target: base completa, treino e teste")
    plt.xlabel("Classe do target")
    plt.ylabel("Percentual (%)")

    plt.xticks(
        rotation=25,
        ha="right",
    )

    plt.legend(
        title="Base",
    )

    for container in ax.containers:
        ax.bar_label(
            container,
            fmt="%.2f%%",
            fontsize=8,
            padding=3,
        )

    plt.tight_layout()

    caminho_arquivo = (
        caminho_saida
        / "balanceamento_target_comparativo_base_treino_teste.png"
    )

    plt.savefig(
        caminho_arquivo,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Gráfico salvo em: {caminho_arquivo}")


def gerar_graficos_balanceamento_base(
    y: pd.Series,
    caminho_saida: Path,
    nome_coluna_target: str,
    y_train: pd.Series | None = None,
    y_test: pd.Series | None = None,
) -> None:
    """
    Gera gráficos para avaliar se o target está balanceado.

    Saídas:
    - gráfico de colunas da base completa
    - gráfico de pizza da base completa
    - gráfico comparativo percentual entre base, treino e teste
    """

    print("\n" + "=" * 80)
    print("GERAÇÃO DE GRÁFICOS DE BALANCEAMENTO DA BASE")
    print("=" * 80)

    print(f"Target analisado: {nome_coluna_target}")

    caminho_saida.mkdir(
        parents=True,
        exist_ok=True,
    )

    distribuicao_base_completa = calcular_distribuicao_target(
        y=y,
        nome_base="Base completa",
    )

    exibir_resumo_balanceamento(
        dataframe_distribuicao=distribuicao_base_completa,
        nome_base="Base completa",
    )

    gerar_grafico_colunas_balanceamento(
        dataframe_distribuicao=distribuicao_base_completa,
        caminho_saida=caminho_saida,
        nome_arquivo="balanceamento_target_colunas_base_completa.png",
        titulo="Balanceamento do target - Base completa",
    )

    gerar_grafico_pizza_balanceamento(
        dataframe_distribuicao=distribuicao_base_completa,
        caminho_saida=caminho_saida,
        nome_arquivo="balanceamento_target_pizza_base_completa.png",
        titulo="Distribuição percentual do target - Base completa",
    )

    distribuicoes = [
        distribuicao_base_completa,
    ]

    if y_train is not None:
        distribuicao_treino = calcular_distribuicao_target(
            y=y_train,
            nome_base="Treino",
        )

        exibir_resumo_balanceamento(
            dataframe_distribuicao=distribuicao_treino,
            nome_base="Treino",
        )

        distribuicoes.append(distribuicao_treino)

    if y_test is not None:
        distribuicao_teste = calcular_distribuicao_target(
            y=y_test,
            nome_base="Teste",
        )

        exibir_resumo_balanceamento(
            dataframe_distribuicao=distribuicao_teste,
            nome_base="Teste",
        )

        distribuicoes.append(distribuicao_teste)

    if y_train is not None and y_test is not None:
        dataframe_distribuicoes = pd.concat(
            distribuicoes,
            ignore_index=True,
        )

        gerar_grafico_comparativo_percentual(
            dataframe_distribuicoes=dataframe_distribuicoes,
            caminho_saida=caminho_saida,
        )

    print("\n" + "=" * 80)
    print("GRÁFICOS DE BALANCEAMENTO FINALIZADOS")
    print("=" * 80)