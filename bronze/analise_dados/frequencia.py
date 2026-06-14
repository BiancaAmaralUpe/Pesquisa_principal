# ======================================================================================
# frequencia_variaveis.py
# ======================================================================================
# Responsabilidade:
# - Analisar distribuição das variáveis categóricas
# - Identificar categorias mais frequentes
# - Identificar categorias raras
# ======================================================================================

import pandas as pd

def analisar_frequencia_coluna(
    dataframe: pd.DataFrame,
    coluna: str,
    top_n: int = 10,
) -> None:
    """
    Exibe as categorias mais frequentes de uma coluna.
    """

    if coluna not in dataframe.columns:
        print(f"[AVISO] Coluna deletada: {coluna}")
        return

    print("\n" + "=" * 80)
    print(f"FREQUÊNCIA - {coluna.upper()}")
    print("=" * 80)

    frequencia = dataframe[coluna].value_counts(dropna=False)

    percentual = (
        dataframe[coluna]
        .value_counts(dropna=False, normalize=True)
        .mul(100)
        .round(2)
    )

    resultado = pd.DataFrame({
        "quantidade": frequencia,
        "percentual (%)": percentual,
    })

    print(resultado.head(top_n))

def executar_analise_frequencias(
    dataframe: pd.DataFrame,
) -> None:
    """
    Executa análises de frequência das principais variáveis.
    """

    colunas_analisar = [
        "tipo_violacao",
        "grupo_vulneravel",
        "especie_violacao",
        "cenario_violacao",
        "denuncia_emergencial",
        "sexo_vitima",
        "faixa_etaria_vitima",
        "sexo_suspeito",
        "faixa_etaria_suspeito",
        "relacao_vitima_suspeito",
    ]

    for coluna in colunas_analisar:
        if coluna in dataframe.columns:
            analisar_frequencia_coluna(
                dataframe=dataframe,
                coluna=coluna,
            )
            
