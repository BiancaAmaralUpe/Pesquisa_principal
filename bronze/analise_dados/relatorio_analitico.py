# ======================================================================================
# relatorio_analitico.py
# ======================================================================================
# Responsabilidade:
# - Documentar decisões da preparação analítica
# - Registrar colunas removidas
# - Registrar motivos metodológicos
# - Validar estado final da base
# ======================================================================================

import pandas as pd

def relatorio_preparacao_analitica(
    dataframe_original: pd.DataFrame,
    dataframe_final: pd.DataFrame,
    colunas_removidas: list[str],
) -> None:
    """
    Gera relatório metodológico da preparação analítica.
    """

    print("\n" + "=" * 80)
    print("PREPARAÇÃO ANALÍTICA DA BASE")
    print("=" * 80)

    print("\nCOLUNAS REMOVIDAS:")

    motivos = {
        "uf": (
            "geográfica",
            "Risco de overfitting e enviesamento regional."
        ),
        "municipio": (
            "geográfica",
            "Alta cardinalidade e enviesamento regional."
        ),
        "estado": (
            "geográfica",
            "Variável geográfica removida."
        ),
        "cidade": (
            "geográfica",
            "Variável geográfica removida."
        ),
        "bairro": (
            "geográfica",
            "Variável geográfica removida."
        ),
        "data_denuncia_ano": (
            "temporal",
            "Variável temporal removida para evitar dependência temporal."
        ),
        "data_denuncia_mes": (
            "temporal",
            "Variável temporal removida para evitar sazonalidade artificial."
        ),
        "data_denuncia_dia": (
            "temporal",
            "Variável temporal removida."
        ),
        "canal_atendimento": (
            "ruído",
            "Baixo potencial analítico para modelagem."
        ),
        "denuncia_emergencial": (
            "ruído",
            "Variável operacional."
        ),
    }

    for coluna in colunas_removidas:

        tipo, motivo = motivos.get(
            coluna,
            ("não definido", "Motivo não documentado.")
        )

        print(f"\n- Coluna: {coluna}")
        print(f"  Tipo de decisão: {tipo}")
        print(f"  Motivo: {motivo}")

    print("\n" + "-" * 80)
    print("VALIDAÇÃO FINAL DA BASE")
    print("-" * 80)

    print(f"Linhas finais: {dataframe_final.shape[0]}")
    print(f"Colunas finais: {dataframe_final.shape[1]}")

    print("\nColunas restantes:")
    for coluna in dataframe_final.columns:
        print(f"- {coluna}")

    print("\nNulos restantes por coluna:")

    nulos = dataframe_final.isna().sum()

    nulos = nulos[nulos > 0]

    if nulos.empty:
        print("Nenhum valor nulo restante.")
    else:
        print(nulos)

    print("\nBase pronta para análise/modelagem.")