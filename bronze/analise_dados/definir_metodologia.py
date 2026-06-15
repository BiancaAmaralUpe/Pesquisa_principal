# ======================================================================================
# definir_metodologia.py
# ======================================================================================
# Responsabilidade:
# - Registrar decisões metodológicas a partir da análise exploratória
# - Classificar variáveis por qualidade/completude
# - Documentar regras de exclusão da base analítica
# - Apoiar a próxima etapa de preparação dos dados
# ======================================================================================
import pandas as pd
from Pesquisa_principal.constants import LIMITE_BAIXA_AUSENCIA, LIMITE_VARIAVEL_CRITICA, VARIAVEIS_EXCLUSAO_BASE_ANALITICA

def imprimir_secao(titulo: str) -> None:
    """
    Imprime uma seção formatada no terminal/output.
    """

    print("\n" + "=" * 80)
    print(titulo)
    print("=" * 80)

def validar_dataframe(dataframe: pd.DataFrame) -> bool:
    """
    Verifica se o DataFrame possui dados para análise metodológica.
    """

    if dataframe.empty:
        print("[AVISO] DataFrame vazio. Definição metodológica não executada.")
        return False

    if dataframe.shape[1] == 0:
        print("[AVISO] DataFrame sem colunas. Definição metodológica não executada.")
        return False

    return True

def classificar_colunas_por_completude(dataframe: pd.DataFrame) -> None:
    """
    Classifica as colunas de acordo com o percentual de valores ausentes.
    """

    imprimir_secao("CLASSIFICAÇÃO METODOLÓGICA DAS VARIÁVEIS POR COMPLETUDE")

    total_linhas = len(dataframe)

    percentual_nulos = (
        dataframe.isna().sum() / total_linhas
    ) * 100

    baixo_risco = percentual_nulos[
        percentual_nulos < LIMITE_BAIXA_AUSENCIA
    ]

    atencao = percentual_nulos[
        (percentual_nulos >= LIMITE_BAIXA_AUSENCIA)
        & (percentual_nulos < LIMITE_VARIAVEL_CRITICA)
    ]

    criticas = percentual_nulos[
        percentual_nulos >= LIMITE_VARIAVEL_CRITICA
    ]

    print(f"\nVariáveis com baixa ausência (< {LIMITE_BAIXA_AUSENCIA:.0f}%):")
    if baixo_risco.empty:
        print("- Nenhuma variável nessa faixa.")
    else:
        for coluna, percentual in baixo_risco.sort_values().items():
            print(f"- {coluna}: {percentual:.2f}% de ausência")

    print(
        f"\nVariáveis que exigem atenção "
        f"({LIMITE_BAIXA_AUSENCIA:.0f}% a {LIMITE_VARIAVEL_CRITICA:.0f}%):"
    )
    if atencao.empty:
        print("- Nenhuma variável nessa faixa.")
    else:
        for coluna, percentual in atencao.sort_values(ascending=False).items():
            print(f"- {coluna}: {percentual:.2f}% de ausência")

    print(f"\nVariáveis críticas (>= {LIMITE_VARIAVEL_CRITICA:.0f}%):")
    if criticas.empty:
        print("- Nenhuma variável crítica.")
    else:
        for coluna, percentual in criticas.sort_values(ascending=False).items():
            print(f"- {coluna}: {percentual:.2f}% de ausência")

def registrar_observacoes_metodologicas() -> None:
    """
    Registra observações metodológicas preliminares.

    Essas observações documentam decisões operacionais tomadas na preparação
    da base analítica. Elas não representam conclusões finais sobre o fenômeno.
    """

    imprimir_secao("OBSERVAÇÕES METODOLÓGICAS PRELIMINARES")

    observacoes = [
        (
            "Valores como 'N/D', 'NULL', 'NAN', 'None' e campos vazios foram "
            "tratados como ausência de informação para padronizar a representação "
            "de dados não disponíveis na base."
        ),
        (
            "A categoria 'info_suspeito_nao_informada' foi mantida como marcador "
            "operacional para registros em que a faixa etária do suspeito não estava "
            "disponível."
        ),
        (
            "A categoria 'info_suspeito_nao_informada' não deve ser interpretada "
            "como característica social, demográfica ou comportamental do suspeito."
        ),
        (
            "A categoria 'info_vitima_nao_informada' foi mantida como marcador "
            "operacional para registros sem faixa etária informada da vítima."
        ),
        (
            "As variáveis geográficas foram removidas da base analítica porque o "
            "objetivo atual do estudo não envolve análise espacial."
        ),
        (
            "A remoção das variáveis geográficas busca reduzir risco de overfitting, "
            "alta cardinalidade e enviesamento regional na etapa de modelagem."
        ),
        (
            "As variáveis temporais derivadas da data da denúncia foram removidas "
            "por não fazerem parte do recorte metodológico atual."
        ),
        (
            "Registros com ausência em 'cenario_violacao', 'sexo_vitima', "
            "'sexo_suspeito' e 'relacao_vitima_suspeito' foram tratados como ruído "
            "analítico e removidos da base de análise."
        ),
        (
            "Variáveis com alto percentual de ausência foram marcadas para avaliação "
            "antes de qualquer uso em score, análise inferencial ou modelagem preditiva."
        ),
        (
            "A coluna 'agravantes_policiais' apresentou percentual elevado de ausência, "
            "mas foi mantida para análise específica por possuir significado analítico "
            "nos registros preenchidos."
        ),
        (
            "Variáveis sociodemográficas sensíveis devem ser analisadas com cautela. "
            "Elas podem ser relevantes para análise social descritiva, mas podem ser "
            "removidas da modelagem para reduzir risco de viés."
        ),
    ]

    for observacao in observacoes:
        print(f"- {observacao}")

def registrar_variaveis_excluidas(dataframe: pd.DataFrame) -> None:
    """
    Registra variáveis definidas para exclusão da base analítica/modelagem.

    A exclusão não significa remoção da camada bronze original.
    """

    imprimir_secao("VARIÁVEIS DEFINIDAS PARA EXCLUSÃO DA BASE ANALÍTICA")

    print("\nAs variáveis abaixo foram classificadas para exclusão:")

    for coluna, motivo in VARIAVEIS_EXCLUSAO_BASE_ANALITICA.items():
        status = "encontrada na base" if coluna in dataframe.columns else "não encontrada na base atual"

        print(f"\n- {coluna}")
        print(f"  Status: {status}")
        print(f"  Motivo: {motivo}")

    print(
        "\nObservação: a exclusão é aplicada na base analítica/modelagem, "
        "preservando a camada bronze como registro rastreável dos dados tratados."
    )

def registrar_recomendacoes_para_proxima_etapa() -> None:
    """
    Registra recomendações para a próxima etapa da pipeline.
    """

    imprimir_secao("RECOMENDAÇÕES PARA A PRÓXIMA ETAPA")

    recomendacoes = [
        (
            "Preservar a base bronze tratada como camada rastreável, sem aplicar "
            "balanceamento ou transformações irreversíveis."
        ),
        (
            "Aplicar remoções metodológicas apenas na base analítica usada para "
            "estatística, score ou modelagem."
        ),
        (
            "Avaliar variáveis críticas de completude antes de incluí-las em modelos "
            "preditivos."
        ),
        (
            "Documentar separadamente quais variáveis foram removidas por baixa qualidade, "
            "por alto percentual de nulos, por alta cardinalidade ou por risco de viés."
        ),
        (
            "Manter variáveis sensíveis disponíveis para análise descritiva quando forem "
            "relevantes para o fenômeno social estudado, mas evitar seu uso direto na "
            "modelagem preditiva sem justificativa metodológica."
        ),
    ]

    for recomendacao in recomendacoes:
        print(f"- {recomendacao}")

def definir_metodologia_analise(dataframe: pd.DataFrame) -> None:
    """
    Executa a definição metodológica inicial com base na qualidade da base.
    """

    if not validar_dataframe(dataframe):
        return

    classificar_colunas_por_completude(dataframe)
    registrar_observacoes_metodologicas()
    registrar_variaveis_excluidas(dataframe)
    registrar_recomendacoes_para_proxima_etapa()