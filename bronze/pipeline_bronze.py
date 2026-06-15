# ======================================================================================
# pipeline_bronze.py
# ======================================================================================
# Responsabilidade:
# - Executar o fluxo da camada bronze
# - Transformar XLSX em CSV
# - Executar diagnóstico inicial
# - Executar limpeza estrutural
# - Preparar base analítica
# - Executar análises da base tratada
# - Preparar dados para etapas futuras de modelagem
# ======================================================================================
from Pesquisa_principal.constants import ARQUIVO_OUTPUT_BRONZE, ARQUIVO_OUTPUT_ANALISE_DADOS, ARQUIVO_CSV_FEMINICIDIO_LIMPO, ARQUIVO_CSV_FEMINICIDIO_ANALITICO, ARQUIVO_OUTPUT_PREPARACAO_DADOS
# pyrefly: ignore [missing-import]
from Pesquisa_principal.bronze.utils import OutputTerminalEArquivo

# pyrefly: ignore [missing-import]
from Pesquisa_principal.bronze.ingestao_dados.transformacao_dados_xlsx_csv import transformar_base_feminicidio

# pyrefly: ignore [missing-import]
from Pesquisa_principal.bronze.analise_dados.informacoes_iniciais import diagnostico_inicial

# pyrefly: ignore [missing-import]
from Pesquisa_principal.bronze.base.tratamento_dados.limpeza_dados import limpar_dados_bronze, salvar_dataframe_limpo

# pyrefly: ignore [missing-import]
from Pesquisa_principal.bronze.base.tratamento_dados.preparacao_base_analitica import preparar_base_analitica

# pyrefly: ignore [missing-import]
from Pesquisa_principal.bronze.relatorios_dados.relatorio_analitico import relatorio_preparacao_analitica

# pyrefly: ignore [missing-import]
from Pesquisa_principal.bronze.analise_dados.estatisticas_descritivas import gerar_estatisticas_descritivas

# pyrefly: ignore [missing-import]
from Pesquisa_principal.bronze.analise_dados.analise_exploratoria import executar_analise_exploratoria

# pyrefly: ignore [missing-import]
from Pesquisa_principal.bronze.analise_dados.frequencia import executar_analise_frequencias

# pyrefly: ignore [missing-import]
from Pesquisa_principal.bronze.analise_dados.definir_metodologia import definir_metodologia_analise

# pyrefly: ignore [missing-import]
from Pesquisa_principal.bronze.analise_dados.analise_completude import executar_analise_completude

# pyrefly: ignore [missing-import]
from Pesquisa_principal.bronze.base.tratamento_dados.preparacao_dados import preparar_dados_modelagem

# pyrefly: ignore [missing-import]
from Pesquisa_principal.bronze.base.tratamento_dados.normalizacao_violencias import unificar_agravantes, classificar_letalidade
from Pesquisa_principal.bronze.base.tratamento_dados.normalizacao_violencias import classificar_letalidade
from Pesquisa_principal.bronze.base.tratamento_dados.normalizacao_violencias import criar_indicador_faixa_etaria_suspeito_informada
from Pesquisa_principal.bronze.base.tratamento_dados.normalizacao_violencias import classificar_tipo_violencia_normalizado
from Pesquisa_principal.bronze.base.tratamento_dados.normalizacao_violencias import criar_indicadores_risco_feminicidio
from Pesquisa_principal.constants import COLUNA_ALVO_MODELAGEM
# ============================================================
# para executar a bronze, rode o comando:
# python -m Pesquisa_principal.bronze
# a branch é a main mesmo
# ============================================================
def pipeline_bronze() -> None:
    """
    Executa a camada bronze.

    Fluxo:
    1. Carrega ou transforma a base XLSX em CSV
    2. Executa diagnóstico inicial da base bruta
    3. Executa limpeza estrutural da base
    4. Salva a base limpa
    5. Prepara a base analítica
    6. Salva a base analítica
    7. Gera relatório da preparação analítica
    8. Executa análises descritivas, exploratórias, frequências, metodologia e completude
    9. Prepara X e y para etapas futuras de modelagem
    """

    # ==========================================================================
    # TRATAMENTO INICIAL DOS DADOS
    # ==========================================================================

    with OutputTerminalEArquivo(ARQUIVO_OUTPUT_BRONZE):
        print("=" * 80)
        print("INÍCIO DA PIPELINE BRONZE - TRATAMENTO INICIAL DOS DADOS")
        print("=" * 80)

        dataframe = transformar_base_feminicidio()

        diagnostico_inicial(dataframe)

        dataframe_limpo = limpar_dados_bronze(dataframe)

        salvar_dataframe_limpo(
            dataframe=dataframe_limpo,
            caminho_saida=ARQUIVO_CSV_FEMINICIDIO_LIMPO,
        )

        print("\n" + "=" * 80)
        print("FIM DA PIPELINE BRONZE - TRATAMENTO INICIAL DOS DADOS")
        print("=" * 80)

        print(f"\nOutput bronze salvo em: {ARQUIVO_OUTPUT_BRONZE}")

    # ==========================================================================
    # PREPARAÇÃO E ANÁLISE DA BASE ANALÍTICA
    # ==========================================================================

    with OutputTerminalEArquivo(ARQUIVO_OUTPUT_ANALISE_DADOS):
        print("=" * 80)
        print("INÍCIO DA ANÁLISE DOS DADOS - CAMADA BRONZE")
        print("=" * 80)

        dataframe_analise = preparar_base_analitica(dataframe_limpo)

        dataframe_analise = unificar_agravantes(dataframe_analise)
        dataframe_analise = classificar_letalidade(dataframe_analise)
        dataframe_analise = criar_indicador_faixa_etaria_suspeito_informada(dataframe_analise)
        dataframe_analise = classificar_tipo_violencia_normalizado(dataframe_analise)
        dataframe_analise = criar_indicadores_risco_feminicidio(dataframe_analise)

        executar_analise_exploratoria(dataframe_analise)
        salvar_dataframe_limpo(
            dataframe=dataframe_analise,
            caminho_saida=ARQUIVO_CSV_FEMINICIDIO_ANALITICO,
        )

        relatorio_preparacao_analitica(
            dataframe_original=dataframe_limpo,
            dataframe_final=dataframe_analise,
        )

        print("\n" + "=" * 80)
        print("ESTATÍSTICAS DESCRITIVAS")
        print("=" * 80)

        gerar_estatisticas_descritivas(dataframe_analise)

        print("\n" + "=" * 80)
        print("ANÁLISE EXPLORATÓRIA")
        print("=" * 80)

        executar_analise_exploratoria(dataframe_analise)

        print("\n" + "=" * 80)
        print("ANÁLISE DE FREQUÊNCIAS")
        print("=" * 80)

        executar_analise_frequencias(dataframe_analise)

        print("\n" + "=" * 80)
        print("DEFINIÇÃO METODOLÓGICA")
        print("=" * 80)

        definir_metodologia_analise(dataframe_analise)

        print("\n" + "=" * 80)
        print("ANÁLISE DE COMPLETUDE")
        print("=" * 80)

        executar_analise_completude(dataframe_analise)

        print("\n" + "=" * 80)
        print("FIM DA ANÁLISE DOS DADOS - CAMADA BRONZE")
        print("=" * 80)

        print(f"\nOutput da análise salvo em: {ARQUIVO_OUTPUT_ANALISE_DADOS}")

    # ==========================================================================
    # PREPARAÇÃO PARA MODELAGEM
    # ==========================================================================

    with OutputTerminalEArquivo(ARQUIVO_OUTPUT_PREPARACAO_DADOS):
        print("=" * 80)
        print("INÍCIO DA PREPARAÇÃO DOS DADOS PARA MODELAGEM")
        print("=" * 80)

        X, y = preparar_dados_modelagem(
            dataframe=dataframe_analise,
            coluna_alvo=COLUNA_ALVO_MODELAGEM,
        )

        print("\nFormato final dos dados preparados:")
        print(f"X: {X.shape}")
        print(f"y: {y.shape}")

        print("\n" + "=" * 80)
        print("FIM DA PREPARAÇÃO DOS DADOS PARA MODELAGEM")
        print("=" * 80)

        print(f"\nOutput da preparação salvo em: {ARQUIVO_OUTPUT_PREPARACAO_DADOS}")
