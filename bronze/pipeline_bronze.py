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
# imports das libs 

# imports das variaveis constants.

from Pesquisa_principal.constants import COLUNAS_REMOVER_BASE_TREINO
from Pesquisa_principal.constants import ARQUIVO_OUTPUT_BRONZE
from Pesquisa_principal.constants import ARQUIVO_OUTPUT_BASE_ANALITICA
from Pesquisa_principal.constants import ARQUIVO_OUTPUT_PREPARACAO_MODELAGEM
from Pesquisa_principal.constants import ARQUIVO_OUTPUT_ANALISE_DADOS
from Pesquisa_principal.constants import ARQUIVO_CSV_FEMINICIDIO_LIMPO
from Pesquisa_principal.constants import ARQUIVO_CSV_FEMINICIDIO_ANALITICO
from Pesquisa_principal.constants import ARQUIVO_CSV_LIMPO_TESTE_TREINO
from Pesquisa_principal.constants import COLUNA_ALVO_MODELAGEM
from Pesquisa_principal.constants import PASTA_GRAFICOS_ANALISE

from Pesquisa_principal.bronze.utils import OutputTerminalEArquivo

# imports dos modulos de transformacao de dados
from Pesquisa_principal.bronze.ingestao_dados.transformacao_dados_xlsx_csv import transformar_base_feminicidio
# imports dos modulos de relatorios de dados
from Pesquisa_principal.bronze.relatorios_dados.relatorio_analitico import relatorio_preparacao_analitica

# imports dos modulos de analise de dados
from Pesquisa_principal.bronze.analise_dados.estatisticas_descritivas import gerar_estatisticas_descritivas
from Pesquisa_principal.bronze.analise_dados.analise_exploratoria import executar_analise_exploratoria
from Pesquisa_principal.bronze.analise_dados.frequencia import executar_analise_frequencias
from Pesquisa_principal.bronze.analise_dados.definir_metodologia import definir_metodologia_analise
from Pesquisa_principal.bronze.analise_dados.analise_completude import executar_analise_completude
from Pesquisa_principal.bronze.analise_dados.informacoes_iniciais import diagnostico_inicial
from Pesquisa_principal.bronze.analise_dados.distribuicoes import gerar_graficos_distribuicao

# imports dos modulos de tratamento de dados
from Pesquisa_principal.bronze.base.tratamento_dados.preparacao_dados import preparar_dados_modelagem
from Pesquisa_principal.bronze.base.tratamento_dados.limpeza_dados import limpar_dados_bronze
from Pesquisa_principal.bronze.base.tratamento_dados.limpeza_dados import salvar_dataframe_limpo
from Pesquisa_principal.bronze.base.tratamento_dados.normalizacao_violencias import unificar_agravantes
from Pesquisa_principal.bronze.base.tratamento_dados.normalizacao_violencias import classificar_letalidade
from Pesquisa_principal.bronze.base.tratamento_dados.normalizacao_violencias import unificar_agravantes, classificar_letalidade
from Pesquisa_principal.bronze.base.tratamento_dados.normalizacao_violencias import criar_indicador_faixa_etaria_suspeito_informada
from Pesquisa_principal.bronze.base.tratamento_dados.normalizacao_violencias import classificar_tipo_violencia_normalizado
from Pesquisa_principal.bronze.base.tratamento_dados.normalizacao_violencias import criar_indicadores_risco_feminicidio
from Pesquisa_principal.bronze.base.tratamento_dados.normalizacao_violencias import corrigir_especie_violacao
from Pesquisa_principal.bronze.base.tratamento_dados.normalizacao_violencias import criar_componentes_violencia
from Pesquisa_principal.bronze.base.tratamento_dados.definicao_target import definir_target_modelagem
from Pesquisa_principal.bronze.base.tratamento_dados.preparacao_base_analitica import preparar_base_analitica

# ============================================================
# para executar a bronze, rode o comando:
# python -m Pesquisa_principal.bronze
# a branch é a main mesmo
# ============================================================
def pipeline_bronze() -> None:
    """
    Executa a camada bronze.

    Fluxo:
    1. Executa ingestão e limpeza inicial
    2. Prepara a base analítica
    3. Cria variáveis derivadas
    4. Prepara a base de treino e teste
    5. Executa análises descritivas e exploratórias
    """

    # ==========================================================================
    # 1. TRATAMENTO INICIAL DOS DADOS
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
    # 2. PREPARAÇÃO DA BASE ANALÍTICA
    # ==========================================================================

    with OutputTerminalEArquivo(ARQUIVO_OUTPUT_BASE_ANALITICA):
        print("=" * 80)
        print("INÍCIO DA PREPARAÇÃO DA BASE ANALÍTICA - CAMADA BRONZE")
        print("=" * 80)

        dataframe_analise = preparar_base_analitica(dataframe_limpo)

        dataframe_analise = corrigir_especie_violacao(dataframe_analise)
        dataframe_analise = unificar_agravantes(dataframe_analise)
        dataframe_analise = classificar_letalidade(dataframe_analise)

        dataframe_analise = criar_indicador_faixa_etaria_suspeito_informada(
            dataframe_analise
        )

        dataframe_analise = classificar_tipo_violencia_normalizado(
            dataframe_analise
        )

        dataframe_analise = criar_indicadores_risco_feminicidio(
            dataframe_analise
        )

        dataframe_analise = criar_componentes_violencia(dataframe_analise)

        salvar_dataframe_limpo(
            dataframe=dataframe_analise,
            caminho_saida=ARQUIVO_CSV_FEMINICIDIO_ANALITICO,
        )

        relatorio_preparacao_analitica(
            dataframe_original=dataframe_limpo,
            dataframe_final=dataframe_analise,
        )

        print("\n" + "=" * 80)
        print("FIM DA PREPARAÇÃO DA BASE ANALÍTICA - CAMADA BRONZE")
        print("=" * 80)

        print(f"\nOutput da base analítica salvo em: {ARQUIVO_OUTPUT_BASE_ANALITICA}")

    # ==========================================================================
    # 3. PREPARAÇÃO DA BASE DE MODELAGEM
    # ==========================================================================

    with OutputTerminalEArquivo(ARQUIVO_OUTPUT_PREPARACAO_MODELAGEM):
        print("=" * 80)
        print("INÍCIO DA PREPARAÇÃO DA BASE DE MODELAGEM")
        print("=" * 80)

        definir_target_modelagem(
            dataframe=dataframe_analise,
            coluna_alvo=COLUNA_ALVO_MODELAGEM,
            colunas_removidas_por_vazamento=COLUNAS_REMOVER_BASE_TREINO,
        )
    
        X, y, dataframe_treino = preparar_dados_modelagem(
            dataframe=dataframe_analise,
            coluna_alvo=COLUNA_ALVO_MODELAGEM,
            caminho_saida=ARQUIVO_CSV_LIMPO_TESTE_TREINO,
        )

        print("\n" + "=" * 80)
        print("FIM DA PREPARAÇÃO DA BASE DE MODELAGEM")
        print("=" * 80)

        print(
            f"\nOutput da preparação de modelagem salvo em: "
            f"{ARQUIVO_OUTPUT_PREPARACAO_MODELAGEM}"
        )

    # ==========================================================================
    # 4. ANÁLISE DOS DADOS
    # ==========================================================================

    with OutputTerminalEArquivo(ARQUIVO_OUTPUT_ANALISE_DADOS):
        print("=" * 80)
        print("INÍCIO DA ANÁLISE DOS DADOS - CAMADA BRONZE")
        print("=" * 80)

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
        print("ANÁLISE DE COMPLETUDE")
        print("=" * 80)
    
        executar_analise_completude(dataframe_analise)
    
        print("\n" + "=" * 80)
        print("DEFINIÇÃO METODOLÓGICA")
        print("=" * 80)
    
        definir_metodologia_analise(dataframe_analise)
    
        print("\n" + "=" * 80)
        print("GRÁFICOS DE DISTRIBUIÇÃO")
        print("=" * 80)
    
        gerar_graficos_distribuicao(
            dataframe=dataframe_analise,
            caminho_saida=str(PASTA_GRAFICOS_ANALISE),
        )
    
        print("\n" + "=" * 80)
        print("FIM DA ANÁLISE DOS DADOS - CAMADA BRONZE")
        print("=" * 80)
    
        print(f"\nOutput da análise salvo em: {ARQUIVO_OUTPUT_ANALISE_DADOS}")