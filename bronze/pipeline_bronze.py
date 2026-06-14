# ======================================================================================
# pipeline.py
# ======================================================================================
# Responsabilidade:
# - Executar o fluxo da camada bronze
# - Transformar XLSX em CSV
# - Exibir informações iniciais da base
# ======================================================================================
from Pesquisa_principal.bronze.analise_dados.analise_dados import remover_registros_com_ruido

from Pesquisa_principal.constants import ARQUIVO_OUTPUT_BRONZE, ARQUIVO_OUTPUT_ANALISE_DADOS, ARQUIVO_CSV_FEMINICIDIO_LIMPO, ARQUIVO_CSV_FEMINICIDIO_ANALITICO, ARQUIVO_OUTPUT_PREPARACAO_DADOS

# pyrefly: ignore [missing-import]
from Pesquisa_principal.bronze.base.utils import OutputTerminalEArquivo

# pyrefly: ignore [missing-import]
from Pesquisa_principal.bronze.base.informacoes_iniciais import diagnostico_inicial

# pyrefly: ignore [missing-import]
from Pesquisa_principal.bronze.base.tratamento_de_dados.transformacao_dados_xlsx_csv import transformar_base_feminicidio

# pyrefly: ignore [missing-import]
from Pesquisa_principal.bronze.base.tratamento_de_dados.limpeza_dados import limpar_dados_bronze, salvar_dataframe_limpo

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
from Pesquisa_principal.bronze.analise_dados.analise_dados import remover_colunas_geograficas, remover_registros_com_ruido

# pyrefly: ignore [missing-import]
from Pesquisa_principal.bronze.analise_dados.relatorio_analitico import relatorio_preparacao_analitica

# pyrefly: ignore [missing-import]
from Pesquisa_principal.bronze.base.preparacao_dados import preparar_dados_modelagem

# pyrefly: ignore [missing-import]
from Pesquisa_principal.bronze.analise_dados.analise_dados import remover_variaveis_enviesamento

def pipeline_bronze() -> None:
    """
    Executa a camada bronze.

    Fluxo:
    1. Executa transformação e limpeza estrutural
    2. Salva output_bronze.txt
    3. Executa análises da base limpa
    4. Salva output_analise.txt
    """

    # ==========================================================================
    # TRATAMENTO DOS DADOS
    # ==========================================================================

    with OutputTerminalEArquivo(ARQUIVO_OUTPUT_BRONZE):
        print("=" * 80)
        print("INÍCIO DA PIPELINE BRONZE - TRATAMENTO DOS DADOS")
        print("=" * 80)

        dataframe = transformar_base_feminicidio()

        diagnostico_inicial(dataframe)

        dataframe_limpo = limpar_dados_bronze(dataframe)

        salvar_dataframe_limpo(
            dataframe=dataframe_limpo,
            caminho_saida=ARQUIVO_CSV_FEMINICIDIO_LIMPO,
        )

        print("\n" + "=" * 80)
        print("FIM DA PIPELINE BRONZE - TRATAMENTO DOS DADOS")
        print("=" * 80)

        print(f"\nOutput bronze salvo em: {ARQUIVO_OUTPUT_BRONZE}")

    # ==========================================================================
    # ANÁLISE DOS DADOS
    # ==========================================================================
    
    with OutputTerminalEArquivo(ARQUIVO_OUTPUT_ANALISE_DADOS):
        print("=" * 80)
        print("INÍCIO DA ANÁLISE DOS DADOS")
        print("=" * 80)

        dataframe_analise = remover_colunas_geograficas(dataframe_limpo)
        dataframe_analise = remover_variaveis_enviesamento(dataframe_analise)
        dataframe_analise = remover_registros_com_ruido(dataframe_analise)

        salvar_dataframe_limpo(
            dataframe=dataframe_analise,
            caminho_saida=ARQUIVO_CSV_FEMINICIDIO_ANALITICO,
        )

        relatorio_preparacao_analitica(
            dataframe_original=dataframe_limpo,
            dataframe_final=dataframe_analise,
            colunas_removidas=[
            "uf",
            "municipio",
            "data_denuncia_ano",
            "data_denuncia_mes",
            "data_denuncia_dia",
            "canal_atendimento",
            "denuncia_emergencial",
            ],
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
        print("FIM DA ANÁLISE DOS DADOS")
        print("=" * 80)

        print(f"\nOutput da análise salvo em: {ARQUIVO_OUTPUT_ANALISE_DADOS}")

    with OutputTerminalEArquivo(ARQUIVO_OUTPUT_PREPARACAO_DADOS):
        print("=" * 80)
        print("INÍCIO DA PREPARAÇÃO DOS DADOS")
        print("=" * 80)

        preparar_dados_modelagem(
            dataframe=dataframe_analise,
            coluna_alvo="tipo_violacao",
        )

        print("\n" + "=" * 80)
        print("FIM DA PREPARAÇÃO DOS DADOS")
        print("=" * 80)

        print(f"\nOutput da preparação salvo em: {ARQUIVO_OUTPUT_PREPARACAO_DADOS}")
