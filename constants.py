# ======================================================================================
# constants.py
# ======================================================================================
# Responsabilidade:
# - Centralizar caminhos e nomes de arquivos usados no projeto
# ======================================================================================

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent

BRONZE_DIR = ROOT_DIR / "bronze"

DIRETORIO_DADOS_FEMINICIDIO = BRONZE_DIR / "dados_feminicidio"

ANALISE_DADOS_DIR = BRONZE_DIR / "analise_dados"


ARQUIVO_XLSX_FEMINICIDIO = (
    DIRETORIO_DADOS_FEMINICIDIO / "2020_primeiro_semestre.xlsx"
)

ARQUIVO_CSV_FEMINICIDIO = (
    DIRETORIO_DADOS_FEMINICIDIO / "2020_primeiro_semestre.csv"
)

ARQUIVO_CSV_FEMINICIDIO_LIMPO = (
    DIRETORIO_DADOS_FEMINICIDIO / "2020_primeiro_semestre_limpo.csv"
)

ARQUIVO_OUTPUT_BRONZE = (
    DIRETORIO_DADOS_FEMINICIDIO / "output_bronze.txt"
)

ARQUIVO_OUTPUT_ANALISE_DADOS = (
    ANALISE_DADOS_DIR / "output_analise.txt"
)

ARQUIVO_CSV_FEMINICIDIO_ANALITICO = (
    "Pesquisa_principal/bronze/output/"
    "2020_primeiro_semestre_analitico.csv"
)

ARQUIVO_OUTPUT_PREPARACAO_DADOS = (
    ROOT_DIR / "bronze" / "base" / "output_preparacao_dados.txt"
)

