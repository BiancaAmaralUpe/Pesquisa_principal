# ======================================================================================
# constants.py
# ======================================================================================
# Responsabilidade:
# - Centralizar caminhos, nomes de arquivos e parâmetros globais do projeto
# ======================================================================================

from pathlib import Path

# ======================================================================================
# Diretórios principais
# ======================================================================================

ROOT_DIR = Path(__file__).resolve().parent

BRONZE_DIR = ROOT_DIR / "bronze"
SILVER_DIR = ROOT_DIR / "silver"
GOLD_DIR = ROOT_DIR / "gold"
# ======================================================================================
# Diretórios da camada bronze
# ======================================================================================

BRONZE_BASE_DIR = BRONZE_DIR / "base"
BRONZE_DADOS_DIR = BRONZE_DIR / "dados"

BRONZE_RAW_DIR = BRONZE_DADOS_DIR / "raw"
BRONZE_INTERMEDIARIOS_DIR = BRONZE_DADOS_DIR / "intermediarios_dados"
BRONZE_LIMPEZA_DIR = BRONZE_DADOS_DIR / "limpeza_dados"
BRONZE_NORMALIZACAO_DIR = BRONZE_DADOS_DIR / "normalizacao_dados"

BRONZE_OUTPUTS_DIR = BRONZE_DIR / "outputs"

BRONZE_ANALISE_DADOS_DIR = BRONZE_DIR / "analise_dados"
BRONZE_INGESTAO_DADOS_DIR = BRONZE_DIR / "ingestao_dados"
BRONZE_TRATAMENTO_DADOS_DIR = BRONZE_BASE_DIR / "tratamento_dados"
BRONZE_RELATORIOS_DADOS_DIR = BRONZE_DIR / "relatorios_dados"
# ======================================================================================
# Diretórios das camadas silver e gold
# ======================================================================================

SILVER_OUTPUTS_DIR = SILVER_DIR / "outputs"
GOLD_OUTPUTS_DIR = GOLD_DIR / "outputs"
# ======================================================================================
# Arquivos de entrada - camada bronze
# ======================================================================================

ARQUIVO_XLSX_FEMINICIDIO = (
    BRONZE_RAW_DIR / "2020_Primeiro_Semestre.xlsx"
)

ARQUIVO_CSV_FEMINICIDIO = (
    BRONZE_RAW_DIR / "2020_primeiro_semestre.csv"
)
# ======================================================================================
# Arquivos processados - camada bronze
# ======================================================================================

ARQUIVO_CSV_FEMINICIDIO_LIMPO = (
    BRONZE_LIMPEZA_DIR / "2020_primeiro_semestre_limpo.csv"
)

ARQUIVO_CSV_FEMINICIDIO_ANALITICO = (
    BRONZE_INTERMEDIARIOS_DIR / "2020_primeiro_semestre_analitico.csv"
)

ARQUIVO_CSV_FEMINICIDIO_NORMALIZADO = (
    BRONZE_NORMALIZACAO_DIR / "2020_primeiro_semestre_normalizado.csv"
)

ARQUIVO_CSV_LIMPO_TESTE_TREINO = (
    BRONZE_INTERMEDIARIOS_DIR / "df_limpo_teste_treino.csv"
)

# ======================================================================================
# Outputs textuais - camada bronze
# ======================================================================================

ARQUIVO_OUTPUT_BRONZE = (
    BRONZE_OUTPUTS_DIR / "output_bronze.txt"
)

ARQUIVO_OUTPUT_ANALISE_DADOS = (
    BRONZE_OUTPUTS_DIR / "output_analise.txt"
)

ARQUIVO_OUTPUT_PREPARACAO_DADOS = (
    BRONZE_OUTPUTS_DIR / "output_preparacao_dados.txt"
)
# ======================================================================================
# Parâmetros de qualidade/completude dos dados
# ======================================================================================

LIMITE_BAIXA_AUSENCIA = 5.0
LIMITE_VARIAVEL_CRITICA = 30.0
LIMITE_PERCENTUAL_NULOS_CRITICO = 30.0
LIMITE_PERCENTUAL_CATEGORIA_RARA = 1.0
# ======================================================================================
# Valores usados para padronização de nulos
# ======================================================================================

VALORES_NULOS_PADRAO = [
    "",
    " ",
    "NA",
    "N/A",
    "N/D",
    "NULL",
    "null",
    "None",
    "none",
    "NaN",
    "nan",
]

VALORES_TEXTUAIS_PROBLEMATICOS = [
    "",
    "nan",
    "none",
    "null",
    "n/d",
    "na",
    "n/a",
]

VALORES_NULOS_TEXTUAIS = {
    "",
    "null",
    "nan",
    "none",
    "n/d",
    "nd",
    "na",
    "n.a",
    "não informado",
    "nao informado",
}
# ======================================================================================
# Cabeçalho esperado na base original
# ======================================================================================

COLUNA_CABECALHO_ESPERADA = "Data da denúncia - Ano"

TEXTO_CABECALHO_INVALIDO_EXCEL = (
    "Violência Doméstica e Familiar Contra a Mulher ouViolência Contra a Mulher"
)
# ======================================================================================
# Colunas principais da pesquisa
# ======================================================================================

COLUNA_ESPECIE_VIOLACAO = "especie_violacao"
COLUNA_TIPO_VIOLENCIA_NORMALIZADO = "tipo_violencia_normalizado"

COLUNA_INDICADOR_LETALIDADE = "indicador_letalidade"
COLUNA_SUBTIPO_LETALIDADE = "subtipo_letalidade"

COLUNA_FAIXA_ETARIA_SUSPEITO = "faixa_etaria_suspeito"
COLUNA_SUSPEITO_FAIXA_ETARIA_INFORMADA = "suspeito_faixa_etaria_informada"

VALOR_INFO_SUSPEITO_NAO_INFORMADA = "info_suspeito_nao_informada"
# ======================================================================================
# Colunas categóricas para estatísticas/frequência da base analítica
# ======================================================================================

COLUNAS_CATEGORICAS_INTERESSE = [
    "tipo_violacao",
    "tipo_violencia_normalizado",
    "grupo_vulneravel",
    "especie_violacao",
    "denuncia_emergencial",
    "sexo_vitima",
    "faixa_etaria_vitima",
    "sexo_suspeito",
    "faixa_etaria_suspeito",
    "suspeito_faixa_etaria_informada",
    "relacao_vitima_suspeito",
    "agravantes_unificados",
    "indicador_letalidade",
    "subtipo_letalidade",
]

COLUNAS_FREQUENCIA = [
    "tipo_violacao",
    "tipo_violencia_normalizado",
    "grupo_vulneravel",
    "especie_violacao",
    "cenario_violacao",
    "denuncia_emergencial",
    "sexo_vitima",
    "faixa_etaria_vitima",
    "sexo_suspeito",
    "faixa_etaria_suspeito",
    "suspeito_faixa_etaria_informada",
    "relacao_vitima_suspeito",
    "agravantes_unificados",
    "indicador_letalidade",
    "subtipo_letalidade",
]
# ======================================================================================
# Marcadores operacionais de informação não informada
# ======================================================================================

CATEGORIAS_NAO_INFORMADAS = {
    "faixa_etaria_suspeito": "info_suspeito_nao_informada",
    "faixa_etaria_vitima": "info_vitima_nao_informada",
}

MARCADORES_NULOS_CRITICOS_BRONZE = {
    "faixa_etaria_suspeito": "info_suspeito_nao_informada",
    "faixa_etaria_vitima": "info_vitima_nao_informada",
    "municipio": "municipio_nao_informado",
}
# ======================================================================================
# Regras da base analítica
# ======================================================================================

COLUNAS_GEOGRAFICAS_TEMPORAIS_REMOVER = [
    "uf",
    "municipio",
    "estado",
    "cidade",
    "bairro",
    "data_denuncia_ano",
    "data_denuncia_mes",
    "data_denuncia_dia",
    "canal_atendimento",
]

COLUNAS_RUIDO_ANALITICO = [
    "cenario_violacao",
    "sexo_vitima",
    "sexo_suspeito",
    "relacao_vitima_suspeito",
]

COLUNAS_ENVIESAMENTO_REMOVER = [
    "raca_cor_suspeito",
    "raca_cor_vitima",
]
# ======================================================================================
# Base de treino/modelagem
# ======================================================================================

COLUNA_ALVO_MODELAGEM = "tipo_violencia_normalizado"
COLUNAS_REMOVER_BASE_TREINO = [
    # Colunas de faixa etária removidas por decisão metodológica
    "suspeito_faixa_etaria_informada",
    "faixa_etaria_vitima",
    "faixa_etaria_suspeito",

    # Colunas originais substituídas por agravantes_unificados
    "agravantes",
    "agravantes_policiais",
]

# ======================================================================================
# Marcadores para preenchimento de nulos na preparação de modelagem
# ======================================================================================
MARCADORES_NULOS_MODELAGEM = {
    "motivacao": "motivacao_nao_informada",
    "grau_instrucao_vitima": "instrucao_vitima_nao_informada",
    "faixa_renda_vitima": "renda_vitima_nao_informada",
    "pais_origem_vitima": "pais_origem_vitima_nao_informado",
    "nacionalidade_vitima": "nacionalidade_vitima_nao_informada",
    "deficiencia_vitima": "deficiencia_vitima_nao_informada",
    "grau_instrucao_suspeito": "instrucao_suspeito_nao_informada",
    "faixa_renda_suspeito": "renda_suspeito_nao_informada",
    "pais_origem_suspeito": "pais_origem_suspeito_nao_informado",
    "nacionalidade_suspeito": "nacionalidade_suspeito_nao_informada",
    "deficiencia_suspeito": "deficiencia_suspeito_nao_informada",
}

# ======================================================================================
# Variáveis definidas para exclusão da base analítica/modelagem
# ======================================================================================
VARIAVEIS_EXCLUSAO_BASE_ANALITICA = {
    "uf": (
        "Variável geográfica removida para reduzir risco de overfitting, "
        "alta cardinalidade e enviesamento regional."
    ),
    "municipio": (
        "Variável geográfica com alta cardinalidade e sem aderência direta "
        "ao objetivo atual da modelagem."
    ),
    "data_denuncia_ano": (
        "Variável temporal fora do recorte metodológico atual."
    ),
    "data_denuncia_mes": (
        "Variável temporal fora do recorte metodológico atual."
    ),
    "data_denuncia_dia": (
        "Variável temporal fora do recorte metodológico atual."
    ),
    "canal_atendimento": (
        "Variável relacionada ao meio de entrada da denúncia, e não diretamente "
        "ao fenômeno analisado."
    ),
    "nacionalidade_suspeito": (
        "Variável sensível com baixo potencial analítico para a modelagem atual "
        "e risco de interpretação enviesada."
    ),
    "pais_origem_suspeito": (
        "Variável sensível e sujeita a vieses interpretativos."
    ),
    "raca_cor_suspeito": (
        "Atributo sensível removido da modelagem para reduzir risco de viés."
    ),
    "raca_cor_vitima": (
        "Atributo sensível removido da modelagem para reduzir risco de viés. "
        "Pode ser preservado em análises descritivas, caso faça parte do recorte social da pesquisa."
    ),
}

# ======================================================================================
# Decisões metodológicas da preparação analítica
# ======================================================================================
DECISOES_PREPARACAO_ANALITICA = {
    "uf": {
        "tipo_decisao": "geográfica",
        "motivo": (
            "Variável geográfica removida para reduzir risco de overfitting "
            "e enviesamento regional."
        ),
    },
    "municipio": {
        "tipo_decisao": "geográfica",
        "motivo": (
            "Variável geográfica com alta cardinalidade e sem aderência direta "
            "ao objetivo atual da modelagem."
        ),
    },
    "estado": {
        "tipo_decisao": "geográfica",
        "motivo": "Variável geográfica removida da base analítica.",
    },
    "cidade": {
        "tipo_decisao": "geográfica",
        "motivo": "Variável geográfica removida da base analítica.",
    },
    "bairro": {
        "tipo_decisao": "geográfica",
        "motivo": "Variável geográfica removida da base analítica.",
    },
    "data_denuncia_ano": {
        "tipo_decisao": "temporal",
        "motivo": (
            "Variável temporal removida por não fazer parte do recorte "
            "metodológico atual."
        ),
    },
    "data_denuncia_mes": {
        "tipo_decisao": "temporal",
        "motivo": (
            "Variável temporal removida para evitar dependência temporal "
            "ou sazonalidade artificial na modelagem."
        ),
    },
    "data_denuncia_dia": {
        "tipo_decisao": "temporal",
        "motivo": (
            "Variável temporal removida por não fazer parte do recorte "
            "metodológico atual."
        ),
    },
    "canal_atendimento": {
        "tipo_decisao": "operacional",
        "motivo": (
            "Variável relacionada ao meio de entrada da denúncia, sem uso "
            "previsto na modelagem."
        ),
    },
    "raca_cor_suspeito": {
        "tipo_decisao": "sensível",
        "motivo": (
            "Atributo sensível removido da modelagem para reduzir risco "
            "de enviesamento."
        ),
    },
    "raca_cor_vitima": {
        "tipo_decisao": "sensível",
        "motivo": (
            "Atributo sensível removido da modelagem para reduzir risco "
            "de enviesamento. Pode ser preservado em análises descritivas, "
            "caso faça parte do recorte social da pesquisa."
        ),
    },
}

# ======================================================================================
# Mapeamento do tipo de violência normalizado
# ======================================================================================
MAPEAMENTO_TIPO_VIOLENCIA_NORMALIZADO = {
    # Violência moral
    "CALÚNIA/INJÚRIA/DIFAMAÇÃO": "violencia_moral",
    "EXPOSIÇÃO": "violencia_moral",
    "ASSÉDIO MORAL": "violencia_moral",

    # Violência psicológica
    "AMEAÇA/COAÇÃO": "violencia_psicologica",
    "CONSTRANGIMENTO": "violencia_psicologica",
    "TORTURA PSÍQUICA": "violencia_psicologica",
    "TORTURA PSÍ QUICA": "violencia_psicologica",
    "CRIMES CONTRA A SEGURANÇA PSÍQUICA": "violencia_psicologica",
    "CRIMES CONTRA A SEGURANÇA PSÍ QUICA": "violencia_psicologica",
    "AUTONOMIA DE VONTADE": "violencia_psicologica",
    "INSUBSISTÊNCIA AFETIVA": "violencia_psicologica",

    # Violência física
    "AGRESSÃO/VIAS DE FATO": "violencia_fisica",
    "LESÃO CORPORAL": "violencia_fisica",
    "MAUS TRATOS": "violencia_fisica",
    "CRIMES CONTRA A SEGURANÇA FÍSICA": "violencia_fisica",
    "TORTURA FÍSICA": "violencia_fisica",
    "EXPOSIÇÃO DE RISCO À SAÚDE": "violencia_fisica",
    "TENTATIVA DE FEMINICÍDIO": "violencia_fisica",
    "TENTATIVA DE HOMICÍDIO": "violencia_fisica",
    "FEMINICÍDIO": "violencia_fisica",
    "HOMICÍDIO": "violencia_fisica",

    # Violência sexual
    "LIBERDADE SEXUAL FÍSICA - ESTUPRO": "violencia_sexual",
    "LIBERDADE SEXUAL PSÍQUICA - ASSÉDIO SEXUAL": "violencia_sexual",
    "LIBERDADE SEXUAL PSÍ QUICA - ASSÉDIO SEXUAL": "violencia_sexual",
    "LIBERDADE SEXUAL PSÍQUICA - ABUSO SEXUAL PSÍQUICO": "violencia_sexual",
    "LIBERDADE SEXUAL PSÍ QUICA - ABUSO SEXUAL PSÍQUICO": "violencia_sexual",
    "LIBERDADE SEXUAL FÍSICA - ABUSO SEXUAL FÍSICO": "violencia_sexual",
    "LIBERDADE SEXUAL FÍSICA - EXPLORAÇÃO SEXUAL": "violencia_sexual",
    "DIREITOS DE REPRODUÇÃO": "violencia_sexual",
    "VIOLÊNCIA OBSTÉTRICA": "violencia_sexual",
    "ABORTO": "violencia_sexual",

    # Violência patrimonial
    "VIOLÊNCIA PATRIMONIAL": "violencia_patrimonial",
    "PROPRIEDADE": "violencia_patrimonial",
    "PROPRIEDADE - PATRIMÔNIO MATERIAL": "violencia_patrimonial",
    "CRIMES CONTRA A SEGURANÇA ECONÔMICA": "violencia_patrimonial",
    "INSUBSISTÊNCIA MATERIAL": "violencia_patrimonial",

    # Reclassificações após análise de outras_violencias
    "CÁRCERE PRIVADO": "violencia_psicologica",
    "SEQUESTRO": "violencia_psicologica",
    "INSUBSISTÊNCIA INTELECTUAL": "violencia_psicologica",

    "TRÁFICO NACIONAL DE PESSOAS": "violencia_sexual",
    "TRÁFICO INTERNACIONAL DE PESSOAS": "violencia_sexual",

    "SITUAÇÃO DE RUA/ ABANDONO MATERIAL": "violencia_patrimonial",

}

# ======================================================================================
# Mapeamento de letalidade
# ======================================================================================
MAPEAMENTO_INDICADOR_LETALIDADE = {
    "TENTATIVA DE FEMINICÍDIO": "tentativa_letal",
    "TENTATIVA DE HOMICÍDIO": "tentativa_letal",
    "FEMINICÍDIO": "letal_consumada",
    "HOMICÍDIO": "letal_consumada",
}

MAPEAMENTO_SUBTIPO_LETALIDADE = {
    "TENTATIVA DE FEMINICÍDIO": "tentativa_feminicidio",
    "TENTATIVA DE HOMICÍDIO": "tentativa_homicidio",
    "FEMINICÍDIO": "feminicidio",
    "HOMICÍDIO": "homicidio",
}

ESPECIES_MANTIDAS_COMO_OUTRAS_VIOLENCIAS = [
    "AGRESSÕES QUE VIOLAM O DIREITO A IGUALDADE FORMAL",
    "AGRESSÕES QUE VIOLAM O DIREITO A IGUALDADE MATERIAL",
    "VIOLÊNCIA CONTRA A LIBERDADE DE EXPRESSÃO",
    "FALTA DE ACESSIBILIDADE",
    "CIDADANIA",
    "RACISMO",
    "OUTROS",
]

COLUNAS_MANTER_BASE_TREINO_TESTE = [
    "tipo_violacao",
    "grupo_vulneravel",
    "especie_violacao",
    "motivacao",
    "denuncia_emergencial",
    "agravantes_unificados",
    "sexo_vitima",
    "grau_instrucao_vitima",
    "faixa_renda_vitima",
    "pais_origem_vitima",
    "nacionalidade_vitima",
    "deficiencia_vitima",
    "sexo_suspeito",
    "grau_instrucao_suspeito",
    "faixa_renda_suspeito",
    "pais_origem_suspeito",
    "nacionalidade_suspeito",
    "deficiencia_suspeito",
    "relacao_vitima_suspeito",
    "indicador_letalidade",
    "subtipo_letalidade",
    "tipo_violencia_normalizado",
    "risco_ameaca_morte",
    "risco_violencia_fisica_grave",
    "risco_escalada_agressoes",
    "risco_controle_extremo",
    "risco_separacao_termino",
    "risco_violencia_sexual_associada",
    "qtd_sinais_risco_feminicidio",
    "classificacao_sinais_risco_feminicidio",
]

COLUNAS_EXIGIR_PREENCHIMENTO_TREINO = COLUNAS_MANTER_BASE_TREINO_TESTE

MAPEAMENTO_CORRECAO_ESPECIE_VIOLACAO = {
"TORTURA PSÍ QUICA": "TORTURA PSÍQUICA",
"CRIMES CONTRA A SEGURANÇA PSÍ QUICA": "CRIMES CONTRA A SEGURANÇA PSÍQUICA",
"LIBERDADE SEXUAL PSÍ QUICA - ASSÉDIO SEXUAL": "LIBERDADE SEXUAL PSÍQUICA - ASSÉDIO SEXUAL",
"LIBERDADE SEXUAL PSÍ QUICA - ABUSO SEXUAL PSÍQUICO": "LIBERDADE SEXUAL PSÍQUICA - ABUSO SEXUAL PSÍQUICO",
}

ESPECIES_REMOVER_BASE_TREINO_TESTE = [
"MEMÓRIA E VERDADE",
"CRIMES CONTRA O MEIO AMBIENTE",
"NACIONALIDADE",
"PROPRIEDADE - PATRIMÔNIO GENÉTICO",
"EXPLORAÇÃO DO TRABALHO IDOSO",
"EXPLORAÇÃO DO TRABALHO DA PESSOA COM DEFICIÊNCIA",
]
