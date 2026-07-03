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
    ROOT_DIR / "bronze" / "dados" / "intermediarios_dados" / "df_limpo_teste_treino.csv"
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

ARQUIVO_OUTPUT_BASE_ANALITICA = (
    BRONZE_OUTPUTS_DIR / "output_base_analitica.txt"
)

ARQUIVO_OUTPUT_PREPARACAO_MODELAGEM = (
    BRONZE_OUTPUTS_DIR / "output_preparacao_modelagem.txt"
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

ARQUIVO_CSV_LIMPO_TESTE_TREINO = (
    BRONZE_INTERMEDIARIOS_DIR / "df_limpo_teste_treino.csv"
)

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

    # Componentes sobrepostos de violência
    "componente_fisico",
    "componente_psicologico",
    "componente_sexual",
    "componente_patrimonial",
    "componente_moral",
    "componente_letal",
]

COLUNAS_REMOVER_BASE_TREINO = [
    # Colunas de faixa etária removidas por decisão metodológica
    "suspeito_faixa_etaria_informada",
    "faixa_etaria_vitima",
    "faixa_etaria_suspeito",

    # Colunas originais substituídas por agravantes_unificados
    "agravantes",
    "agravantes_policiais",
]

# Usar somente para validação mínima, se necessário.
# Não usar esta lista para remover nulos de todas as colunas da base.
COLUNAS_EXIGIR_PREENCHIMENTO_TREINO = [
    "tipo_violencia_normalizado",
    "tipo_violacao",
    "grupo_vulneravel",
    "especie_violacao",
    "denuncia_emergencial",
    "sexo_vitima",
    "sexo_suspeito",
    "relacao_vitima_suspeito",
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

# ======================================================================================
# Componentes sobrepostos de violência
# ======================================================================================

COMPONENTES_VIOLENCIA = {
    "TORTURA FÍSICA": {
        "componente_fisico": True,
        "componente_psicologico": True,
    },
    "TENTATIVA DE FEMINICÍDIO": {
        "componente_fisico": True,
        "componente_letal": True,
    },
    "TENTATIVA DE HOMICÍDIO": {
        "componente_fisico": True,
        "componente_letal": True,
    },
    "FEMINICÍDIO": {
        "componente_fisico": True,
        "componente_letal": True,
    },
    "HOMICÍDIO": {
        "componente_fisico": True,
        "componente_letal": True,
    },
    "LIBERDADE SEXUAL PSÍQUICA - ASSÉDIO SEXUAL": {
        "componente_sexual": True,
        "componente_psicologico": True,
    },
    "LIBERDADE SEXUAL PSÍQUICA - ABUSO SEXUAL PSÍQUICO": {
        "componente_sexual": True,
        "componente_psicologico": True,
    },
}

# ======================================================================================
# Target da modelagem
# ======================================================================================

COLUNA_ALVO_MODELAGEM = "classificacao_sinais_risco_feminicidio"

# ======================================================================================
# Colunas removidas da base de treino
# ======================================================================================

COLUNAS_REMOVER_BASE_TREINO = [
    "qtd_sinais_risco_feminicidio",
    "risco_ameaca_morte",
    "risco_violencia_fisica_grave",
    "risco_escalada_agressoes",
    "risco_controle_extremo",
    "risco_separacao_termino",
    "risco_violencia_sexual_associada",
]

# ======================================================================================
# Pasta de gráficos
# ======================================================================================
PASTA_GRAFICOS_ANALISE = ROOT_DIR / "bronze" / "graficos" / "analise_dados"

# ======================================================================================
# Diretórios da camada Silver
# ======================================================================================

SILVER_DIR = ROOT_DIR / "silver"

SILVER_OUTPUTS_DIR = SILVER_DIR / "outputs"

SILVER_GRAFICOS_DIR = SILVER_DIR / "graficos"

SILVER_ARTEFATOS_DIR = SILVER_DIR / "artefatos"

SILVER_DADOS_DIR = SILVER_DIR / "dados"

# ======================================================================================
# Logs de experimentos dos modelos - Silver
# ======================================================================================

ARQUIVO_LOG_ARVORE_DECISAO = (
    SILVER_DIR / "modelos" / "arvore_decisao.txt"
)

ARQUIVO_LOG_RANDOM_FOREST = (
    SILVER_DIR / "modelos" / "random_forest.txt"
)

ARQUIVO_LOG_REGRESSAO_LOGISTICA = (
    SILVER_DIR / "modelos" / "regressao_logistica.txt"
)

ARQUIVO_LOG_XGBOOST = (
    SILVER_DIR / "modelos" / "xgboost_modelo.txt"
)

ARQUIVO_LOG_LIGHTGBM = (
    SILVER_DIR / "modelos" / "lightgbm_modelo.txt"
)

# ======================================================================================
# Arquivos de saída da camada Silver
# ======================================================================================

ARQUIVO_OUTPUT_SILVER = (
    SILVER_OUTPUTS_DIR / "output_silver.txt"
)

ARQUIVO_METRICAS_MODELOS = (
    SILVER_OUTPUTS_DIR / "metricas_modelos.csv"
)

ARQUIVO_COMPARACAO_MODELOS = (
    SILVER_OUTPUTS_DIR / "comparacao_modelos.csv"
)

ARQUIVO_MELHOR_MODELO = (
    SILVER_ARTEFATOS_DIR / "melhor_modelo.pkl"
)

ARQUIVO_ENCODER = (
    SILVER_ARTEFATOS_DIR / "encoder.pkl"
)

ARQUIVO_SCALER = (
    SILVER_ARTEFATOS_DIR / "scaler.pkl"
)

# ======================================================================================
# Parâmetros da modelagem - Silver
# ======================================================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20

# ======================================================================================
# Parâmetros da Árvore de Decisão - Silver
# ======================================================================================

MAX_DEPTH_ARVORE_DECISAO = 20
MIN_SAMPLES_LEAF_ARVORE_DECISAO = 100

# ======================================================================================
# Parâmetros da Random Forest - Silver
# ======================================================================================

N_ESTIMATORS_RANDOM_FOREST = 100

MAX_DEPTH_RANDOM_FOREST = 20

MIN_SAMPLES_LEAF_RANDOM_FOREST = 100

MAX_FEATURES_RANDOM_FOREST = "sqrt"

N_JOBS_RANDOM_FOREST = -1

# ======================================================================================
# Parâmetros da Regressão Logística - Silver
# ======================================================================================

MAX_ITER_REGRESSAO_LOGISTICA = 1000

C_REGRESSAO_LOGISTICA = 1.0

SOLVER_REGRESSAO_LOGISTICA = "saga"

PENALTY_REGRESSAO_LOGISTICA = "l2"

N_JOBS_REGRESSAO_LOGISTICA = -1

# ======================================================================================
# Parâmetros do XGBoost - Silver
# ======================================================================================

N_ESTIMATORS_XGBOOST = 100

MAX_DEPTH_XGBOOST = 6

LEARNING_RATE_XGBOOST = 0.1

SUBSAMPLE_XGBOOST = 0.8

COLSAMPLE_BYTREE_XGBOOST = 0.8

TREE_METHOD_XGBOOST = "hist"

N_JOBS_XGBOOST = -1

# ======================================================================================
# Parâmetros do LightGBM - Silver
# ======================================================================================

N_ESTIMATORS_LIGHTGBM = 100

MAX_DEPTH_LIGHTGBM = 6

LEARNING_RATE_LIGHTGBM = 0.1

NUM_LEAVES_LIGHTGBM = 31

SUBSAMPLE_LIGHTGBM = 0.8

COLSAMPLE_BYTREE_LIGHTGBM = 0.8

N_JOBS_LIGHTGBM = -1


# ======================================================================================
# Diretórios e logs de : - Silver
# ARVORE DE DECISAO
# RANDOM FOREST
# REGRESSAO LOGISTICA
# XGBOOST
# LIGHTGBM
# ======================================================================================
PASTA_GRAFICOS_ARVORE_DECISAO = (
    SILVER_GRAFICOS_DIR / "arvore_decisao"
)

PASTA_GRAFICOS_RANDOM_FOREST = (
    SILVER_GRAFICOS_DIR / "random_forest"
)

PASTA_GRAFICOS_REGRESSAO_LOGISTICA = (
    SILVER_GRAFICOS_DIR / "regressao_logistica"
)

PASTA_GRAFICOS_XGBOOST = (
    SILVER_GRAFICOS_DIR / "xgboost_modelo"
)

PASTA_GRAFICOS_LIGHTGBM = (
    SILVER_GRAFICOS_DIR / "lightgbm_modelo"
)