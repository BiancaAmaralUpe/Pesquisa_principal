# ======================================================================================
# pipeline_silver.py
# ======================================================================================
# Responsabilidade:
# - Ler a base de modelagem gerada pela Bronze
# - Preparar os dados para machine learning
# - Executar modelos de classificação
# - Avaliar e comparar os modelos
# - Salvar métricas, gráficos e melhor modelo
# ======================================================================================
from Pesquisa_principal.constants import ARQUIVO_CSV_LIMPO_TESTE_TREINO
from Pesquisa_principal.constants import ARQUIVO_OUTPUT_SILVER
from Pesquisa_principal.constants import COLUNA_ALVO_MODELAGEM
from Pesquisa_principal.constants import RANDOM_STATE
from Pesquisa_principal.constants import TEST_SIZE
from Pesquisa_principal.constants import ARQUIVO_ENCODER
# constants arvore de decisao
from Pesquisa_principal.constants import MAX_DEPTH_ARVORE_DECISAO
from Pesquisa_principal.constants import MIN_SAMPLES_LEAF_ARVORE_DECISAO
from Pesquisa_principal.constants import PASTA_GRAFICOS_ARVORE_DECISAO
from Pesquisa_principal.constants import ARQUIVO_LOG_ARVORE_DECISAO
from Pesquisa_principal.constants import ATIVAR_PARADA_ANTECIPADA_ARVORE_DECISAO
from Pesquisa_principal.constants import TOLERANCIA_MELHORIA_ARVORE_DECISAO
from Pesquisa_principal.constants import PACIENCIA_ARVORE_DECISAO
# constants random forest
from Pesquisa_principal.constants import N_ESTIMATORS_RANDOM_FOREST
from Pesquisa_principal.constants import MAX_DEPTH_RANDOM_FOREST
from Pesquisa_principal.constants import MIN_SAMPLES_LEAF_RANDOM_FOREST
from Pesquisa_principal.constants import MAX_FEATURES_RANDOM_FOREST
from Pesquisa_principal.constants import N_JOBS_RANDOM_FOREST
from Pesquisa_principal.constants import PASTA_GRAFICOS_RANDOM_FOREST
from Pesquisa_principal.constants import ARQUIVO_LOG_RANDOM_FOREST
# constants regressao logistica
from Pesquisa_principal.constants import MAX_ITER_REGRESSAO_LOGISTICA
from Pesquisa_principal.constants import C_REGRESSAO_LOGISTICA
from Pesquisa_principal.constants import SOLVER_REGRESSAO_LOGISTICA
from Pesquisa_principal.constants import L1_RATIO_REGRESSAO_LOGISTICA
from Pesquisa_principal.constants import PASTA_GRAFICOS_REGRESSAO_LOGISTICA
from Pesquisa_principal.constants import ARQUIVO_LOG_REGRESSAO_LOGISTICA

# constants xgboost
from Pesquisa_principal.constants import N_ESTIMATORS_XGBOOST
from Pesquisa_principal.constants import MAX_DEPTH_XGBOOST
from Pesquisa_principal.constants import LEARNING_RATE_XGBOOST
from Pesquisa_principal.constants import SUBSAMPLE_XGBOOST
from Pesquisa_principal.constants import COLSAMPLE_BYTREE_XGBOOST
from Pesquisa_principal.constants import TREE_METHOD_XGBOOST
from Pesquisa_principal.constants import N_JOBS_XGBOOST
from Pesquisa_principal.constants import PASTA_GRAFICOS_XGBOOST
from Pesquisa_principal.constants import ARQUIVO_LOG_XGBOOST
# constants lightgbm
from Pesquisa_principal.constants import N_ESTIMATORS_LIGHTGBM
from Pesquisa_principal.constants import MAX_DEPTH_LIGHTGBM
from Pesquisa_principal.constants import LEARNING_RATE_LIGHTGBM
from Pesquisa_principal.constants import NUM_LEAVES_LIGHTGBM
from Pesquisa_principal.constants import SUBSAMPLE_LIGHTGBM
from Pesquisa_principal.constants import COLSAMPLE_BYTREE_LIGHTGBM
from Pesquisa_principal.constants import N_JOBS_LIGHTGBM
from Pesquisa_principal.constants import PASTA_GRAFICOS_LIGHTGBM
from Pesquisa_principal.constants import ARQUIVO_LOG_LIGHTGBM
# constants diagnostico target
from Pesquisa_principal.constants import PASTA_GRAFICOS_DIAGNOSTICO_TARGET
# constants balanceamento base
from Pesquisa_principal.constants import APLICAR_BALANCEAMENTO_TREINO
from Pesquisa_principal.constants import ESTRATEGIA_BALANCEAMENTO_TREINO
from Pesquisa_principal.constants import QUANTIDADE_ALVO_BALANCEAMENTO

from Pesquisa_principal.bronze.utils import OutputTerminalEArquivo
from Pesquisa_principal.silver.processamento.normalizacao_colunas_modelo import normalizar_nomes_colunas_modelo
from Pesquisa_principal.silver.diagnostico_target import gerar_graficos_diagnostico_target
from Pesquisa_principal.silver.avaliacao.metricas import exibir_metricas_modelo
from Pesquisa_principal.silver.graficos.volume_relacao_risco import gerar_graficos_volume_relacao_risco
from Pesquisa_principal.silver.graficos.balanceamento_base import gerar_graficos_balanceamento_base
from Pesquisa_principal.silver.processamento.balanceamento_base import aplicar_balanceamento_treino

from Pesquisa_principal.silver.processamento.leitura_dados_bronze import ler_base_modelagem_bronze
from Pesquisa_principal.silver.processamento.separacao_target import separar_variaveis_explicativas_e_target
from Pesquisa_principal.silver.processamento.divisao_treino_teste import dividir_treino_teste_estratificado
from Pesquisa_principal.silver.processamento.encoding import aplicar_encoding_treino_teste

from Pesquisa_principal.silver.modelos.arvore_decisao import treinar_arvore_decisao
from Pesquisa_principal.silver.modelos.arvore_decisao import avaliar_arvore_decisao
from Pesquisa_principal.silver.modelos.grafico_arvore_decisao import gerar_grafico_overfitting_arvore_decisao
from Pesquisa_principal.silver.modelos.arvore_decisao import registrar_treinamento_arvore_decisao
from Pesquisa_principal.silver.modelos.matriz_confusao_arvore_decisao import gerar_matrizes_confusao_arvore_decisao

from Pesquisa_principal.silver.modelos.random_forest import treinar_random_forest
from Pesquisa_principal.silver.modelos.random_forest import avaliar_random_forest
from Pesquisa_principal.silver.modelos.random_forest import registrar_treinamento_random_forest
from Pesquisa_principal.silver.modelos.grafico_random_forest import gerar_grafico_overfitting_random_forest
from Pesquisa_principal.silver.modelos.matriz_confusao_random_forest import gerar_matrizes_confusao_random_forest

from Pesquisa_principal.silver.modelos.regressao_logistica import treinar_regressao_logistica
from Pesquisa_principal.silver.modelos.regressao_logistica import avaliar_regressao_logistica
from Pesquisa_principal.silver.modelos.regressao_logistica import registrar_treinamento_regressao_logistica
from Pesquisa_principal.silver.modelos.grafico_regressao_logistica import gerar_grafico_overfitting_regressao_logistica
from Pesquisa_principal.silver.modelos.matriz_confusao_regressao_logistica import gerar_matrizes_confusao_regressao_logistica

from Pesquisa_principal.silver.modelos.xgboost_modelo import treinar_xgboost
from Pesquisa_principal.silver.modelos.xgboost_modelo import avaliar_xgboost
from Pesquisa_principal.silver.modelos.xgboost_modelo import registrar_treinamento_xgboost
from Pesquisa_principal.silver.modelos.grafico_xgboost_modelo import gerar_grafico_overfitting_xgboost
from Pesquisa_principal.silver.modelos.matriz_confusao_xgboost_modelo import gerar_matrizes_confusao_xgboost

from Pesquisa_principal.silver.modelos.lightgbm_modelo import treinar_lightgbm
from Pesquisa_principal.silver.modelos.lightgbm_modelo import avaliar_lightgbm  
from Pesquisa_principal.silver.modelos.lightgbm_modelo import registrar_treinamento_lightgbm
from Pesquisa_principal.silver.modelos.matriz_confusao_lightgbm_modelo import gerar_matrizes_confusao_lightgbm
from Pesquisa_principal.silver.modelos.grafico_lightgbm_modelo import gerar_grafico_overfitting_lightgbm

# ======================================================================================
# Orquestração dos modelos da camada Silver
# ======================================================================================

MODELOS_TREINAMENTO_SILVER = [
    #"arvore_decisao",
    #"random_forest",
    "regressao_logistica",
    #"xgboost_modelo",
    #"lightgbm_modelo",
]

# ======================================================================================
# Orquestração dos gráficos por modelo - Silver
# ======================================================================================

GRAFICOS_POR_MODELO_SILVER = {
   #"arvore_decisao": [
   #    "overfitting",
   #    "matriz_confusao",
   #],

   #"random_forest": [
   #    "overfitting",
   #    "matriz_confusao",
   #],

    "regressao_logistica": [
        "overfitting",
        "matriz_confusao",
    ],

   # "xgboost_modelo": [
   #     "overfitting",
   #     "matriz_confusao",
   # ],
#
   # "lightgbm_modelo": [
   #     "overfitting",
   #     "matriz_confusao",
   # ],
}
# ==============================================================
# Fluxos dos modelos - Silver
# ==============================================================
def executar_fluxo_lightgbm(
    X_train_encoded,
    X_test_encoded,
    y_train,
    y_test,
) -> dict | None:
    """
    Executa o fluxo completo do LightGBM:
    - treino
    - avaliação
    - registro do experimento
    - gráficos configurados
    """

    if "lightgbm_modelo" not in MODELOS_TREINAMENTO_SILVER:
        print("\nLightGBM desativado na orquestração da Silver.")
        return None

    print("\n" + "=" * 80)
    print("ORQUESTRAÇÃO DO MODELO: LIGHTGBM")
    print("=" * 80)

    modelo_lightgbm = treinar_lightgbm(
        X_train=X_train_encoded,
        y_train=y_train,
        random_state=RANDOM_STATE,
        n_estimators=N_ESTIMATORS_LIGHTGBM,
        max_depth=MAX_DEPTH_LIGHTGBM,
        learning_rate=LEARNING_RATE_LIGHTGBM,
        num_leaves=NUM_LEAVES_LIGHTGBM,
        subsample=SUBSAMPLE_LIGHTGBM,
        colsample_bytree=COLSAMPLE_BYTREE_LIGHTGBM,
        n_jobs=N_JOBS_LIGHTGBM,
    )

    metricas_lightgbm, y_pred_lightgbm = avaliar_lightgbm(
        modelo=modelo_lightgbm,
        X_train=X_train_encoded,
        X_test=X_test_encoded,
        y_train=y_train,
        y_test=y_test,
    )

    registrar_treinamento_lightgbm(
        caminho_log=ARQUIVO_LOG_LIGHTGBM,
        n_estimators=N_ESTIMATORS_LIGHTGBM,
        max_depth=MAX_DEPTH_LIGHTGBM,
        learning_rate=LEARNING_RATE_LIGHTGBM,
        num_leaves=NUM_LEAVES_LIGHTGBM,
        subsample=SUBSAMPLE_LIGHTGBM,
        colsample_bytree=COLSAMPLE_BYTREE_LIGHTGBM,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS_LIGHTGBM,
        X_train_shape=X_train_encoded.shape,
        X_test_shape=X_test_encoded.shape,
        y_train_shape=y_train.shape,
        y_test_shape=y_test.shape,
        metricas=metricas_lightgbm,
        observacao=(
            "Experimento com LightGBM usando One-Hot Encoding, "
            "sample_weight balanceado e base treino/teste estratificada."
        ),
    )

    graficos_ativos = GRAFICOS_POR_MODELO_SILVER.get(
        "lightgbm_modelo",
        [],
    )

    if "overfitting" in graficos_ativos:
        gerar_grafico_overfitting_lightgbm(
            X_train=X_train_encoded,
            X_test=X_test_encoded,
            y_train=y_train,
            y_test=y_test,
            caminho_saida=str(PASTA_GRAFICOS_LIGHTGBM),
            caminho_log=ARQUIVO_LOG_LIGHTGBM,
            n_estimators=N_ESTIMATORS_LIGHTGBM,
            learning_rate=LEARNING_RATE_LIGHTGBM,
            num_leaves=NUM_LEAVES_LIGHTGBM,
            subsample=SUBSAMPLE_LIGHTGBM,
            colsample_bytree=COLSAMPLE_BYTREE_LIGHTGBM,
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS_LIGHTGBM,
        )

    if "matriz_confusao" in graficos_ativos:
        gerar_matrizes_confusao_lightgbm(
            y_real=y_test,
            y_predito=y_pred_lightgbm,
            caminho_saida=str(PASTA_GRAFICOS_LIGHTGBM),
        )

    return metricas_lightgbm

def executar_fluxo_xgboost(
    X_train_encoded,
    X_test_encoded,
    y_train,
    y_test,
) -> dict | None:
    """
    Executa o fluxo completo do XGBoost:
    - treino
    - avaliação
    - registro do experimento
    - gráficos configurados
    """

    if "xgboost_modelo" not in MODELOS_TREINAMENTO_SILVER:
        print("\nXGBoost desativado na orquestração da Silver.")
        return None

    print("\n" + "=" * 80)
    print("ORQUESTRAÇÃO DO MODELO: XGBOOST")
    print("=" * 80)

    # ============================================================== #
    # Treino do XGBoost
    # ============================================================== #
    modelo_xgboost = treinar_xgboost(
        X_train=X_train_encoded,
        y_train=y_train,
        random_state=RANDOM_STATE,
        n_estimators=N_ESTIMATORS_XGBOOST,
        max_depth=MAX_DEPTH_XGBOOST,
        learning_rate=LEARNING_RATE_XGBOOST,
        subsample=SUBSAMPLE_XGBOOST,
        colsample_bytree=COLSAMPLE_BYTREE_XGBOOST,
        tree_method=TREE_METHOD_XGBOOST,
        n_jobs=N_JOBS_XGBOOST,
    )

    # ============================================================== #
    # Avaliação do XGBoost
    # ============================================================== #
    metricas_xgboost, y_pred_xgboost = avaliar_xgboost(
        modelo=modelo_xgboost,
        X_train=X_train_encoded,
        X_test=X_test_encoded,
        y_train=y_train,
        y_test=y_test,
    )

    # ============================================================== #
    # Registro do treinamento do XGBoost
    # ============================================================== #
    registrar_treinamento_xgboost(
        caminho_log=ARQUIVO_LOG_XGBOOST,
        n_estimators=N_ESTIMATORS_XGBOOST,
        max_depth=MAX_DEPTH_XGBOOST,
        learning_rate=LEARNING_RATE_XGBOOST,
        subsample=SUBSAMPLE_XGBOOST,
        colsample_bytree=COLSAMPLE_BYTREE_XGBOOST,
        tree_method=TREE_METHOD_XGBOOST,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS_XGBOOST,
        X_train_shape=X_train_encoded.shape,
        X_test_shape=X_test_encoded.shape,
        y_train_shape=y_train.shape,
        y_test_shape=y_test.shape,
        metricas=metricas_xgboost,
        observacao=(
            "Experimento com XGBoost usando One-Hot Encoding, "
            "sample_weight balanceado e base treino/teste estratificada."
        ),
    )

    graficos_ativos = GRAFICOS_POR_MODELO_SILVER.get(
        "xgboost_modelo",
        [],
    )

    # ============================================================== #
    # Gerar os gráficos configurados
    # ============================================================== #
    if "overfitting" in graficos_ativos:
        gerar_grafico_overfitting_xgboost(
            X_train=X_train_encoded,
            X_test=X_test_encoded,
            y_train=y_train,
            y_test=y_test,
            caminho_saida=str(PASTA_GRAFICOS_XGBOOST),
            caminho_log=ARQUIVO_LOG_XGBOOST,
            n_estimators=N_ESTIMATORS_XGBOOST,
            learning_rate=LEARNING_RATE_XGBOOST,
            subsample=SUBSAMPLE_XGBOOST,
            colsample_bytree=COLSAMPLE_BYTREE_XGBOOST,
            tree_method=TREE_METHOD_XGBOOST,
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS_XGBOOST,
        )

    if "matriz_confusao" in graficos_ativos:
        gerar_matrizes_confusao_xgboost(
            y_real=y_test,
            y_predito=y_pred_xgboost,
            caminho_saida=str(PASTA_GRAFICOS_XGBOOST),
        )

    return metricas_xgboost

# ============================================================== #
# 
# ============================================================== #

def executar_fluxo_regressao_logistica(
    X_train_encoded,
    X_test_encoded,
    y_train,
    y_test,
) -> dict | None:
    """
    Executa o fluxo completo da Regressão Logística:
    - treino
    - avaliação
    - registro do experimento
    - gráficos configurados
    """

    if "regressao_logistica" not in MODELOS_TREINAMENTO_SILVER:
        print("\nRegressão Logística desativada na orquestração da Silver.")
        return None

    print("\n" + "=" * 80)
    print("ORQUESTRAÇÃO DO MODELO: REGRESSÃO LOGÍSTICA")
    print("=" * 80)

    # ============================================================== #
    # 
    # ============================================================== #
    modelo_regressao_logistica = treinar_regressao_logistica(
        X_train=X_train_encoded,
        y_train=y_train,
        random_state=RANDOM_STATE,
        max_iter=MAX_ITER_REGRESSAO_LOGISTICA,
        C=C_REGRESSAO_LOGISTICA,
        solver=SOLVER_REGRESSAO_LOGISTICA,
        l1_ratio=L1_RATIO_REGRESSAO_LOGISTICA,
    )

    # ============================================================== #
    # 
    # ============================================================== #
    metricas_regressao_logistica, y_pred_regressao_logistica = avaliar_regressao_logistica(
        modelo=modelo_regressao_logistica,
        X_train=X_train_encoded,
        X_test=X_test_encoded,
        y_train=y_train,
        y_test=y_test,
    )
    
    # ============================================================== #
    # 
    # ============================================================== #
    registrar_treinamento_regressao_logistica(
        caminho_log=ARQUIVO_LOG_REGRESSAO_LOGISTICA,
        max_iter=MAX_ITER_REGRESSAO_LOGISTICA,
        C=C_REGRESSAO_LOGISTICA,
        solver=SOLVER_REGRESSAO_LOGISTICA,
        l1_ratio=L1_RATIO_REGRESSAO_LOGISTICA,
        random_state=RANDOM_STATE,
        X_train_shape=X_train_encoded.shape,
        X_test_shape=X_test_encoded.shape,
        y_train_shape=y_train.shape,
        y_test_shape=y_test.shape,
        metricas=metricas_regressao_logistica,
        observacao=(
            "Experimento com Regressão Logística usando One-Hot Encoding, "
            "class_weight='balanced', l1_ratio=0.0 e base treino/teste estratificada."
        ),
    )
    # ============================================================== #
    # 
    # ============================================================== #
    graficos_ativos = GRAFICOS_POR_MODELO_SILVER.get(
        "regressao_logistica",
        [],
    )
    # ============================================================== #
    # 
    # ============================================================== #

    if "overfitting" in graficos_ativos:
        gerar_grafico_overfitting_regressao_logistica(
            X_train=X_train_encoded,
            X_test=X_test_encoded,
            y_train=y_train,
            y_test=y_test,
            caminho_saida=str(PASTA_GRAFICOS_REGRESSAO_LOGISTICA),
            caminho_log=ARQUIVO_LOG_REGRESSAO_LOGISTICA,
            max_iter=MAX_ITER_REGRESSAO_LOGISTICA,
            solver=SOLVER_REGRESSAO_LOGISTICA,
            l1_ratio=L1_RATIO_REGRESSAO_LOGISTICA,
            random_state=RANDOM_STATE,
        )
    # ============================================================== #
    # 
    # ============================================================== #
    if "matriz_confusao" in graficos_ativos:
        gerar_matrizes_confusao_regressao_logistica(
            y_real=y_test,
            y_predito=y_pred_regressao_logistica,
            caminho_saida=str(PASTA_GRAFICOS_REGRESSAO_LOGISTICA),
        )
    return metricas_regressao_logistica

# ============================================================== #
# 
# ============================================================== #
def executar_fluxo_arvore_decisao(
    X_train_encoded,
    X_test_encoded,
    y_train,
    y_test,
) -> dict | None:
    """
    Executa o fluxo completo da Árvore de Decisão:
    - treino
    - avaliação
    - registro do experimento
    - gráficos configurados
    """

    if "arvore_decisao" not in MODELOS_TREINAMENTO_SILVER:
        print("\nÁrvore de Decisão desativada na orquestração da Silver.")
        return None

    print("\n" + "=" * 80)
    print("ORQUESTRAÇÃO DO MODELO: ÁRVORE DE DECISÃO")
    print("=" * 80)
    
    # ============================================================== #
    # Treino do Árvore de Decisão
    # ============================================================== #
    modelo_arvore_decisao = treinar_arvore_decisao(
        X_train=X_train_encoded,
        y_train=y_train,
        random_state=RANDOM_STATE,
        max_depth=MAX_DEPTH_ARVORE_DECISAO,
        min_samples_leaf=MIN_SAMPLES_LEAF_ARVORE_DECISAO,
    )
    # ============================================================== #
    # 
    # ============================================================== #
    metricas_arvore_decisao, y_pred_arvore_decisao = avaliar_arvore_decisao(
        modelo=modelo_arvore_decisao,
        X_train=X_train_encoded,
        X_test=X_test_encoded,
        y_train=y_train,
        y_test=y_test,
    )
    # ============================================================== #
    # 
    # ============================================================== #
    registrar_treinamento_arvore_decisao(
        caminho_log=ARQUIVO_LOG_ARVORE_DECISAO,
        max_depth=MAX_DEPTH_ARVORE_DECISAO,
        min_samples_leaf=MIN_SAMPLES_LEAF_ARVORE_DECISAO,
        random_state=RANDOM_STATE,
        X_train_shape=X_train_encoded.shape,
        X_test_shape=X_test_encoded.shape,
        y_train_shape=y_train.shape,
        y_test_shape=y_test.shape,
        metricas=metricas_arvore_decisao,
        observacao=(
            "Experimento com Árvore de Decisão usando One-Hot Encoding, "
            "class_weight='balanced' e base treino/teste estratificada."
        ),
    )
    # ============================================================== #
    # 
    # ============================================================== #
    graficos_ativos = GRAFICOS_POR_MODELO_SILVER.get(
        "arvore_decisao",
        [],
    )
    # ============================================================== #
    # 
    # ============================================================== #

    if "overfitting" in graficos_ativos:
        gerar_grafico_overfitting_arvore_decisao(
            X_train=X_train_encoded,
            X_test=X_test_encoded,
            y_train=y_train,
            y_test=y_test,
            caminho_saida=str(PASTA_GRAFICOS_ARVORE_DECISAO),
            caminho_log=ARQUIVO_LOG_ARVORE_DECISAO,
            min_samples_leaf=MIN_SAMPLES_LEAF_ARVORE_DECISAO,
            random_state=RANDOM_STATE,
            ativar_parada_antecipada=ATIVAR_PARADA_ANTECIPADA_ARVORE_DECISAO,
            tolerancia_melhoria=TOLERANCIA_MELHORIA_ARVORE_DECISAO,
            paciencia=PACIENCIA_ARVORE_DECISAO,
        )
    # ============================================================== #
    # 
    # ============================================================== #
    if "matriz_confusao" in graficos_ativos:
        gerar_matrizes_confusao_arvore_decisao(
            y_real=y_test,
            y_predito=y_pred_arvore_decisao,
            caminho_saida=str(PASTA_GRAFICOS_ARVORE_DECISAO),
        )

    return metricas_arvore_decisao

# ============================================================== #
# 
# ============================================================== #
def executar_fluxo_random_forest(
    X_train_encoded,
    X_test_encoded,
    y_train,
    y_test,
) -> dict | None:
    """
    Executa o fluxo completo da Random Forest:
    - treino
    - avaliação
    - registro do experimento
    - gráficos configurados
    """

    if "random_forest" not in MODELOS_TREINAMENTO_SILVER:
        print("\nRandom Forest desativada na orquestração da Silver.")
        return None

    print("\n" + "=" * 80)
    print("ORQUESTRAÇÃO DO MODELO: RANDOM FOREST")
    print("=" * 80)
    
    # ============================================================== #
    # Treino da Random Forest
    # ============================================================== #
    modelo_random_forest = treinar_random_forest(
        X_train=X_train_encoded,
        y_train=y_train,
        random_state=RANDOM_STATE,
        n_estimators=N_ESTIMATORS_RANDOM_FOREST,
        max_depth=MAX_DEPTH_RANDOM_FOREST,
        min_samples_leaf=MIN_SAMPLES_LEAF_RANDOM_FOREST,
        max_features=MAX_FEATURES_RANDOM_FOREST,
        n_jobs=N_JOBS_RANDOM_FOREST,
    )
    # ============================================================== #
    # 
    # ============================================================== #
    metricas_random_forest, y_pred_random_forest = avaliar_random_forest(
        modelo=modelo_random_forest,
        X_train=X_train_encoded,
        X_test=X_test_encoded,
        y_train=y_train,
        y_test=y_test,
    )
    # ============================================================== #
    # 
    # ============================================================== #
    registrar_treinamento_random_forest(
        caminho_log=ARQUIVO_LOG_RANDOM_FOREST,
        n_estimators=N_ESTIMATORS_RANDOM_FOREST,
        max_depth=MAX_DEPTH_RANDOM_FOREST,
        min_samples_leaf=MIN_SAMPLES_LEAF_RANDOM_FOREST,
        max_features=MAX_FEATURES_RANDOM_FOREST,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS_RANDOM_FOREST,
        X_train_shape=X_train_encoded.shape,
        X_test_shape=X_test_encoded.shape,
        y_train_shape=y_train.shape,
        y_test_shape=y_test.shape,
        metricas=metricas_random_forest,
        observacao=(
            "Experimento com Random Forest usando One-Hot Encoding, "
            "class_weight='balanced_subsample' e base treino/teste estratificada."
        ),
    )
    # ============================================================== #
    # 
    # ============================================================== #
    graficos_ativos = GRAFICOS_POR_MODELO_SILVER.get(
        "random_forest",
        [],
    )
    # ============================================================== #
    # 
    # ============================================================== #

    if "overfitting" in graficos_ativos:
        gerar_grafico_overfitting_random_forest(
            X_train=X_train_encoded,
            X_test=X_test_encoded,
            y_train=y_train,
            y_test=y_test,
            caminho_saida=str(PASTA_GRAFICOS_RANDOM_FOREST),
            caminho_log=ARQUIVO_LOG_RANDOM_FOREST,
            n_estimators=N_ESTIMATORS_RANDOM_FOREST,
            min_samples_leaf=MIN_SAMPLES_LEAF_RANDOM_FOREST,
            max_features=MAX_FEATURES_RANDOM_FOREST,
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS_RANDOM_FOREST,
        )
    # ============================================================== #
    # 
    # ============================================================== #
    if "matriz_confusao" in graficos_ativos:
        gerar_matrizes_confusao_random_forest(
            y_real=y_test,
            y_predito=y_pred_random_forest,
            caminho_saida=str(PASTA_GRAFICOS_RANDOM_FOREST),
        )

    return metricas_random_forest

def pipeline_silver() -> None:
    """
    Executa a camada Silver.
    """

    with OutputTerminalEArquivo(ARQUIVO_OUTPUT_SILVER):
        print("=" * 80)
        print("INÍCIO DA PIPELINE SILVER - MODELAGEM")
        print("=" * 80)

        # ==============================================================
        # leitura da base de modelagem
        # ==============================================================
        dataframe_modelagem = ler_base_modelagem_bronze(
            caminho_entrada=ARQUIVO_CSV_LIMPO_TESTE_TREINO,
            coluna_alvo=COLUNA_ALVO_MODELAGEM,
        )

        # ==============================================================
        # diagnóstico visual do target
        # ==============================================================
        gerar_graficos_diagnostico_target(
            dataframe=dataframe_modelagem,
            caminho_saida=PASTA_GRAFICOS_DIAGNOSTICO_TARGET,
            coluna_target=COLUNA_ALVO_MODELAGEM,
        )
        # ==============================================================
        # volume de registros com relação com risco de morte
        # ==============================================================
        gerar_graficos_volume_relacao_risco(
            dataframe=dataframe_modelagem,
            caminho_saida=PASTA_GRAFICOS_DIAGNOSTICO_TARGET,
            coluna_target=COLUNA_ALVO_MODELAGEM,
            coluna_relacao_risco="agravantes_unificados",
            termo_relacao_risco="RISCO DE MORTE",
        )

        # ==============================================================
        # separação variáveis explicativas e target
        # ==============================================================
        X, y = separar_variaveis_explicativas_e_target(
            dataframe=dataframe_modelagem,
            coluna_alvo=COLUNA_ALVO_MODELAGEM,
        )

        # ==============================================================
        # divisão treino/teste
        # ==============================================================
        X_train, X_test, y_train, y_test = dividir_treino_teste_estratificado(
            X=X,
            y=y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
        )

        # ==============================================================
        # diagnóstico de balanceamento da base
        # ==============================================================
        gerar_graficos_balanceamento_base(
            y=y,
            y_train=y_train,
            y_test=y_test,
            caminho_saida=PASTA_GRAFICOS_DIAGNOSTICO_TARGET,
            nome_coluna_target=COLUNA_ALVO_MODELAGEM,
        )
        
        # ==============================================================
        # encoding
        # ==============================================================
        X_train_encoded, X_test_encoded, encoder = aplicar_encoding_treino_teste(
            X_train=X_train,
            X_test=X_test,
            caminho_encoder=ARQUIVO_ENCODER,
        )
        # ==============================================================
        # normalização dos nomes das colunas para compatibilidade com modelos
        # ==============================================================
        X_train_encoded, X_test_encoded = normalizar_nomes_colunas_modelo(
            X_train=X_train_encoded,
            X_test=X_test_encoded,
        )
        # ==============================================================
        # balanceamento da base de treino
        # ==============================================================
        if APLICAR_BALANCEAMENTO_TREINO:
            X_train_modelo, y_train_modelo = aplicar_balanceamento_treino(
                X_train=X_train_encoded,
                y_train=y_train,
                estrategia=ESTRATEGIA_BALANCEAMENTO_TREINO,
                random_state=RANDOM_STATE,
                quantidade_alvo=QUANTIDADE_ALVO_BALANCEAMENTO,
            )
        else:
            X_train_modelo = X_train_encoded
            y_train_modelo = y_train

        # ==============================================================
        # lista de métricas dos modelos
        # ==============================================================
        metricas_modelos = []

        # ==============================================================
        # Árvore de Decisão
        # ==============================================================
        metricas_arvore_decisao = executar_fluxo_arvore_decisao(
            X_train_encoded=X_train_modelo,
            X_test_encoded=X_test_encoded,
            y_train=y_train_modelo,
            y_test=y_test,
        )

        if metricas_arvore_decisao is not None:
            metricas_modelos.append(metricas_arvore_decisao)

        # ==============================================================
        # Random Forest
        # ==============================================================
        metricas_random_forest = executar_fluxo_random_forest(
            X_train_encoded=X_train_modelo,
            X_test_encoded=X_test_encoded,
            y_train=y_train_modelo,
            y_test=y_test,
        )

        if metricas_random_forest is not None:
            metricas_modelos.append(metricas_random_forest)

        # ==============================================================
        # Regressão Logística
        # ==============================================================
        metricas_regressao_logistica = executar_fluxo_regressao_logistica(
            X_train_encoded=X_train_modelo,
            X_test_encoded=X_test_encoded,
            y_train=y_train_modelo,
            y_test=y_test,
        )

        if metricas_regressao_logistica is not None:
            metricas_modelos.append(metricas_regressao_logistica)
        
        # ==============================================================
        # XGBoost
        # ==============================================================
        metricas_xgboost = executar_fluxo_xgboost(
            X_train_encoded=X_train_modelo,
            X_test_encoded=X_test_encoded,
            y_train=y_train_modelo,
            y_test=y_test,
        )

        if metricas_xgboost is not None:
            metricas_modelos.append(metricas_xgboost)

        # ==============================================================
        # LightGBM
        # ==============================================================
        metricas_lightgbm = executar_fluxo_lightgbm(
            X_train_encoded=X_train_modelo,
            X_test_encoded=X_test_encoded,
            y_train=y_train_modelo,
            y_test=y_test,
        )

        if metricas_lightgbm is not None:
            metricas_modelos.append(metricas_lightgbm)

        print("\n" + "=" * 80)
        print("BASE CODIFICADA E MODELOS SILVER AVALIADOS")
        print("=" * 80)

        print(f"X_train original: {X_train.shape}")
        print(f"X_test original: {X_test.shape}")
        print(f"X_train encoded: {X_train_encoded.shape}")
        print(f"X_train modelo: {X_train_modelo.shape}")
        print(f"y_train modelo: {y_train_modelo.shape}")
        print(f"y_train original: {y_train.shape}")
        print(f"y_test: {y_test.shape}")

        for metricas_modelo in metricas_modelos:
            exibir_metricas_modelo(metricas_modelo)

        print("\n" + "=" * 80)
        print("FIM DA PIPELINE SILVER - MODELAGEM")
        print("=" * 80)

        print(f"\nOutput Silver salvo em: {ARQUIVO_OUTPUT_SILVER}")