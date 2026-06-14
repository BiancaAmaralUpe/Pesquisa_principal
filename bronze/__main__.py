# ======================================================================================
# __main__.py
# ======================================================================================
# Responsabilidade:
# - Ponto de entrada da camada bronze
# - como executar:
#   python -m Pesquisa_principal.bronze
# ======================================================================================

# pyrefly: ignore [missing-import]
from Pesquisa_principal.bronze.pipeline_bronze import pipeline_bronze


def main() -> None:
    pipeline_bronze()


if __name__ == "__main__":
    main()