# ======================================================================================
# utils.py
# ======================================================================================
# Responsabilidade:
# - Salvar o output da execução em arquivo .txt
# - Manter o output aparecendo também no terminal
# ======================================================================================

import sys
from pathlib import Path
class OutputTerminalEArquivo:
    """
    Classe auxiliar para duplicar o output:
    - mostra no terminal
    - salva em arquivo .txt
    """

    def __init__(self, caminho_arquivo: str | Path):
        self.caminho_arquivo = Path(caminho_arquivo)
        self.terminal = sys.stdout
        self.arquivo = None

    def __enter__(self):
        self.caminho_arquivo.parent.mkdir(parents=True, exist_ok=True)

        self.arquivo = open(
            self.caminho_arquivo,
            mode="w",
            encoding="utf-8"
        )

        sys.stdout = self

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self.terminal

        if self.arquivo:
            self.arquivo.close()

    def write(self, mensagem):
        self.terminal.write(mensagem)

        if self.arquivo:
            self.arquivo.write(mensagem)

    def flush(self):
        self.terminal.flush()

        if self.arquivo:
            self.arquivo.flush()