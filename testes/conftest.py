# Permite que os testes importem o pacote "monitoramento" a partir da raiz do projeto
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
