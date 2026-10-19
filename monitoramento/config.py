"""
Configurações do projeto.

Tudo que pode ser ajustado (limites clínicos, chaves do Azure, nomes de
modelos) fica concentrado aqui, para não precisar procurar valores
espalhados pelo código.

IMPORTANTE: os limites abaixo são ILUSTRATIVOS, para fins acadêmicos.
Em um hospital de verdade eles precisam ser definidos pela equipe médica
e pela farmácia clínica.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Pastas
# ---------------------------------------------------------------------------
PASTA_PROJETO = Path(__file__).resolve().parent.parent
PASTA_DADOS = PASTA_PROJETO / "dados"
PASTA_SAIDA = PASTA_PROJETO / "saida"

# Lê as variáveis do arquivo .env (se existir)
load_dotenv(PASTA_PROJETO / ".env")

# ---------------------------------------------------------------------------
# Azure (preenchidos pelo arquivo .env)
# ---------------------------------------------------------------------------
AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY", "")
AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION", "")
AZURE_LANGUAGE_KEY = os.getenv("AZURE_LANGUAGE_KEY", "")
AZURE_LANGUAGE_ENDPOINT = os.getenv("AZURE_LANGUAGE_ENDPOINT", "")
IDIOMA = "pt-BR"

# Para onde os alertas são enviados.
#
# Exemplo fixo para a demonstração: URL do https://webhook.site gerada pelo grupo.
# Para ver os alertas chegando em tempo real, abra https://webhook.site no mesmo navegador
# em que esta URL foi gerada. Se ela expirar, gere outra no site e substitua abaixo.
WEBHOOK_DEMONSTRACAO = "https://webhook.site/bb185afd-8555-45f3-9e21-2377dc8f58d1"
#
# Opção para enviar ao canal correto da equipe (Discord, Slack...) sem mexer no código:
# defina WEBHOOK_EQUIPE_MEDICA no arquivo .env. Se existir, ele tem prioridade sobre o exemplo fixo.
WEBHOOK_EQUIPE_MEDICA = os.getenv("WEBHOOK_EQUIPE_MEDICA") or WEBHOOK_DEMONSTRACAO


def azure_speech_configurado():
    return bool(AZURE_SPEECH_KEY and AZURE_SPEECH_REGION)


def azure_language_configurado():
    return bool(AZURE_LANGUAGE_KEY and AZURE_LANGUAGE_ENDPOINT)


# ---------------------------------------------------------------------------
# Vídeo
# ---------------------------------------------------------------------------
# Os pesos são baixados automaticamente pela biblioteca ultralytics.
MODELO_POSE = "yolov8n-pose.pt"     # fisioterapia -> YOLOv8-Pose: 17 pontos do corpo (padrão COCO/OpenPose)
MODELO_OBJETOS = "yolov8n.pt"       # cirurgia     -> YOLOv8: detecção de objetos

PROCESSAR_A_CADA_N_QUADROS = 3      # analisa 1 a cada 3 quadros (deixa mais rápido)
CONFIANCA_MINIMA_DETECCAO = 0.4     # ignora pessoas/objetos detectados com confiança menor que isso
CONFIANCA_MINIMA_PONTOS = 0.3       # ignora pontos do corpo com confiança menor que isso

LIMITE_INCLINACAO_TRONCO_GRAUS = 45   # tronco mais inclinado que isso = postura inadequada
LIMITE_DESCIDA_QUEDA = 0.25           # quadril desceu 25% da altura do corpo em até 1 s...
INCLINACAO_DEITADO_GRAUS = 60         # ...e o tronco ficou mais inclinado que isso = possível queda
DURACAO_MINIMA_EVENTO = 0.5           # eventos mais curtos que isso são ignorados (ruído), exceto queda

# Área crítica do vídeo (ex.: campo cirúrgico), em frações da imagem:
# (x_inicial, y_inicial, x_final, y_final) -> 0 = canto esquerdo/topo, 1 = direito/baixo
AREA_CRITICA = (0.30, 0.30, 0.70, 0.70)

# Objetos (classes do YOLO, em inglês) que não deveriam aparecer na área crítica
OBJETOS_DE_RISCO = {
    "knife": "instrumento cortante",
    "scissors": "tesoura",
    "cell phone": "celular",
    "bottle": "garrafa",
    "cup": "copo",
}

# ---------------------------------------------------------------------------
# Áudio
# ---------------------------------------------------------------------------
LIMITE_PROPORCAO_PAUSAS = 0.45     # mais de 45% do áudio em silêncio -> possível cansaço
LIMITE_PAUSAS_POR_MINUTO = 25      # fala muito "picotada" -> possível dificuldade respiratória
LIMITE_PALAVRAS_POR_MINUTO = 90    # fala lenta (pessoas falam ~130 a 170 palavras/min) -> fadiga/disartria

# Termos críticos procurados na transcrição -> gravidade do alerta
TERMOS_CRITICOS = {
    "dor no peito": "alta",
    "falta de ar": "alta",
    "não consigo respirar": "alta",
    "desmaio": "alta",
    "desmaiei": "alta",
    "convulsão": "alta",
    "sangramento": "alta",
    "palpitação": "media",
    "tontura": "media",
    "febre": "media",
    "vômito": "media",
    "dormência": "media",
    "visão turva": "media",
    "confusão": "media",
    "fraqueza": "media",
    "cansaço": "media",
    "dor forte": "media",
}

# ---------------------------------------------------------------------------
# Sinais vitais: (mínimo, máximo) aceitável
# ---------------------------------------------------------------------------
LIMITES_SINAIS_VITAIS = {
    "frequencia_cardiaca": (50, 120),   # batimentos por minuto
    "pressao_sistolica": (90, 180),     # mmHg
    "pressao_diastolica": (50, 110),    # mmHg
    "saturacao_oxigenio": (92, 100),    # %
}

NOMES_SINAIS_VITAIS = {
    "frequencia_cardiaca": "Frequência cardíaca",
    "pressao_sistolica": "Pressão sistólica",
    "pressao_diastolica": "Pressão diastólica",
    "saturacao_oxigenio": "Saturação de oxigênio",
}

JANELA_HISTORICO = 12        # quantas leituras anteriores formam a "linha de base" do paciente
LIMITE_ZSCORE = 4.0          # quantos desvios-padrão de diferença contam como variação brusca

# Para ser "variação brusca" a mudança também precisa ser clinicamente relevante
# (evita alertas por pequenas oscilações normais)
VARIACAO_MINIMA = {
    "frequencia_cardiaca": 15,   # bpm
    "pressao_sistolica": 20,     # mmHg
    "pressao_diastolica": 15,    # mmHg
    "saturacao_oxigenio": 3,     # pontos percentuais
}

CONTAMINACAO_ISOLATION_FOREST = 0.005   # fração esperada de leituras anômalas (0,5%)

# ---------------------------------------------------------------------------
# Prescrições
# ---------------------------------------------------------------------------
# Dose máxima por administração (na mesma unidade usada na prescrição)
DOSE_MAXIMA = {
    "dipirona": 1000,          # mg
    "paracetamol": 1000,       # mg
    "morfina": 10,             # mg
    "omeprazol": 40,           # mg
    "enoxaparina": 80,         # mg
    "insulina regular": 20,    # UI
}

# Intervalo mínimo entre duas doses do mesmo medicamento (horas)
INTERVALO_MINIMO_HORAS = {
    "dipirona": 6,
    "paracetamol": 4,
    "morfina": 4,
    "omeprazol": 12,
    "enoxaparina": 12,
}
INTERVALO_MINIMO_PADRAO = 4

FATOR_AUMENTO_BRUSCO = 2.0   # dose 2x maior que a anterior = aumento brusco

# Medicamentos de alta vigilância: alerta quando são introduzidos no meio da internação
MEDICAMENTOS_ALTA_VIGILANCIA = [
    "morfina",
    "insulina regular",
    "heparina",
    "enoxaparina",
    "cloreto de potassio",
]

# ---------------------------------------------------------------------------
# Movimentação do paciente (leituras por hora de um sensor no leito)
# ---------------------------------------------------------------------------
LIMITE_INATIVIDADE = 3              # nível de atividade abaixo disso = paciente parado
HORAS_INATIVIDADE_PARA_ALERTA = 3   # parado por 3 horas seguidas = risco de lesão por pressão
LIMITE_AGITACAO_NOTURNA = 50        # atividade acima disso durante a noite = agitação
HORA_INICIO_NOITE = 22
HORA_FIM_NOITE = 6
QUEDA_MOBILIDADE_PERCENTUAL = 50    # média do dia caiu mais de 50% em relação ao dia anterior

# ---------------------------------------------------------------------------
# Alertas
# ---------------------------------------------------------------------------
# Alertas iguais (mesmo paciente, mesma origem e mesmo tipo) dentro desta janela viram um só
JANELA_ALERTAS_REPETIDOS_MINUTOS = 30
