"""
Análise de vídeos clínicos (sessões de fisioterapia ou cirurgias).

Cada tipo de vídeo usa o modelo pedido no enunciado:
  - Fisioterapia -> YOLOv8-Pose (análise postural). Encontra a pessoa e 17
    pontos do corpo (ombros, quadril, punhos...), o mesmo padrão do OpenPose.
      * Postura inadequada: tronco muito inclinado.
      * Possível queda: quadril desce rápido e o tronco fica deitado.
  - Cirurgia -> YOLOv8 (objetos e áreas críticas).
      * Objeto de risco dentro da área crítica (ex.: celular no campo cirúrgico).

O vídeo é lido quadro a quadro com OpenCV, como nas aulas de análise de vídeo.
Cada quadro com problema gera uma "ocorrência"; ocorrências seguidas do mesmo
tipo são juntadas em um "evento" com início e fim, que vira alerta e entra no
relatório.
"""

import math
import unicodedata
from datetime import datetime
from pathlib import Path

import cv2

from monitoramento import config
from monitoramento.alertas import Alerta

# Índices dos pontos do corpo no padrão COCO (usado pelo YOLOv8-Pose e pelo OpenPose)
OMBRO_ESQ, OMBRO_DIR = 5, 6
QUADRIL_ESQ, QUADRIL_DIR = 11, 12
PONTOS_DO_TRONCO = [OMBRO_ESQ, OMBRO_DIR, QUADRIL_ESQ, QUADRIL_DIR]

GRAVIDADE_EVENTO = {
    "Possível queda": "alta",
    "Objeto de risco na área crítica": "alta",
    "Postura inadequada": "media",
}


# ---------------------------------------------------------------------------
# Geometria
# ---------------------------------------------------------------------------
def ponto_medio(ponto_a, ponto_b):
    return ((ponto_a[0] + ponto_b[0]) / 2, (ponto_a[1] + ponto_b[1]) / 2)


def inclinacao_tronco(pontos):
    """Ângulo (graus) entre o tronco e a vertical. 0 = em pé, 90 = deitado."""
    ombros = ponto_medio(pontos[OMBRO_ESQ], pontos[OMBRO_DIR])
    quadril = ponto_medio(pontos[QUADRIL_ESQ], pontos[QUADRIL_DIR])
    dx = ombros[0] - quadril[0]
    dy = quadril[1] - ombros[1]   # no vídeo o eixo y cresce para baixo
    return abs(math.degrees(math.atan2(dx, dy)))


def area_critica_em_pixels(largura, altura):
    x1, y1, x2, y2 = config.AREA_CRITICA
    return int(x1 * largura), int(y1 * altura), int(x2 * largura), int(y2 * altura)


def dentro_da_area_critica(x, y, largura, altura):
    x1, y1, x2, y2 = area_critica_em_pixels(largura, altura)
    return x1 <= x <= x2 and y1 <= y <= y2


# ---------------------------------------------------------------------------
# Fisioterapia: regras de postura (YOLOv8-Pose)
# ---------------------------------------------------------------------------
def pessoa_principal(resultado):
    """Escolhe a maior pessoa detectada (normalmente o paciente).
    Devolve (pontos, confiancas, altura_do_corpo) ou None."""
    if resultado.keypoints is None or len(resultado.boxes) == 0:
        return None

    caixas = resultado.boxes.xyxy.cpu().numpy()
    areas = (caixas[:, 2] - caixas[:, 0]) * (caixas[:, 3] - caixas[:, 1])
    indice = int(areas.argmax())

    pontos = resultado.keypoints.xy[indice].cpu().numpy()
    confiancas = resultado.keypoints.conf[indice].cpu().numpy()
    altura_corpo = max(caixas[indice, 3] - caixas[indice, 1], 1.0)
    return pontos, confiancas, altura_corpo


def verificar_postura(pessoa, historico_quadril, segundo):
    """Devolve a lista de (tipo, detalhe) encontrados neste quadro.

    historico_quadril: lista de (segundo, altura do quadril na imagem) do
    último 1 segundo; é atualizada aqui e serve para perceber uma queda.
    """
    pontos, confiancas, altura_corpo = pessoa
    if any(confiancas[i] < config.CONFIANCA_MINIMA_PONTOS for i in PONTOS_DO_TRONCO):
        return []   # tronco não está visível neste quadro

    ocorrencias = []
    inclinacao = inclinacao_tronco(pontos)
    if inclinacao > config.LIMITE_INCLINACAO_TRONCO_GRAUS:
        ocorrencias.append(("Postura inadequada", f"tronco inclinado {inclinacao:.0f}°"))

    # Possível queda: em até 1 s o quadril desceu bastante e o tronco ficou deitado
    quadril_y = ponto_medio(pontos[QUADRIL_ESQ], pontos[QUADRIL_DIR])[1]
    historico_quadril.append((segundo, quadril_y))
    while segundo - historico_quadril[0][0] > 1.0:
        historico_quadril.pop(0)   # descarta posições com mais de 1 segundo

    mais_alto = min(y for _, y in historico_quadril)   # no vídeo, y menor = mais alto
    descida = (quadril_y - mais_alto) / altura_corpo   # em "alturas do corpo"
    if descida > config.LIMITE_DESCIDA_QUEDA and inclinacao > config.INCLINACAO_DEITADO_GRAUS:
        ocorrencias.append(("Possível queda", f"quadril desceu {descida:.0%} da altura do corpo em 1 s"))

    return ocorrencias


# ---------------------------------------------------------------------------
# Cirurgia: objetos na área crítica (YOLOv8)
# ---------------------------------------------------------------------------
def objetos_na_area_critica(resultado, largura, altura):
    ocorrencias = []
    for caixa in resultado.boxes:
        nome = resultado.names[int(caixa.cls)]
        x1, y1, x2, y2 = caixa.xyxy[0].tolist()
        if nome in config.OBJETOS_DE_RISCO and dentro_da_area_critica((x1 + x2) / 2, (y1 + y2) / 2,
                                                                      largura, altura):
            ocorrencias.append(("Objeto de risco na área crítica", config.OBJETOS_DE_RISCO[nome]))
    return ocorrencias


# ---------------------------------------------------------------------------
# Ocorrências -> eventos
# ---------------------------------------------------------------------------
def agrupar_em_eventos(ocorrencias, intervalo_max_segundos=1.0):
    """ocorrencias: lista de (segundo, tipo, detalhe), em ordem de tempo.
    Ocorrências do mesmo tipo com menos de 1 s entre si viram um evento só."""
    eventos = []
    aberto = {}   # tipo -> evento que ainda está acontecendo

    for segundo, tipo, detalhe in ocorrencias:
        evento = aberto.get(tipo)
        if evento and segundo - evento["fim_segundos"] <= intervalo_max_segundos:
            evento["fim_segundos"] = segundo
        else:
            evento = {"tipo": tipo, "inicio_segundos": segundo, "fim_segundos": segundo,
                      "detalhe": detalhe, "gravidade": GRAVIDADE_EVENTO[tipo]}
            eventos.append(evento)
            aberto[tipo] = evento

    # Eventos muito curtos são ruído do modelo; a queda é a exceção, porque é rápida mesmo
    return [e for e in eventos
            if e["tipo"] == "Possível queda"
            or e["fim_segundos"] - e["inicio_segundos"] >= config.DURACAO_MINIMA_EVENTO]


def formatar_tempo(segundos):
    return f"{int(segundos // 60):02d}:{segundos % 60:04.1f}"


# ---------------------------------------------------------------------------
# Função principal
# ---------------------------------------------------------------------------
def analisar_video(caminho_video, paciente_id, tipo_video="fisioterapia"):
    # Importado aqui para que as outras etapas funcionem mesmo sem o ultralytics instalado
    from ultralytics import YOLO

    caminho_video = Path(caminho_video)
    fisioterapia = tipo_video == "fisioterapia"
    print(f"Analisando vídeo ({tipo_video}): {caminho_video.name}")

    captura = cv2.VideoCapture(str(caminho_video))
    if not captura.isOpened():
        raise FileNotFoundError(f"Não foi possível abrir o vídeo {caminho_video}")

    fps = captura.get(cv2.CAP_PROP_FPS) or 30
    largura = int(captura.get(cv2.CAP_PROP_FRAME_WIDTH))
    altura = int(captura.get(cv2.CAP_PROP_FRAME_HEIGHT))
    modelo = YOLO(config.MODELO_POSE if fisioterapia else config.MODELO_OBJETOS)

    config.PASTA_SAIDA.mkdir(parents=True, exist_ok=True)
    caminho_anotado = config.PASTA_SAIDA / f"{caminho_video.stem}_anotado.mp4"
    gravador = cv2.VideoWriter(str(caminho_anotado), cv2.VideoWriter_fourcc(*"mp4v"),
                               fps / config.PROCESSAR_A_CADA_N_QUADROS, (largura, altura))

    ocorrencias = []
    historico_quadril = []
    numero_quadro = 0
    quadros_analisados = 0

    while True:
        conseguiu_ler, quadro = captura.read()
        if not conseguiu_ler:
            break
        numero_quadro += 1
        if numero_quadro % config.PROCESSAR_A_CADA_N_QUADROS != 0:
            continue   # analisa só 1 a cada N quadros

        quadros_analisados += 1
        segundo = numero_quadro / fps
        resultado = modelo(quadro, conf=config.CONFIANCA_MINIMA_DETECCAO, verbose=False)[0]

        if fisioterapia:
            pessoa = pessoa_principal(resultado)
            problemas = verificar_postura(pessoa, historico_quadril, segundo) if pessoa else []
        else:
            problemas = objetos_na_area_critica(resultado, largura, altura)

        ocorrencias += [(segundo, tipo, detalhe) for tipo, detalhe in problemas]
        gravador.write(desenhar_quadro(resultado, problemas, segundo, largura, altura,
                                       mostrar_area_critica=not fisioterapia))

    captura.release()
    gravador.release()

    eventos = agrupar_em_eventos(ocorrencias)
    agora = datetime.now()
    alertas = [
        Alerta(agora, paciente_id, "video", evento["tipo"], evento["gravidade"],
               f"{evento['detalhe']} entre {formatar_tempo(evento['inicio_segundos'])} e "
               f"{formatar_tempo(evento['fim_segundos'])} do vídeo {caminho_video.name}.")
        for evento in eventos
    ]

    print(f"  {quadros_analisados} quadros analisados, {len(eventos)} eventos fora do padrão.")
    resultado = {
        "arquivo": caminho_video.name,
        "paciente_id": paciente_id,
        "tipo_video": tipo_video,
        "modelo": config.MODELO_POSE if fisioterapia else config.MODELO_OBJETOS,
        "duracao_segundos": round(numero_quadro / fps, 1),
        "quadros_analisados": quadros_analisados,
        "eventos": eventos,
        "video_anotado": f"saida/{caminho_anotado.name}",
    }
    return resultado, alertas


def desenhar_quadro(resultado, problemas, segundo, largura, altura, mostrar_area_critica):
    """Desenha esqueleto ou objetos, a área crítica (cirurgia) e os problemas do quadro."""
    imagem = resultado.plot()

    if mostrar_area_critica:
        x1, y1, x2, y2 = area_critica_em_pixels(largura, altura)
        cv2.rectangle(imagem, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(imagem, "area critica", (x1 + 5, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.putText(imagem, formatar_tempo(segundo), (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    for linha, (tipo, detalhe) in enumerate(problemas):
        texto = sem_acentos(f"ALERTA: {tipo} ({detalhe})")
        cv2.putText(imagem, texto, (10, 55 + 28 * linha), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return imagem


def sem_acentos(texto):
    """O OpenCV não desenha acentos; troca 'ç' por 'c', 'ã' por 'a' etc."""
    return unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode("ascii")
