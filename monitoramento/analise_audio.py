"""
Análise de áudio de consultas médicas.

Etapas:
  1. Indicadores da voz (biblioteca librosa): pausas e ritmo da fala. Servem
     como INDÍCIOS de cansaço, dificuldade respiratória ou disartria (não é
     diagnóstico; é triagem para chamar a atenção da equipe).
  2. Transcrição com Azure Speech to Text.
  3. Sentimento e frases-chave com Azure Text Analytics (Azure AI Language).
  4. Busca de termos críticos na transcrição ("dor no peito", "falta de ar"...).

Sem chaves do Azure no .env, o sistema continua funcionando: se existir um
arquivo .txt com o mesmo nome do áudio, ele é usado como transcrição.
"""

import re
import sys
import threading
import unicodedata
from datetime import datetime
from pathlib import Path

import librosa
import soundfile

from monitoramento import config
from monitoramento.alertas import Alerta

TAXA_AMOSTRAGEM = 16000   # 16 kHz mono: formato esperado pelo Azure Speech


# ---------------------------------------------------------------------------
# 1. Indicadores da voz
# ---------------------------------------------------------------------------
def extrair_caracteristicas_voz(sinal, taxa):
    duracao = len(sinal) / taxa

    # Trechos com fala: tudo que está 30 dB abaixo do pico é considerado silêncio
    trechos_fala = librosa.effects.split(sinal, top_db=30)
    tempo_fala = sum(fim - inicio for inicio, fim in trechos_fala) / taxa

    # Pausa = espaço de pelo menos 0,25 s entre dois trechos de fala
    pausas = sum(1 for (_, fim_anterior), (inicio_proximo, _) in zip(trechos_fala, trechos_fala[1:])
                 if (inicio_proximo - fim_anterior) / taxa >= 0.25)

    minutos = max(duracao / 60, 1e-6)
    return {
        "duracao_segundos": round(duracao, 1),
        "tempo_de_fala_segundos": round(tempo_fala, 1),
        "proporcao_pausas": round(1 - tempo_fala / duracao, 2) if duracao > 0 else 0,
        "pausas_por_minuto": round(pausas / minutos, 1),
    }


def extrair_caracteristicas_voz_falante(sinal, taxa, falas, speaker_id):
    intervalos = []
    for fala in falas:
        if fala.get("speaker_id") != speaker_id:
            continue
        try:
            inicio = float(fala["inicio_segundos"])
            fim = inicio + float(fala["duracao_segundos"])
        except (KeyError, TypeError, ValueError):
            continue

        inicio_amostra = max(0, round(inicio * taxa))
        fim_amostra = min(len(sinal), round(fim * taxa))
        if fim_amostra > inicio_amostra:
            intervalos.append((inicio_amostra, fim_amostra))

    if not intervalos:
        return None

    intervalos.sort()
    intervalos_unidos = []
    for inicio, fim in intervalos:
        if intervalos_unidos and inicio <= intervalos_unidos[-1][1]:
            anterior_inicio, anterior_fim = intervalos_unidos[-1]
            intervalos_unidos[-1] = (anterior_inicio, max(anterior_fim, fim))
        else:
            intervalos_unidos.append((inicio, fim))

    duracao = 0
    tempo_fala = 0
    pausas = 0
    for inicio, fim in intervalos_unidos:
        trecho = sinal[inicio:fim]
        duracao += len(trecho)
        trechos_fala = librosa.effects.split(trecho, top_db=30)
        tempo_fala += sum(fim_trecho - inicio_trecho
                          for inicio_trecho, fim_trecho in trechos_fala)
        pausas += sum(1 for (_, fim_anterior), (inicio_proximo, _) in
                      zip(trechos_fala, trechos_fala[1:])
                      if (inicio_proximo - fim_anterior) / taxa >= 0.25)

    duracao_segundos = duracao / taxa
    minutos = max(duracao_segundos / 60, 1e-6)
    tempo_fala_segundos = tempo_fala / taxa
    return {
        "duracao_segundos": round(duracao_segundos, 1),
        "tempo_de_fala_segundos": round(tempo_fala_segundos, 1),
        "proporcao_pausas": round(1 - tempo_fala / duracao, 2) if duracao else 0,
        "pausas_por_minuto": round(pausas / minutos, 1),
    }


def avaliar_alteracoes_vocais(caracteristicas):
    """Compara os indicadores com os limites do config e devolve uma lista
    de textos descrevendo as alterações encontradas."""
    alteracoes = []

    if caracteristicas["proporcao_pausas"] > config.LIMITE_PROPORCAO_PAUSAS:
        alteracoes.append(f"Muitas pausas ({caracteristicas['proporcao_pausas']:.0%} do áudio em silêncio) "
                          "- possível cansaço.")

    if caracteristicas["pausas_por_minuto"] > config.LIMITE_PAUSAS_POR_MINUTO:
        alteracoes.append(f"Fala entrecortada ({caracteristicas['pausas_por_minuto']} pausas/min) "
                          "- possível dificuldade respiratória.")

    ppm = caracteristicas.get("palavras_por_minuto")
    if ppm is not None and ppm < config.LIMITE_PALAVRAS_POR_MINUTO:
        alteracoes.append(f"Fala lenta ({ppm} palavras/min) - possível fadiga ou disartria.")

    return alteracoes


# ---------------------------------------------------------------------------
# 2. Transcrição (Azure Speech to Text)
# ---------------------------------------------------------------------------
def transcrever_com_azure(caminho_wav):
    import azure.cognitiveservices.speech as speechsdk

    configuracao = speechsdk.SpeechConfig(subscription=config.AZURE_SPEECH_KEY,
                                          region=config.AZURE_SPEECH_REGION)
    configuracao.speech_recognition_language = config.IDIOMA
    configuracao.set_property(
        speechsdk.PropertyId.SpeechServiceResponse_DiarizeIntermediateResults,
        "true",
    )
    audio = speechsdk.audio.AudioConfig(filename=str(caminho_wav))
    transcritor = speechsdk.transcription.ConversationTranscriber(
        speech_config=configuracao,
        audio_config=audio,
    )

    falas = []
    erros = []
    terminou = threading.Event()

    def ao_transcrever(evento):
        if evento.result.reason == speechsdk.ResultReason.RecognizedSpeech:
            falas.append({
                "speaker_id": evento.result.speaker_id or "Unknown",
                "texto": evento.result.text,
                "inicio_segundos": evento.result.offset / 10_000_000,
                "duracao_segundos": evento.result.duration / 10_000_000,
            })

    def ao_cancelar(evento):
        detalhes = getattr(evento, "cancellation_details", None)
        motivo = getattr(evento, "reason", None)
        if detalhes:
            motivo = detalhes.reason
        if motivo == speechsdk.CancellationReason.Error:
            erros.append(getattr(detalhes, "error_details", None)
                         or getattr(evento, "error_details", "Erro desconhecido"))
        terminou.set()

    transcritor.transcribed.connect(ao_transcrever)
    transcritor.session_stopped.connect(lambda evento: terminou.set())
    transcritor.canceled.connect(ao_cancelar)

    transcritor.start_transcribing_async()
    if not terminou.wait(timeout=900):
        transcritor.stop_transcribing_async()
        raise TimeoutError("A diarização do áudio excedeu o limite de 15 minutos.")
    transcritor.stop_transcribing_async()

    if erros:
        raise RuntimeError(f"Erro no Azure Speech: {erros[0]}")
    return falas

def transcrever_com_diarizacao_api(caminho_wav, max_falantes=2):
    import json

    import requests

    endpoint = config.AZURE_SPEECH_ENDPOINT.strip().rstrip("/")
    if not endpoint:
        if not config.AZURE_SPEECH_REGION:
            raise ValueError("Configure AZURE_SPEECH_ENDPOINT ou AZURE_SPEECH_REGION no .env.")
        endpoint = f"https://{config.AZURE_SPEECH_REGION}.api.cognitive.microsoft.com"

    url = f"{endpoint}/speechtotext/transcriptions:transcribe?api-version=2025-10-15"
    definicao = {
        "locales": [config.IDIOMA],
        "diarization": {"enabled": True, "maxSpeakers": max_falantes},
    }

    with Path(caminho_wav).open("rb") as arquivo_audio:
        resposta = requests.post(
            url,
            headers={"Ocp-Apim-Subscription-Key": config.AZURE_SPEECH_KEY},
            files={"audio": (Path(caminho_wav).name, arquivo_audio, "audio/wav")},
            data={"definition": json.dumps(definicao)},
            timeout=300,
        )

    resposta.raise_for_status()
    resultado = resposta.json()
    frases = resultado.get("phrases")
    if not isinstance(frases, list):
        raise RuntimeError("A resposta da Fast Transcription API não contém a lista 'phrases'.")

    falas = []
    for frase in frases:
        speaker = frase.get("speaker")
        if speaker is None:
            speaker_id = "Unknown"
        elif isinstance(speaker, int) or str(speaker).isdigit():
            speaker_id = f"Guest-{int(speaker) + 1}"
        else:
            speaker_id = f"Guest-{speaker}"

        falas.append({
            "speaker_id": speaker_id,
            "texto": frase.get("text", ""),
            "inicio_segundos": float(frase.get("offsetMilliseconds", 0)) / 1000,
            "duracao_segundos": float(frase.get("durationMilliseconds", 0)) / 1000,
        })

    return falas


def obter_transcricao(caminho_audio, caminho_wav):
    if config.azure_speech_api_configurado():
        print("  Transcrevendo com diarização pela Fast Transcription API...")
        return transcrever_com_diarizacao_api(caminho_wav), "Azure Fast Transcription API"

    if config.azure_speech_configurado():
        print("  Transcrevendo e separando falantes com Azure Speech to Text...")
        return transcrever_com_azure(caminho_wav), "Azure Fast Transcription API"

    arquivo_texto = Path(caminho_audio).with_suffix(".txt")
    if arquivo_texto.exists():
        print(f"  Azure Speech não configurado. Usando transcrição do arquivo {arquivo_texto.name}.")
        return arquivo_texto.read_text(encoding="utf-8"), f"arquivo {arquivo_texto.name}"

    print("  Azure Speech não configurado e sem arquivo .txt: áudio analisado sem transcrição.")
    return "", "sem transcrição"


# ---------------------------------------------------------------------------
# 3. Sentimento e frases-chave (Azure Text Analytics)
# ---------------------------------------------------------------------------
def dividir_em_frases(texto):
    frases = re.split(r"(?<=[.!?])\s+", texto.strip())
    return [frase for frase in frases if len(frase) > 3]


def analisar_texto_com_azure(texto):
    from azure.ai.textanalytics import TextAnalyticsClient
    from azure.core.credentials import AzureKeyCredential

    cliente = TextAnalyticsClient(endpoint=config.AZURE_LANGUAGE_ENDPOINT,
                                  credential=AzureKeyCredential(config.AZURE_LANGUAGE_KEY))
    frases = dividir_em_frases(texto)
    contagem = {"positive": 0, "neutral": 0, "negative": 0, "mixed": 0}
    frases_negativas = []
    frases_chave = set()

    # Cada frase é um "documento"; o serviço aceita no máximo 10 por chamada
    for inicio in range(0, len(frases), 10):
        lote = frases[inicio:inicio + 10]

        for frase, analise in zip(lote, cliente.analyze_sentiment(lote, language=config.IDIOMA)):
            if analise.is_error:
                continue
            contagem[analise.sentiment] += 1
            if analise.sentiment == "negative":
                frases_negativas.append({"frase": frase, "confianca": round(analise.confidence_scores.negative, 2)})

        for analise in cliente.extract_key_phrases(lote, language=config.IDIOMA):
            if not analise.is_error:
                frases_chave.update(analise.key_phrases)

    total = sum(contagem.values()) or 1
    if contagem["negative"] / total >= 0.5:
        sentimento_geral = "negativo"
    elif contagem["positive"] / total >= 0.5:
        sentimento_geral = "positivo"
    else:
        sentimento_geral = "neutro/misto"

    return {"sentimento_geral": sentimento_geral, "contagem": contagem,
            "frases_negativas": frases_negativas, "frases_chave": sorted(frases_chave)}


# ---------------------------------------------------------------------------
# 4. Termos críticos
# ---------------------------------------------------------------------------
def normalizar_texto(texto):
    """Minúsculas e sem acentos: 'Convulsão' -> 'convulsao'."""
    sem_acentos = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode("ascii")
    return sem_acentos.lower()


def encontrar_termos_criticos(texto):
    texto_normalizado = normalizar_texto(texto)
    return [{"termo": termo, "gravidade": gravidade}
            for termo, gravidade in config.TERMOS_CRITICOS.items()
            if re.search(rf"\b{re.escape(normalizar_texto(termo))}", texto_normalizado)]


# ---------------------------------------------------------------------------
# Função principal
# ---------------------------------------------------------------------------
def analisar_audio(caminho_audio, paciente_id, falante_paciente=None):
    caminho_audio = Path(caminho_audio)
    print(f"Analisando áudio: {caminho_audio.name}")

    # Carrega e converte para WAV 16 kHz mono (formato aceito pelo Azure)
    sinal, taxa = librosa.load(caminho_audio, sr=TAXA_AMOSTRAGEM, mono=True)
    config.PASTA_SAIDA.mkdir(parents=True, exist_ok=True)
    caminho_wav = config.PASTA_SAIDA / f"{caminho_audio.stem}_16khz.wav"
    soundfile.write(caminho_wav, sinal, taxa, subtype="PCM_16")

    caracteristicas = extrair_caracteristicas_voz(sinal, taxa)
    transcricao, fonte_transcricao = obter_transcricao(caminho_audio, caminho_wav)

    falas = transcricao if isinstance(transcricao, list) else []
    transcricao = (" ".join(fala["texto"] for fala in falas)
                   if falas else transcricao)
 
    id_falante_paciente = selecionar_falante_paciente(falas, falante_paciente)
    texto_paciente = texto_do_falante(falas, id_falante_paciente)

    caracteristicas_falante = (extrair_caracteristicas_voz_falante(
        sinal, taxa, falas, id_falante_paciente) if id_falante_paciente else None)
    
    metricas_vocais_atribuiveis = bool(texto_paciente) and caracteristicas_falante is not None

    if metricas_vocais_atribuiveis:
        caracteristicas = caracteristicas_falante
        minutos_de_fala = max(caracteristicas["tempo_de_fala_segundos"] / 60, 1e-6)
        caracteristicas["palavras_por_minuto"] = round(len(texto_paciente.split()) / minutos_de_fala)

    alteracoes = (avaliar_alteracoes_vocais(caracteristicas)
                  if metricas_vocais_atribuiveis else [])
    
    termos = encontrar_termos_criticos(transcricao)

    analise_texto = None
    if texto_paciente and config.azure_language_configurado():
        print("  Analisando sentimento e frases-chave com Azure Text Analytics...")
        analise_texto = analisar_texto_com_azure(texto_paciente)
    elif transcricao and not texto_paciente:
        print("  Falante do paciente não identificado: termos e sentimento não serão atribuídos.")
    elif texto_paciente:
        print("  Azure Text Analytics não configurado: sentimento não analisado.")
    if texto_paciente and not metricas_vocais_atribuiveis:
        print("  Métricas vocais não atribuídas: sem intervalos diarizados válidos para o falante escolhido.")

    # Alertas
    agora = datetime.now()
    alertas = [Alerta(agora, paciente_id, "audio", "Alteração vocal", "media",
                      f"{alteracao} (áudio {caminho_audio.name})")
               for alteracao in alteracoes]
    alertas += [Alerta(agora, paciente_id, "audio", "Termo crítico na consulta", item["gravidade"],
                       f"Paciente mencionou '{item['termo']}' (áudio {caminho_audio.name}).")
                for item in termos]
    if analise_texto and analise_texto["sentimento_geral"] == "negativo":
        alertas.append(Alerta(agora, paciente_id, "audio", "Sentimento negativo predominante", "baixa",
                              f"{analise_texto['contagem']['negative']} frases negativas na consulta."))

    print(f"  {len(alteracoes)} alterações vocais, {len(termos)} termos críticos.")
    resultado = {
        "arquivo": caminho_audio.name,
        "paciente_id": paciente_id,
        "caracteristicas": caracteristicas,
        "alteracoes_vocais": alteracoes,
        "fonte_transcricao": fonte_transcricao,
        "transcricao": transcricao,
        "falas": falas,
        "falante_paciente": id_falante_paciente,
        "transcricao_paciente": texto_paciente,
        "metricas_vocais_atribuiveis": metricas_vocais_atribuiveis,
        "termos_criticos": termos,
        "analise_texto": analise_texto,
    }
    return resultado, alertas


def selecionar_falante_paciente(falas, escolha=None):
    """Retorna o ID confirmado do paciente; nunca infere pelo Guest-N."""
    falantes = sorted({fala["speaker_id"] for fala in falas
                       if fala["speaker_id"] and fala["speaker_id"] != "Unknown"})
    if escolha:
        if escolha not in falantes:
            raise ValueError(f"Falante {escolha!r} não encontrado. IDs disponíveis: {falantes}")
        return escolha
    if not falantes or not sys.stdin.isatty():
        return None

    print("  Falantes identificados (confirme qual é o paciente):")
    for falante in falantes:
        exemplos = [fala["texto"] for fala in falas if fala["speaker_id"] == falante]
        resumo = " ".join(exemplos)[:160]
        print(f"    {falante}: {resumo}")

    while True:
        escolha = input("  ID do paciente (ou Enter para não atribuir): ").strip()
        if not escolha:
            return None
        if escolha in falantes:
            return escolha
        print(f"  ID inválido. Escolha um destes: {', '.join(falantes)}")

def texto_do_falante(falas, speaker_id):
    if not speaker_id:
        return ""
    return " ".join(fala["texto"] for fala in falas
                    if fala["speaker_id"] == speaker_id)