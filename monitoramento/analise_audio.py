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
    audio = speechsdk.audio.AudioConfig(filename=str(caminho_wav))
    reconhecedor = speechsdk.SpeechRecognizer(speech_config=configuracao, audio_config=audio)

    frases = []
    erros = []
    terminou = threading.Event()

    def ao_reconhecer_frase(evento):
        if evento.result.reason == speechsdk.ResultReason.RecognizedSpeech:
            frases.append(evento.result.text)

    def ao_cancelar(evento):
        if evento.cancellation_details.reason == speechsdk.CancellationReason.Error:
            erros.append(evento.cancellation_details.error_details)
        terminou.set()

    reconhecedor.recognized.connect(ao_reconhecer_frase)
    reconhecedor.session_stopped.connect(lambda evento: terminou.set())
    reconhecedor.canceled.connect(ao_cancelar)

    # Reconhecimento contínuo: o "recognize_once" para na primeira frase (~30 s)
    reconhecedor.start_continuous_recognition()
    terminou.wait(timeout=900)
    reconhecedor.stop_continuous_recognition()

    if erros:
        raise RuntimeError(f"Erro no Azure Speech: {erros[0]}")
    return " ".join(frases)


def obter_transcricao(caminho_audio, caminho_wav):
    if config.azure_speech_configurado():
        print("  Transcrevendo com Azure Speech to Text...")
        return transcrever_com_azure(caminho_wav), "Azure Speech to Text"

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
def analisar_audio(caminho_audio, paciente_id):
    caminho_audio = Path(caminho_audio)
    print(f"Analisando áudio: {caminho_audio.name}")

    # Carrega e converte para WAV 16 kHz mono (formato aceito pelo Azure)
    sinal, taxa = librosa.load(caminho_audio, sr=TAXA_AMOSTRAGEM, mono=True)
    config.PASTA_SAIDA.mkdir(parents=True, exist_ok=True)
    caminho_wav = config.PASTA_SAIDA / f"{caminho_audio.stem}_16khz.wav"
    soundfile.write(caminho_wav, sinal, taxa, subtype="PCM_16")

    caracteristicas = extrair_caracteristicas_voz(sinal, taxa)
    transcricao, fonte_transcricao = obter_transcricao(caminho_audio, caminho_wav)

    if transcricao:
        minutos_de_fala = max(caracteristicas["tempo_de_fala_segundos"] / 60, 1e-6)
        caracteristicas["palavras_por_minuto"] = round(len(transcricao.split()) / minutos_de_fala)

    alteracoes = avaliar_alteracoes_vocais(caracteristicas)
    termos = encontrar_termos_criticos(transcricao)

    analise_texto = None
    if transcricao and config.azure_language_configurado():
        print("  Analisando sentimento e frases-chave com Azure Text Analytics...")
        analise_texto = analisar_texto_com_azure(transcricao)
    elif transcricao:
        print("  Azure Text Analytics não configurado: sentimento não analisado.")

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
        "termos_criticos": termos,
        "analise_texto": analise_texto,
    }
    return resultado, alertas
