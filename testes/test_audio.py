"""Testes da análise de áudio (sem precisar do Azure)."""

import numpy as np

from monitoramento.analise_audio import (avaliar_alteracoes_vocais, encontrar_termos_criticos,
                                         extrair_caracteristicas_voz,
                                         extrair_caracteristicas_voz_falante,
                                         transcrever_com_diarizacao_api)


def test_termos_criticos_ignoram_acentos_e_maiusculas():
    termos = [t["termo"] for t in encontrar_termos_criticos("Sinto FALTA DE AR e muito cansaco.")]
    assert "falta de ar" in termos
    assert "cansaço" in termos


def test_texto_sem_termos_criticos():
    assert encontrar_termos_criticos("Estou me sentindo bem hoje.") == []


def test_audio_com_muito_silencio_tem_muitas_pausas():
    taxa = 16000
    tempo = np.arange(taxa) / taxa
    tom = 0.3 * np.sin(2 * np.pi * 150 * tempo)   # 1 s de "voz"
    silencio = np.zeros(taxa * 2)                 # 2 s de silêncio
    sinal = np.concatenate([tom, silencio, tom, silencio, tom]).astype(np.float32)

    caracteristicas = extrair_caracteristicas_voz(sinal, taxa)
    assert caracteristicas["proporcao_pausas"] > 0.5
    assert 15 < caracteristicas["pausas_por_minuto"] < 20   # 2 pausas em 7 s ≈ 17 por minuto


def test_caracteristicas_vocais_usam_apenas_intervalos_do_falante():
    taxa = 16000
    tempo = np.arange(taxa * 6) / taxa
    sinal = np.zeros(taxa * 6, dtype=np.float32)
    tom = 0.3 * np.sin(2 * np.pi * 150 * tempo[:taxa])
    sinal[:taxa] = tom
    sinal[taxa:taxa * 3] = np.tile(tom, 2)
    sinal[taxa * 3:taxa * 4] = tom
    sinal[taxa * 4:] = np.tile(tom, 2)
    falas = [
        {"speaker_id": "Guest-1", "inicio_segundos": 0, "duracao_segundos": 1},
        {"speaker_id": "Guest-2", "inicio_segundos": 1, "duracao_segundos": 2},
        {"speaker_id": "Guest-1", "inicio_segundos": 3, "duracao_segundos": 1},
        {"speaker_id": "Guest-2", "inicio_segundos": 4, "duracao_segundos": 2},
    ]

    caracteristicas = extrair_caracteristicas_voz_falante(sinal, taxa, falas, "Guest-1")

    assert caracteristicas["duracao_segundos"] == 2.0
    assert caracteristicas["tempo_de_fala_segundos"] > 1.8
    assert caracteristicas["proporcao_pausas"] < 0.1


def test_fala_lenta_gera_alteracao_vocal():
    caracteristicas = {"proporcao_pausas": 0.2, "pausas_por_minuto": 10, "palavras_por_minuto": 70}
    alteracoes = avaliar_alteracoes_vocais(caracteristicas)
    assert len(alteracoes) == 1
    assert "Fala lenta" in alteracoes[0]


def test_api_fast_transcription_limita_falantes_e_normaliza_resultado(tmp_path, monkeypatch):
    import json

    import requests

    from monitoramento import config

    arquivo_audio = tmp_path / "consulta.wav"
    arquivo_audio.write_bytes(b"wav-data")
    chamada = {}

    class Resposta:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "phrases": [
                    {"speaker": 0, "offsetMilliseconds": 1250,
                     "durationMilliseconds": 800, "text": "Bom dia."},
                    {"speaker": 1, "offsetMilliseconds": 2300,
                     "durationMilliseconds": 500, "text": "Olá."},
                ]
            }

    def post(url, **kwargs):
        chamada["url"] = url
        chamada.update(kwargs)
        return Resposta()

    monkeypatch.setattr(config, "AZURE_SPEECH_ENDPOINT", "https://speech.example.test/")
    monkeypatch.setattr(config, "AZURE_SPEECH_KEY", "test-key")
    monkeypatch.setattr(requests, "post", post)

    falas = transcrever_com_diarizacao_api(arquivo_audio)

    definicao = json.loads(chamada["data"]["definition"])
    assert definicao["diarization"] == {"enabled": True, "maxSpeakers": 2}
    assert definicao["locales"] == [config.IDIOMA]
    assert chamada["url"].startswith("https://speech.example.test/")
    assert falas == [
        {"speaker_id": "Guest-1", "texto": "Bom dia.",
         "inicio_segundos": 1.25, "duracao_segundos": 0.8},
        {"speaker_id": "Guest-2", "texto": "Olá.",
         "inicio_segundos": 2.3, "duracao_segundos": 0.5},
    ]
