"""Testes da análise de áudio (sem precisar do Azure)."""

import numpy as np

from monitoramento.analise_audio import (avaliar_alteracoes_vocais, encontrar_termos_criticos,
                                         extrair_caracteristicas_voz)


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


def test_fala_lenta_gera_alteracao_vocal():
    caracteristicas = {"proporcao_pausas": 0.2, "pausas_por_minuto": 10, "palavras_por_minuto": 70}
    alteracoes = avaliar_alteracoes_vocais(caracteristicas)
    assert len(alteracoes) == 1
    assert "Fala lenta" in alteracoes[0]
