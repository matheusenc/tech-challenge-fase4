"""Testes da detecção de anomalias em sinais vitais.
Rodar com:  python -m pytest testes/"""

import pandas as pd

from monitoramento.anomalias_sinais_vitais import verificar_leitura


def criar_leitura(minuto, fc=75, sistolica=120, diastolica=80, saturacao=97):
    return {
        "data_hora": pd.Timestamp("2026-10-01") + pd.Timedelta(minutes=minuto),
        "paciente_id": "P001",
        "frequencia_cardiaca": fc,
        "pressao_sistolica": sistolica,
        "pressao_diastolica": diastolica,
        "saturacao_oxigenio": saturacao,
    }


def test_leitura_normal_nao_gera_alerta():
    assert verificar_leitura(criar_leitura(0), historico=[]) == []


def test_saturacao_baixa_gera_alerta_grave():
    alertas = verificar_leitura(criar_leitura(0, saturacao=85), historico=[])
    assert len(alertas) == 1
    assert alertas[0].gravidade == "alta"
    assert "Saturação" in alertas[0].tipo


def test_salto_de_frequencia_dentro_do_limite_gera_variacao_brusca():
    historico = [criar_leitura(i * 5, fc=74 + (i % 3)) for i in range(12)]
    alertas = verificar_leitura(criar_leitura(60, fc=105), historico)
    assert len(alertas) == 1
    assert alertas[0].tipo.startswith("Variação brusca")


def test_pequena_oscilacao_nao_gera_alerta():
    historico = [criar_leitura(i * 5, fc=74 + (i % 3)) for i in range(12)]
    assert verificar_leitura(criar_leitura(60, fc=80), historico) == []
