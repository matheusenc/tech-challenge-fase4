"""Testes da detecção de anomalias em prescrições."""

import pandas as pd

from monitoramento.anomalias_prescricoes import detectar_anomalias_prescricoes


def salvar_csv(pasta, linhas):
    caminho = pasta / "prescricoes.csv"
    pd.DataFrame(linhas, columns=["data_hora", "paciente_id", "medicamento", "dose", "unidade"]) \
        .to_csv(caminho, index=False)
    return caminho


def test_dose_acima_do_maximo(tmp_path):
    caminho = salvar_csv(tmp_path, [["2026-10-01 08:00", "P001", "Dipirona", 3000, "mg"]])
    tipos = [a.tipo for a in detectar_anomalias_prescricoes(caminho)]
    assert "Dose acima do máximo" in tipos


def test_dose_duplicada_e_aumento_brusco(tmp_path):
    caminho = salvar_csv(tmp_path, [
        ["2026-10-01 08:00", "P001", "morfina", 2, "mg"],
        ["2026-10-01 09:00", "P001", "morfina", 5, "mg"],
    ])
    tipos = [a.tipo for a in detectar_anomalias_prescricoes(caminho)]
    assert "Intervalo entre doses muito curto" in tipos
    assert "Aumento brusco de dose" in tipos


def test_prescricao_normal_sem_alertas(tmp_path):
    caminho = salvar_csv(tmp_path, [
        ["2026-10-01 08:00", "P001", "dipirona", 1000, "mg"],
        ["2026-10-01 14:00", "P001", "dipirona", 1000, "mg"],
    ])
    assert detectar_anomalias_prescricoes(caminho) == []
