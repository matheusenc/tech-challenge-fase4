"""Testes da detecção de anomalias de movimentação."""

import pandas as pd

from monitoramento.anomalias_movimentacao import detectar_anomalias_movimentacao


def salvar_csv(pasta, atividades, inicio="2026-10-01 08:00"):
    horarios = pd.date_range(inicio, periods=len(atividades), freq="h")
    caminho = pasta / "movimentacao.csv"
    pd.DataFrame({"data_hora": horarios, "paciente_id": "P001", "nivel_atividade": atividades}) \
        .to_csv(caminho, index=False)
    return caminho


def test_inatividade_prolongada(tmp_path):
    caminho = salvar_csv(tmp_path, [30, 30, 0, 0, 0, 0, 30])
    tipos = [a.tipo for a in detectar_anomalias_movimentacao(caminho)]
    assert tipos == ["Inatividade prolongada"]


def test_agitacao_noturna(tmp_path):
    caminho = salvar_csv(tmp_path, [10, 80, 10], inicio="2026-10-01 01:00")
    tipos = [a.tipo for a in detectar_anomalias_movimentacao(caminho)]
    assert tipos == ["Agitação noturna"]
