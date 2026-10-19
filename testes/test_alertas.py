"""Testes da remoção de repetidos e da fusão de alertas por paciente."""

from datetime import datetime, timedelta

from monitoramento.alertas import Alerta, remover_repetidos, resumo_por_paciente

INICIO = datetime(2026, 10, 1, 8, 0)


def alerta(paciente, origem, gravidade, minutos=0, tipo="Teste"):
    return Alerta(INICIO + timedelta(minutes=minutos), paciente, origem, tipo, gravidade, "descrição")


def test_mesmo_alerta_em_pouco_tempo_e_removido():
    alertas = [alerta("P001", "sinais_vitais", "alta", 0), alerta("P001", "sinais_vitais", "alta", 5),
               alerta("P001", "sinais_vitais", "alta", 60)]
    assert len(remover_repetidos(alertas, janela_minutos=30)) == 2


def test_tempo_real_lembra_o_que_ja_foi_enviado():
    ultimo_envio = {}
    assert len(remover_repetidos([alerta("P001", "sinais_vitais", "alta", 0)], ultimo_envio=ultimo_envio)) == 1
    assert remover_repetidos([alerta("P001", "sinais_vitais", "alta", 5)], ultimo_envio=ultimo_envio) == []


def test_fusao_prioriza_paciente_com_mais_fontes():
    alertas = [
        alerta("P001", "sinais_vitais", "alta"), alerta("P001", "audio", "media"),   # 2 fontes, 5 pontos
        alerta("P002", "prescricoes", "alta"), alerta("P002", "prescricoes", "alta", tipo="Outro"),  # 1 fonte, 6
    ]
    resumo = resumo_por_paciente(alertas)
    assert [r["paciente_id"] for r in resumo] == ["P001", "P002"]
    assert resumo[0]["fontes"] == ["audio", "sinais_vitais"]
    assert resumo[0]["pontuacao"] == 5 and resumo[1]["pontuacao"] == 6
