"""
Detecção de anomalias nos padrões de movimentação do paciente durante a
internação.

Arquivo de entrada (CSV), uma linha por hora:
    data_hora, paciente_id, nivel_atividade
`nivel_atividade` = quantidade de movimentos registrada por um sensor
(acelerômetro na pulseira, sensor de pressão no leito, etc.).

Regras:
  1. Inatividade prolongada: paciente parado por várias horas seguidas
     (risco de lesão por pressão / piora do estado geral).
  2. Agitação noturna: muita atividade de madrugada (risco de queda,
     possível delirium).
  3. Queda de mobilidade: média do dia caiu muito em relação ao dia
     anterior (possível piora clínica).
"""

import pandas as pd

from monitoramento import config
from monitoramento.alertas import Alerta, remover_repetidos


def carregar_movimentacao(caminho_csv):
    tabela = pd.read_csv(caminho_csv, parse_dates=["data_hora"])
    tabela["paciente_id"] = tabela["paciente_id"].astype(str)
    return tabela.sort_values(["paciente_id", "data_hora"]).reset_index(drop=True)


def verificar_inatividade(tabela):
    alertas = []
    for paciente_id, leituras in tabela.groupby("paciente_id"):
        horas_paradas = 0
        inicio_parado = None
        for _, linha in leituras.iterrows():
            if linha["nivel_atividade"] < config.LIMITE_INATIVIDADE:
                if horas_paradas == 0:
                    inicio_parado = linha["data_hora"]
                horas_paradas += 1
                # Alerta exatamente quando atinge o limite (não repete a cada hora)
                if horas_paradas == config.HORAS_INATIVIDADE_PARA_ALERTA:
                    alertas.append(Alerta(
                        data_hora=linha["data_hora"],
                        paciente_id=str(paciente_id),
                        origem="movimentacao",
                        tipo="Inatividade prolongada",
                        gravidade="media",
                        descricao=(f"Sem movimentação desde {inicio_parado:%d/%m %H:%M} "
                                   f"({horas_paradas}h). Avaliar mudança de decúbito e estado geral."),
                    ))
            else:
                horas_paradas = 0
    return alertas


def eh_noite(data_hora):
    return data_hora.hour >= config.HORA_INICIO_NOITE or data_hora.hour < config.HORA_FIM_NOITE


def verificar_agitacao_noturna(tabela):
    alertas = []
    for _, linha in tabela.iterrows():
        if eh_noite(linha["data_hora"]) and linha["nivel_atividade"] > config.LIMITE_AGITACAO_NOTURNA:
            alertas.append(Alerta(
                data_hora=linha["data_hora"],
                paciente_id=linha["paciente_id"],
                origem="movimentacao",
                tipo="Agitação noturna",
                gravidade="media",
                descricao=(f"Atividade {linha['nivel_atividade']:.0f} durante a noite "
                           f"(limite {config.LIMITE_AGITACAO_NOTURNA}). Risco de queda; avaliar delirium."),
            ))
    return alertas


def verificar_queda_de_mobilidade(tabela):
    alertas = []
    tabela = tabela.copy()
    tabela["dia"] = tabela["data_hora"].dt.date

    for paciente_id, leituras in tabela.groupby("paciente_id"):
        media_por_dia = leituras.groupby("dia")["nivel_atividade"].mean()
        dias = list(media_por_dia.index)
        for dia_anterior, dia in zip(dias, dias[1:]):
            anterior, atual = media_por_dia[dia_anterior], media_por_dia[dia]
            if anterior <= 0:
                continue
            queda = (anterior - atual) / anterior * 100
            if queda > config.QUEDA_MOBILIDADE_PERCENTUAL:
                alertas.append(Alerta(
                    data_hora=pd.Timestamp(dia) + pd.Timedelta(hours=23, minutes=59),
                    paciente_id=str(paciente_id),
                    origem="movimentacao",
                    tipo="Queda de mobilidade",
                    gravidade="media",
                    descricao=(f"Atividade média caiu {queda:.0f}% em {dia:%d/%m} "
                               f"({anterior:.1f} -> {atual:.1f}). Possível piora clínica."),
                ))
    return alertas


def detectar_anomalias_movimentacao(caminho_csv):
    tabela = carregar_movimentacao(caminho_csv)
    alertas = (verificar_inatividade(tabela)
               + verificar_agitacao_noturna(tabela)
               + verificar_queda_de_mobilidade(tabela))
    alertas = remover_repetidos(alertas, janela_minutos=180)
    print(f"Movimentação: {len(tabela)} leituras analisadas, {len(alertas)} alertas.")
    return alertas
