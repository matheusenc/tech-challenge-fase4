"""
Detecção de anomalias na evolução das prescrições (alterações inesperadas
no tratamento).

Arquivo de entrada (CSV), uma linha por administração de medicamento:
    data_hora, paciente_id, medicamento, dose, unidade

Regras:
  1. Dose acima do máximo permitido por administração.
  2. Aumento brusco: dose muito maior que a dose anterior do mesmo remédio.
  3. Intervalo curto: duas doses do mesmo remédio muito próximas
     (possível dose duplicada).
  4. Medicamento de alta vigilância introduzido no meio da internação.
"""

import pandas as pd

from monitoramento import config
from monitoramento.alertas import Alerta


def carregar_prescricoes(caminho_csv):
    tabela = pd.read_csv(caminho_csv, parse_dates=["data_hora"])
    tabela["paciente_id"] = tabela["paciente_id"].astype(str)
    tabela["medicamento"] = tabela["medicamento"].str.strip().str.lower()
    return tabela.sort_values(["paciente_id", "data_hora"]).reset_index(drop=True)


def novo_alerta(linha, tipo, gravidade, descricao):
    return Alerta(
        data_hora=linha["data_hora"],
        paciente_id=linha["paciente_id"],
        origem="prescricoes",
        tipo=tipo,
        gravidade=gravidade,
        descricao=descricao,
    )


def verificar_dose_maxima(tabela):
    alertas = []
    for _, linha in tabela.iterrows():
        maximo = config.DOSE_MAXIMA.get(linha["medicamento"])
        if maximo is not None and linha["dose"] > maximo:
            alertas.append(novo_alerta(
                linha, "Dose acima do máximo", "alta",
                f"{linha['medicamento']} {linha['dose']:g} {linha['unidade']} "
                f"(máximo por dose: {maximo:g} {linha['unidade']}).",
            ))
    return alertas


def verificar_aumento_brusco_e_intervalo(tabela):
    alertas = []
    for (_, medicamento), doses in tabela.groupby(["paciente_id", "medicamento"]):
        anterior = None
        for _, linha in doses.iterrows():
            if anterior is not None:
                # Regra 2: aumento brusco de dose
                if linha["dose"] >= anterior["dose"] * config.FATOR_AUMENTO_BRUSCO:
                    alertas.append(novo_alerta(
                        linha, "Aumento brusco de dose", "media",
                        f"{medicamento}: dose passou de {anterior['dose']:g} para "
                        f"{linha['dose']:g} {linha['unidade']}.",
                    ))

                # Regra 3: intervalo menor que o mínimo
                horas = (linha["data_hora"] - anterior["data_hora"]).total_seconds() / 3600
                minimo = config.INTERVALO_MINIMO_HORAS.get(medicamento, config.INTERVALO_MINIMO_PADRAO)
                if horas < minimo:
                    alertas.append(novo_alerta(
                        linha, "Intervalo entre doses muito curto", "media",
                        f"{medicamento} administrado {horas:.1f}h após a dose anterior "
                        f"(mínimo {minimo}h). Possível dose duplicada.",
                    ))
            anterior = linha
    return alertas


def verificar_alta_vigilancia(tabela):
    """Remédios de alta vigilância que NÃO estavam na prescrição inicial
    (primeiras 12 horas do paciente) e aparecem depois."""
    alertas = []
    for _, prescricoes_paciente in tabela.groupby("paciente_id"):
        inicio_internacao = prescricoes_paciente["data_hora"].min()
        primeira_dose = prescricoes_paciente.groupby("medicamento")["data_hora"].min()

        for medicamento, data_primeira in primeira_dose.items():
            introduzido_depois = data_primeira - inicio_internacao > pd.Timedelta(hours=12)
            if medicamento in config.MEDICAMENTOS_ALTA_VIGILANCIA and introduzido_depois:
                linha = prescricoes_paciente[
                    (prescricoes_paciente["medicamento"] == medicamento)
                    & (prescricoes_paciente["data_hora"] == data_primeira)
                ].iloc[0]
                alertas.append(novo_alerta(
                    linha, "Medicamento de alta vigilância introduzido", "media",
                    f"{medicamento} iniciado durante a internação ({linha['dose']:g} "
                    f"{linha['unidade']}). Conferir indicação e monitorar.",
                ))
    return alertas


def detectar_anomalias_prescricoes(caminho_csv):
    tabela = carregar_prescricoes(caminho_csv)
    alertas = (verificar_dose_maxima(tabela)
               + verificar_aumento_brusco_e_intervalo(tabela)
               + verificar_alta_vigilancia(tabela))
    print(f"Prescrições: {len(tabela)} administrações analisadas, {len(alertas)} alertas.")
    return alertas
