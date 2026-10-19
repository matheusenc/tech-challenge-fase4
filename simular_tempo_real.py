"""
Simulação de monitoramento EM TEMPO REAL dos sinais vitais.

Lê o CSV de sinais vitais como se cada linha estivesse chegando agora de um
monitor multiparamétrico. Cada leitura é analisada no momento em que chega
e, se houver anomalia, o alerta é mostrado e enviado imediatamente.

    python simular_tempo_real.py
    python simular_tempo_real.py --paciente P001 --intervalo 0.02
"""

import argparse
import time
from pathlib import Path

from monitoramento import config
from monitoramento.alertas import enviar_para_webhook, formatar_alerta, remover_repetidos, salvar_alertas
from monitoramento.anomalias_sinais_vitais import carregar_sinais_vitais, verificar_leitura


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arquivo", default=config.PASTA_DADOS / "sinais_vitais.csv", type=Path)
    parser.add_argument("--paciente", default=None, help="Monitorar só um paciente (ex.: P001)")
    parser.add_argument("--intervalo", default=0.01, type=float, help="Segundos entre leituras")
    args = parser.parse_args()

    tabela = carregar_sinais_vitais(args.arquivo).sort_values("data_hora")
    if args.paciente:
        tabela = tabela[tabela["paciente_id"] == args.paciente]

    historico_por_paciente = {}       # paciente -> leituras anteriores
    ultimo_envio = {}                 # lembra o que já foi enviado (evita repetir alerta)
    alertas_emitidos = []

    print(f"Monitorando {len(tabela)} leituras... (Ctrl+C para parar)\n")
    try:
        for leitura in tabela.to_dict("records"):
            historico = historico_por_paciente.setdefault(leitura["paciente_id"], [])

            novos = remover_repetidos(verificar_leitura(leitura, historico), ultimo_envio=ultimo_envio)
            for alerta in novos:
                alertas_emitidos.append(alerta)
                print(formatar_alerta(alerta))
                if config.WEBHOOK_EQUIPE_MEDICA and alerta.gravidade == "alta":
                    enviar_para_webhook([alerta])

            historico.append(leitura)
            time.sleep(args.intervalo)
    except KeyboardInterrupt:
        print("\nMonitoramento interrompido.")

    salvar_alertas(alertas_emitidos, config.PASTA_SAIDA / "alertas_tempo_real.json")
    print(f"\n{len(alertas_emitidos)} alertas emitidos. Salvos em saida/alertas_tempo_real.json")


if __name__ == "__main__":
    main()
