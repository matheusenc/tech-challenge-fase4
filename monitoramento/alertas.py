"""
Alertas para a equipe médica.

Todo módulo de análise (vídeo, áudio, sinais vitais, prescrições,
movimentação) devolve uma lista de objetos `Alerta`. Este arquivo
cuida de:
  1. Remover alertas repetidos
  2. Ordenar por gravidade
  3. Fundir os alertas de todas as fontes por paciente (prioridade de atendimento)
  4. Salvar em arquivo JSON
  5. Enviar para a equipe médica (tela + webhook opcional)
"""

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta

import requests

from monitoramento import config

ORDEM_GRAVIDADE = {"alta": 0, "media": 1, "baixa": 2}
PESO_GRAVIDADE = {"alta": 3, "media": 2, "baixa": 1}


@dataclass
class Alerta:
    data_hora: datetime
    paciente_id: str
    origem: str        # "video", "audio", "sinais_vitais", "prescricoes" ou "movimentacao"
    tipo: str          # nome curto do problema
    gravidade: str     # "alta", "media" ou "baixa"
    descricao: str     # explicação para a equipe médica

    def como_dicionario(self):
        dados = asdict(self)
        dados["data_hora"] = self.data_hora.strftime("%Y-%m-%d %H:%M:%S")
        return dados


def remover_repetidos(alertas, janela_minutos=None, ultimo_envio=None):
    """Se o mesmo problema se repete para o mesmo paciente em pouco tempo,
    mantém só o primeiro alerta (evita "spam" para a equipe).

    ultimo_envio: dicionário (paciente, origem, tipo) -> data/hora. A simulação
    em tempo real passa sempre o mesmo dicionário, para lembrar o que já enviou.
    """
    janela = timedelta(minutes=janela_minutos or config.JANELA_ALERTAS_REPETIDOS_MINUTOS)
    if ultimo_envio is None:
        ultimo_envio = {}

    resultado = []
    for alerta in sorted(alertas, key=lambda a: a.data_hora):
        chave = (alerta.paciente_id, alerta.origem, alerta.tipo)
        ultima_vez = ultimo_envio.get(chave)
        if ultima_vez is None or alerta.data_hora - ultima_vez > janela:
            resultado.append(alerta)
            ultimo_envio[chave] = alerta.data_hora
    return resultado


def ordenar_alertas(alertas):
    """Alertas mais graves primeiro; dentro da mesma gravidade, por data."""
    return sorted(alertas, key=lambda a: (ORDEM_GRAVIDADE.get(a.gravidade, 9), a.data_hora))


def resumo_por_paciente(alertas):
    """Fusão multimodal: junta os alertas de TODAS as fontes por paciente.

    Pontuação de risco = soma dos pesos (alta 3, média 2, baixa 1). Pacientes com
    problemas vindos de mais fontes diferentes aparecem primeiro, porque sinais
    independentes que concordam (ex.: SpO2 baixa + "falta de ar" na consulta)
    são mais confiáveis do que um sinal isolado.
    """
    pacientes = {}
    for alerta in alertas:
        resumo = pacientes.setdefault(alerta.paciente_id, {
            "paciente_id": alerta.paciente_id, "fontes": set(), "pontuacao": 0,
            "alta": 0, "media": 0, "baixa": 0})
        resumo["fontes"].add(alerta.origem)
        resumo["pontuacao"] += PESO_GRAVIDADE.get(alerta.gravidade, 0)
        resumo[alerta.gravidade] += 1

    for resumo in pacientes.values():
        resumo["fontes"] = sorted(resumo["fontes"])
    return sorted(pacientes.values(), key=lambda r: (-len(r["fontes"]), -r["pontuacao"]))


def salvar_alertas(alertas, caminho_arquivo):
    caminho_arquivo.parent.mkdir(parents=True, exist_ok=True)
    lista = [alerta.como_dicionario() for alerta in alertas]
    with open(caminho_arquivo, "w", encoding="utf-8") as arquivo:
        json.dump(lista, arquivo, ensure_ascii=False, indent=2)


def formatar_alerta(alerta):
    icone = {"alta": "[ALTA] ", "media": "[MÉDIA]", "baixa": "[BAIXA]"}.get(alerta.gravidade, "")
    data = alerta.data_hora.strftime("%d/%m %H:%M")
    return f"{icone} {data} | Paciente {alerta.paciente_id} | {alerta.tipo}: {alerta.descricao}"


def formatar_resumo(resumo):
    return (f"Paciente {resumo['paciente_id']}: {resumo['pontuacao']} pontos | "
            f"{len(resumo['fontes'])} fontes ({', '.join(resumo['fontes'])}) | "
            f"alta {resumo['alta']}, média {resumo['media']}, baixa {resumo['baixa']}")


def notificar_equipe_medica(alertas, enviar_webhook=True):
    """Mostra a prioridade por paciente e os alertas na tela e, se houver
    webhook no .env, envia o resumo e os alertas de gravidade alta e média."""
    if not alertas:
        print("Nenhum alerta para enviar.")
        return

    prioridade = [formatar_resumo(r) for r in resumo_por_paciente(alertas)]
    print("\n" + "=" * 80)
    print("PRIORIDADE DE ATENDIMENTO (fusão de todas as fontes)")
    print("=" * 80)
    print("\n".join(prioridade))
    print("\n" + "=" * 80)
    print(f"ALERTAS PARA A EQUIPE MÉDICA ({len(alertas)})")
    print("=" * 80)
    for alerta in alertas:
        print(formatar_alerta(alerta))

    if enviar_webhook and config.WEBHOOK_EQUIPE_MEDICA:
        importantes = [a for a in alertas if a.gravidade in ("alta", "media")]
        if importantes:
            enviar_para_webhook(importantes, cabecalho=prioridade)


def enviar_para_webhook(alertas, cabecalho=(), limite=20):
    linhas = list(cabecalho) + [""] if cabecalho else []
    linhas += [formatar_alerta(a) for a in alertas[:limite]]
    if len(alertas) > limite:
        linhas.append(f"... e mais {len(alertas) - limite} alertas (ver relatório).")
    mensagem = "🚨 Monitoramento de pacientes\n" + "\n".join(linhas)

    try:
        # "text" é o campo usado por Teams/Slack; "content" é o usado pelo Discord
        resposta = requests.post(
            config.WEBHOOK_EQUIPE_MEDICA,
            json={"text": mensagem, "content": mensagem[:1900]},
            timeout=10,
        )
        resposta.raise_for_status()
        print(f"\nAlertas enviados para o webhook da equipe médica ({len(alertas)}).")
    except requests.RequestException as erro:
        # Falha no envio não pode derrubar o sistema: os alertas continuam salvos no JSON
        print(f"\nNão foi possível enviar para o webhook (confira a URL em config.py ou no .env): {erro}")
