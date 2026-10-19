"""
Geração do relatório automático (arquivo Markdown) com tudo que foi
analisado: eventos do vídeo, análise do áudio e alertas de anomalias.
O Markdown pode ser aberto no VS Code, no GitHub ou convertido para PDF.
"""

from collections import Counter
from datetime import datetime

from monitoramento.alertas import formatar_alerta, resumo_por_paciente
from monitoramento.analise_video import formatar_tempo

NOMES_GRAVIDADE = {"alta": "alta", "media": "média", "baixa": "baixa"}

NOMES_ORIGEM = {
    "video": "Vídeo",
    "audio": "Áudio",
    "sinais_vitais": "Sinais vitais",
    "prescricoes": "Prescrições",
    "movimentacao": "Movimentação",
}


def secao_resumo(alertas):
    linhas = ["## Resumo dos alertas", ""]
    if not alertas:
        return linhas + ["Nenhum alerta gerado.", ""]

    por_origem = Counter(a.origem for a in alertas)
    por_gravidade = Counter(a.gravidade for a in alertas)
    linhas += ["| Origem | Alta | Média | Baixa | Total |", "|---|---|---|---|---|"]
    for origem, total in por_origem.items():
        contagem = Counter(a.gravidade for a in alertas if a.origem == origem)
        linhas.append(f"| {NOMES_ORIGEM.get(origem, origem)} | {contagem['alta']} | {contagem['media']} "
                      f"| {contagem['baixa']} | {total} |")
    linhas.append(f"| **Total** | **{por_gravidade['alta']}** | **{por_gravidade['media']}** "
                  f"| **{por_gravidade['baixa']}** | **{len(alertas)}** |")
    return linhas + [""]


def secao_pacientes(alertas):
    """Fusão multimodal: uma linha por paciente, com todas as fontes juntas."""
    linhas = ["## Prioridade por paciente (fusão multimodal)", "",
              "Soma de pesos dos alertas (alta = 3, média = 2, baixa = 1). Pacientes com problemas em mais "
              "fontes diferentes aparecem primeiro.", "",
              "| Paciente | Fontes com alerta | Alta | Média | Baixa | Pontuação |", "|---|---|---|---|---|---|"]
    for r in resumo_por_paciente(alertas):
        fontes = ", ".join(NOMES_ORIGEM.get(f, f) for f in r["fontes"])
        linhas.append(f"| {r['paciente_id']} | {fontes} | {r['alta']} | {r['media']} | {r['baixa']} "
                      f"| **{r['pontuacao']}** |")
    return linhas + [""]


def secao_video(resultado):
    linhas = [f"## Vídeo: {resultado['arquivo']}", "",
              f"- Tipo: {resultado['tipo_video']} (modelo `{resultado['modelo']}`)",
              f"- Paciente: {resultado['paciente_id']}",
              f"- Duração: {resultado['duracao_segundos']} s ({resultado['quadros_analisados']} quadros analisados)",
              f"- Vídeo anotado: `{resultado['video_anotado']}`", ""]

    if not resultado["eventos"]:
        return linhas + ["Nenhum desvio encontrado no procedimento.", ""]

    linhas += ["### Desvios e falhas detectados", "",
               "| Início | Fim | Evento | Gravidade | Detalhe |", "|---|---|---|---|---|"]
    for evento in resultado["eventos"]:
        linhas.append(f"| {formatar_tempo(evento['inicio_segundos'])} | {formatar_tempo(evento['fim_segundos'])} "
                      f"| {evento['tipo']} | {NOMES_GRAVIDADE[evento['gravidade']]} | {evento['detalhe']} |")
    return linhas + [""]


def secao_audio(resultado):
    c = resultado["caracteristicas"]
    linhas = [f"## Áudio: {resultado['arquivo']}", "",
              f"- Paciente: {resultado['paciente_id']}",
              f"- Duração: {c['duracao_segundos']} s (fala: {c['tempo_de_fala_segundos']} s)",
              f"- Proporção de pausas: {c['proporcao_pausas']:.0%}",
              f"- Pausas por minuto: {c['pausas_por_minuto']}",
              f"- Palavras por minuto: {c.get('palavras_por_minuto', 'n/d')}",
              f"- Transcrição obtida de: {resultado['fonte_transcricao']}", ""]

    linhas += ["### Alterações vocais", ""]
    linhas += [f"- {a}" for a in resultado["alteracoes_vocais"]] or ["- Nenhuma alteração identificada."]
    linhas += ["", "### Termos críticos", ""]
    linhas += [f"- **{t['termo']}** (gravidade {NOMES_GRAVIDADE[t['gravidade']]})"
               for t in resultado["termos_criticos"]] \
        or ["- Nenhum termo crítico encontrado."]

    analise = resultado["analise_texto"]
    linhas += ["", "### Sentimento (Azure Text Analytics)", ""]
    if analise:
        cont = analise["contagem"]
        linhas.append(f"- Sentimento geral: **{analise['sentimento_geral']}** "
                      f"(positivas {cont['positive']}, neutras {cont['neutral']}, "
                      f"negativas {cont['negative']}, mistas {cont['mixed']})")
        for item in analise["frases_negativas"][:10]:
            linhas.append(f"  - \"{item['frase']}\" (negativo {item['confianca']:.0%})")
        if analise["frases_chave"]:
            linhas.append(f"- Frases-chave: {', '.join(analise['frases_chave'][:25])}")
    else:
        linhas.append("- Não analisado (Azure Text Analytics não configurado).")

    if resultado["transcricao"]:
        linhas += ["", "### Transcrição", "", f"> {resultado['transcricao']}"]
    return linhas + [""]


def secao_alertas(alertas, limite=100):
    linhas = ["## Alertas enviados à equipe médica", ""]
    if not alertas:
        return linhas + ["Nenhum.", ""]
    linhas += [f"- {formatar_alerta(a)}" for a in alertas[:limite]]
    if len(alertas) > limite:
        linhas.append(f"- ... e mais {len(alertas) - limite} (ver `alertas.json`).")
    return linhas + [""]


def gerar_relatorio(caminho_arquivo, alertas, resultados_video=(), resultados_audio=()):
    linhas = ["# Relatório de Monitoramento Multimodal de Pacientes", "",
              f"Gerado em {datetime.now():%d/%m/%Y %H:%M}", ""]
    linhas += secao_resumo(alertas)
    if alertas:
        linhas += secao_pacientes(alertas)
    for resultado in resultados_video:
        linhas += secao_video(resultado)
    for resultado in resultados_audio:
        linhas += secao_audio(resultado)
    linhas += secao_alertas(alertas)

    caminho_arquivo.parent.mkdir(parents=True, exist_ok=True)
    caminho_arquivo.write_text("\n".join(linhas), encoding="utf-8")
    return caminho_arquivo
