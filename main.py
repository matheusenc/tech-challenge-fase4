"""
Ponto de entrada do sistema de monitoramento multimodal.

Uso mais simples (usa os arquivos da pasta dados/):
    python main.py

Escolhendo os arquivos:
    python main.py --video dados/fisioterapia.mp4 --tipo-video fisioterapia
    python main.py --audio dados/consulta.wav --paciente P002
    python main.py --sinais dados/sinais_vitais.csv --prescricoes dados/prescricoes.csv

Cada etapa só roda se o arquivo existir. Se uma etapa der erro, as outras
continuam e o erro aparece na tela.

Saídas (pasta saida/):
    alertas.json                 -> todos os alertas gerados
    relatorio_monitoramento.md   -> relatório automático
    <video>_anotado.mp4          -> vídeo com esqueleto, objetos e alertas desenhados
"""

import argparse
import traceback
from pathlib import Path

from monitoramento import config
from monitoramento.alertas import notificar_equipe_medica, ordenar_alertas, salvar_alertas
from monitoramento.analise_audio import analisar_audio
from monitoramento.analise_video import analisar_video
from monitoramento.anomalias_movimentacao import detectar_anomalias_movimentacao
from monitoramento.anomalias_prescricoes import detectar_anomalias_prescricoes
from monitoramento.anomalias_sinais_vitais import detectar_anomalias_sinais_vitais
from monitoramento.relatorio import gerar_relatorio


def ler_argumentos():
    dados = config.PASTA_DADOS
    parser = argparse.ArgumentParser(description="Monitoramento multimodal de pacientes")
    parser.add_argument("--video", default=dados / "video_fisioterapia.mp4", type=Path)
    parser.add_argument("--tipo-video", default="fisioterapia", choices=["fisioterapia", "cirurgia"])
    parser.add_argument("--audio", default=dados / "teste.wav", type=Path)
    parser.add_argument("--paciente", default="P001", help="Paciente do vídeo/áudio")
    parser.add_argument("--falante-paciente", help="ID diarizado do paciente no áudio (ex.: Guest-2)")
    parser.add_argument("--sinais", default=dados / "sinais_vitais.csv", type=Path)
    parser.add_argument("--prescricoes", default=dados / "prescricoes.csv", type=Path)
    parser.add_argument("--movimentacao", default=dados / "movimentacao.csv", type=Path)
    parser.add_argument("--sem-webhook", action="store_true", help="Não envia alertas para o webhook")
    return parser.parse_args()


def executar_etapa(nome, caminho, funcao):
    """Roda uma etapa se o arquivo existir. Erros não derrubam o programa."""
    print(f"\n--- {nome} ---")
    if not caminho.exists():
        print(f"Arquivo {caminho} não encontrado. Etapa ignorada.")
        return None
    try:
        return funcao()
    except Exception:
        print(f"Erro na etapa '{nome}':")
        traceback.print_exc()
        return None


def main():
    args = ler_argumentos()
    print("Status do Azure:")
    print(f"  Speech to Text:  {'configurado' if config.azure_speech_configurado() else 'NÃO configurado'}")
    print(f"  Text Analytics:  {'configurado' if config.azure_language_configurado() else 'NÃO configurado'}")

    todos_alertas = []
    resultados_video = []
    resultados_audio = []

    # 1. Vídeo
    retorno = executar_etapa("Análise de vídeo", args.video,
                             lambda: analisar_video(args.video, args.paciente, args.tipo_video))
    if retorno:
        resultado, alertas = retorno
        resultados_video.append(resultado)
        todos_alertas += alertas

    # 2. Áudio
    retorno = executar_etapa("Análise de áudio", args.audio,
                             lambda: analisar_audio(args.audio, args.paciente, args.falante_paciente))
    if retorno:
        resultado, alertas = retorno
        resultados_audio.append(resultado)
        todos_alertas += alertas

    # 3. Anomalias
    todos_alertas += executar_etapa("Sinais vitais", args.sinais,
                                    lambda: detectar_anomalias_sinais_vitais(args.sinais)) or []
    todos_alertas += executar_etapa("Prescrições", args.prescricoes,
                                    lambda: detectar_anomalias_prescricoes(args.prescricoes)) or []
    todos_alertas += executar_etapa("Movimentação", args.movimentacao,
                                    lambda: detectar_anomalias_movimentacao(args.movimentacao)) or []

    # 4. Alertas e relatório
    todos_alertas = ordenar_alertas(todos_alertas)
    salvar_alertas(todos_alertas, config.PASTA_SAIDA / "alertas.json")
    notificar_equipe_medica(todos_alertas, enviar_webhook=not args.sem_webhook)

    caminho_relatorio = gerar_relatorio(config.PASTA_SAIDA / "relatorio_monitoramento.md",
                                        todos_alertas, resultados_video, resultados_audio)
    print(f"\nRelatório salvo em: {caminho_relatorio}")
    print(f"Alertas salvos em:  {config.PASTA_SAIDA / 'alertas.json'}")


if __name__ == "__main__":
    main()
