# Tech Challenge Fase 4: Monitoramento Multimodal de Pacientes

Sistema em Python que analisa **vídeo**, **áudio** e **dados clínicos** de pacientes
para identificar sinais precoces de risco e **alertar a equipe médica**.

| Tipo de dado | O que o sistema faz | Tecnologia |
|---|---|---|
| Vídeo de fisioterapia | Detecta postura inadequada e possível queda | YOLOv8-Pose (análise postural por pontos corporais) |
| Vídeo de cirurgia | Detecta objeto de risco dentro da área crítica | YOLOv8 (detecção de objetos) |
| Áudio (consultas) | Transcreve, separa falantes, mede pausas e ritmo da fala, procura termos críticos e analisa sentimento | Azure Speech to Text, librosa, Azure Text Analytics |
| Sinais vitais | Limites clínicos, variação brusca (z-score) e combinações incomuns | Regras + Isolation Forest |
| Prescrições | Dose acima do máximo, aumento brusco, dose duplicada, remédio de alta vigilância | Regras |
| Movimentação | Inatividade prolongada, agitação noturna, queda de mobilidade | Regras |

Todos os módulos geram **alertas** no mesmo formato. Os alertas são **fundidos por paciente**
(prioridade de atendimento considerando todas as fontes), salvos em JSON, mostrados na tela,
enviados para um webhook (opcional) e reunidos em um **relatório automático**.

📘 **Manual com diagrama interativo:** abra [`docs/manual.html`](docs/manual.html) no navegador.
Ele explica cada etapa do fluxo, as ligações entre os módulos e a relação com as aulas da fase.

---

## Estrutura do projeto

```
tech-challenge-fase4/
├── main.py                       # roda tudo: vídeo, áudio, anomalias, alertas e relatório
├── simular_tempo_real.py         # demonstração de monitoramento em tempo real dos sinais vitais
├── gerar_dados_exemplo.py        # cria dados sintéticos com anomalias conhecidas
├── converter_physionet.py        # converte dados reais do PhysioNet para o formato do projeto
├── monitoramento/
│   ├── config.py                 # TODOS os limites e configurações ficam aqui
│   ├── alertas.py                # formato do alerta, remoção de repetidos, fusão por paciente, envio
│   ├── analise_video.py          # fisioterapia: YOLOv8-Pose | cirurgia: YOLOv8
│   ├── analise_audio.py          # librosa + Azure Speech + Azure Text Analytics
│   ├── anomalias_sinais_vitais.py
│   ├── anomalias_prescricoes.py
│   ├── anomalias_movimentacao.py
│   └── relatorio.py              # gera o relatório em Markdown
├── dados/                        # arquivos de entrada (CSV, vídeo, áudio)
├── saida/                        # gerada ao rodar: alertas.json, relatório, vídeo anotado
├── testes/                       # testes automáticos (pytest), inclusive das regras de vídeo e áudio
├── docs/manual.html              # manual com diagrama interativo do fluxo
└── RELATORIO_TECNICO.md          # relatório técnico exigido na entrega
```

---

## 1. Instalação

Requisitos: **Python 3.10 ou mais novo** (execução e testes validados com Python 3.12.10).

```bash
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

> A instalação do `ultralytics` traz o PyTorch e pode demorar alguns minutos.
> Os pesos dos modelos (`yolov8n.pt` e `yolov8n-pose.pt`) são baixados
> automaticamente na primeira execução.

## 2. Configurar o Azure

1. No [portal do Azure](https://portal.azure.com), crie os recursos necessários:
   - **Speech** (Azure AI Speech)
   - **Language** (Azure AI Language, antigo Text Analytics)
2. Em cada recurso, abra **Chaves e Ponto de Extremidade** e copie os valores correspondentes. Verifique a disponibilidade e as limitações do plano escolhido, especialmente para diarização.
3. Copie o arquivo `.env.exemplo` para `.env` e preencha:

```dotenv
AZURE_SPEECH_KEY=sua-chave
AZURE_SPEECH_REGION=brazilsouth
# Opcional: preencha com o endpoint real se utilizar a Fast Transcription API.
AZURE_SPEECH_ENDPOINT=
AZURE_LANGUAGE_KEY=sua-chave
AZURE_LANGUAGE_ENDPOINT=https://seu-recurso.cognitiveservices.azure.com/
WEBHOOK_EQUIPE_MEDICA=
```

No PowerShell, a cópia pode ser feita com `Copy-Item .env.exemplo .env`.
A região `brazilsouth` é apenas um exemplo; utilize a região e o endpoint do seu recurso.
**Não publique o arquivo `.env` nem suas chaves no GitHub.**

**Sem o Azure configurado o projeto continua funcionando parcialmente:** se existir um
`.txt` com o mesmo nome do áudio (ex.: `consulta.wav` → `consulta.txt`), ele pode ser
utilizado como transcrição alternativa. Sem diarização, o sistema não atribui automaticamente
termos críticos ou sentimento a um paciente específico.

**Webhook (alerta à equipe):** por padrão, não há destino fixo configurado. Para enviar alertas,
defina `WEBHOOK_EQUIPE_MEDICA` no `.env` com uma URL autorizada. Para executar sem envio externo,
use `--sem-webhook`.

## 3. Executar

```bash
# 1) Gerar dados de exemplo (sinais vitais, prescrições, movimentação)
python gerar_dados_exemplo.py

# 2) Rodar o sistema completo (arquivos padrão da pasta dados/)
python main.py

# Execução integrada validada: vídeo, áudio e dados clínicos
python main.py --audio dados/consulta.wav --paciente P001 --falante-paciente Guest-2 --sem-webhook

# Com vídeo e áudio próprios:
python main.py --video dados/video_fisioterapia.mp4 --tipo-video fisioterapia --audio dados/consulta.wav --paciente P001 --falante-paciente Guest-2 --sem-webhook
python main.py --video dados/cirurgia.mp4 --tipo-video cirurgia --sem-webhook

# 3) Simulação de monitoramento em tempo real (sinais vitais chegando um a um)
python simular_tempo_real.py --intervalo 0.02

# 4) Testes automáticos
python -m pytest testes/ -q
```

O identificador `P001` é o paciente nos dados clínicos; `Guest-2` é o identificador
atribuído à voz do paciente na diarização do áudio de demonstração. Em outras gravações,
confirme qual `Guest-N` corresponde ao paciente antes de executar.

**Resultados da execução integrada de 11/11/2026:** 129 quadros de vídeo analisados
(1 evento postural), 2 termos críticos no áudio, 11 alertas de sinais vitais,
5 de prescrições e 3 de movimentação; **22 alertas** no total. Os **31 testes**
automatizados passaram. Consulte [`RELATORIO_TECNICO.md`](RELATORIO_TECNICO.md)
para os detalhes e limitações.

Resultados na pasta `saida/`:

- `relatorio_monitoramento.md`: relatório automático com prioridade por paciente, desvios do vídeo, análise do áudio e alertas
- `alertas.json`: todos os alertas (para integrar com outros sistemas)
- `<nome_do_video>_anotado.mp4`: vídeo com esqueleto, objetos, área crítica e alertas desenhados

## 4. Onde conseguir os dados

| Dado | Sugestão |
|---|---|
| Sinais vitais reais | [PhysioNet BIDMC](https://physionet.org/content/bidmc/1.0.0/) (acesso aberto). Baixe a pasta `bidmc_csv` e rode `python converter_physionet.py caminho/bidmc_csv`, depois `python main.py --sinais dados/sinais_vitais_physionet.csv` |
| Vídeo de fisioterapia | Grave um integrante fazendo exercícios ou use vídeos livres do [Pexels](https://www.pexels.com/search/videos/physiotherapy/) |
| Vídeo de "cirurgia" | Grave uma mesa simulando o campo cirúrgico e coloque um celular ou tesoura no centro (área crítica) |
| Áudio de consulta | Use o arquivo de demonstração `dados/consulta.wav` ou grave um diálogo simulado entre duas pessoas, convertendo para `.wav` se necessário |
| Sons respiratórios | [Google AudioSet](https://research.google.com/audioset/): classes *Cough*, *Breathing*, *Wheeze*, *Gasp* para estudos adicionais |

## 5. Ajustando o sistema

Todos os limites ficam em `monitoramento/config.py`, com comentários. Exemplos:

- `LIMITES_SINAIS_VITAIS`: faixa normal de cada sinal vital
- `LIMITE_INCLINACAO_TRONCO_GRAUS`, `LIMITE_DESCIDA_QUEDA`: sensibilidade da análise postural
- `AREA_CRITICA`, `OBJETOS_DE_RISCO`: região do campo cirúrgico e objetos proibidos nela
- `TERMOS_CRITICOS`: palavras procuradas na transcrição
- `LIMITE_PROPORCAO_PAUSAS`, `LIMITE_PAUSAS_POR_MINUTO`, `LIMITE_PALAVRAS_POR_MINUTO`: indicadores vocais
- `DOSE_MAXIMA`, `INTERVALO_MINIMO_HORAS`: regras de prescrição

## Sobre OpenPose x YOLOv8-Pose

O enunciado cita o OpenPose como exemplo de modelo de análise postural. Usamos o **YOLOv8-Pose**,
que estima **17 pontos do corpo (padrão COCO)**, instala com `pip` (o OpenPose
exige uma configuração mais complexa) e vem da mesma biblioteca do YOLOv8 usado para os objetos.
Nas aulas o conceito de keypoints/landmarks + regras sobre as coordenadas foi mostrado com o
MediaPipe; utilizamos o YOLOv8-Pose para usar uma única biblioteca para pose e objetos.

## Aviso

Projeto acadêmico. Os limites são ilustrativos e as análises de voz são **heurísticas de triagem,
não diagnóstico**. Em uso real, os critérios precisam ser validados pela equipe médica.
A inclinação detectada no vídeo representa um desvio de um limite configurado, não comprova
que o movimento de fisioterapia foi executado incorretamente.