 Relatório Técnico — Tech Challenge Fase 4

**Projeto:** Monitoramento Multimodal de Pacientes
**Natureza:** Protótipo acadêmico de apoio ao monitoramento
**Data da validação integrada:** 11 de novembro de 2026
**Ambiente de teste informado:** Windows, Python 3.12.10 e pytest 9.2.0

## 1. Introdução

Este projeto apresenta uma solução acadêmica de monitoramento multimodal de pacientes, desenvolvida em Python. A aplicação combina análise de vídeo de fisioterapia, gravações de consultas e registros de sinais vitais, prescrições e movimentação para identificar eventos que merecem avaliação da equipe médica.

As diferentes fontes produzem alertas padronizados, que são consolidados por paciente. O sistema apresenta uma classificação de prioridade, registra as ocorrências e gera um relatório automático.

**Escopo e segurança:** a solução é um protótipo educacional. Seus limites, regras de gravidade e pontuações são ilustrativos e não representam diagnóstico, triagem clínica validada nem recomendação de tratamento.

## 2. Objetivos

- Processar vídeos de fisioterapia com estimativa de postura e identificar desvios em relação a limites configurados.
- Disponibilizar análise de objetos em vídeos de simulação cirúrgica.
- Transcrever consultas em português, identificar os falantes e analisar o conteúdo atribuído ao paciente.
- Calcular indicadores vocais de pausas e velocidade da fala com `librosa`.
- Identificar possíveis anomalias em sinais vitais, prescrições e movimentação.
- Consolidar alertas de diferentes fontes, priorizar pacientes e gerar registros para análise.
- Demonstrar o funcionamento integrado dos componentes e verificar as regras com testes automatizados.

## 3. Arquitetura da solução

A aplicação usa uma arquitetura modular em Python. O arquivo `main.py` recebe os parâmetros de execução, aciona os módulos e consolida os resultados.

| Arquivo | Responsabilidade |
|---|---|
| `monitoramento/analise_video.py` | Processamento de vídeo com YOLOv8 e análise de postura ou objetos. |
| `monitoramento/analise_audio.py` | Características da voz, transcrição e diarização, análise textual e termos críticos. |
| `monitoramento/anomalias_sinais_vitais.py` | Regras, variações e identificação de padrões incomuns em sinais vitais. |
| `monitoramento/anomalias_prescricoes.py` | Verificações ilustrativas de doses, intervalos e medicamentos. |
| `monitoramento/anomalias_movimentacao.py` | Inatividade, agitação e redução de mobilidade. |
| `monitoramento/alertas.py` | Padronização, consolidação, priorização e correlação de alertas. |
| `monitoramento/relatorio.py` | Relatório de monitoramento em Markdown. |
| `monitoramento/config.py` | Parâmetros e limites de análise, além de configuração dos serviços. |

O fluxo principal é: **arquivos de entrada → análise por módulo → alertas padronizados → consolidação por paciente → relatório e JSON de alertas**. O envio por webhook é opcional; na validação descrita neste relatório, foi desativado por meio de `--sem-webhook`.

### 3.1. Entradas e saídas

Na execução integrada, foram utilizados `dados/video_fisioterapia.mp4`, `dados/consulta.wav` e os arquivos CSV de exemplo dos módulos clínicos.

As saídas incluem `saida/relatorio_monitoramento.md`, `saida/alertas.json` e o vídeo anotado gerado pelo processamento. Os arquivos de saída são produzidos a cada execução e podem ser substituídos por execuções posteriores; para conservar evidências, é necessário copiá-los antes de executar novamente.

## 4. Modelos e tecnologias

### 4.1. Análise de vídeo

Foi utilizado **YOLOv8-Pose (`yolov8n-pose.pt`)** para obter pontos corporais e calcular a inclinação do tronco. A implementação identifica eventos que ultrapassam limites geométricos e gera um vídeo anotado. A aplicação também prevê análise de objetos com YOLOv8 em simulações de procedimentos cirúrgicos, mas **não foi apresentado nesta validação um ensaio integrado do cenário cirúrgico**.

A identificação de uma inclinação acima do limite é um *desvio em relação à regra programada*. Durante um exercício de fisioterapia, a inclinação pode ser voluntária e adequada ao movimento; por isso, o alerta não deve ser interpretado isoladamente como postura clinicamente inadequada.

### 4.2. Análise de áudio

O processamento emprega:

- **`librosa`** para estimar trechos de atividade vocal, pausas e ritmo da fala.
- **Azure AI Speech** para transcrever a consulta e separar os participantes por identificadores de falante.
- **Azure AI Language (Text Analytics)** para análise de sentimento e extração de frases-chave.
- **Regras de texto** para localizar termos críticos e tratar casos simples de negação explícita.

Na consulta testada, a médica foi identificada como `Guest-1` e o paciente como `Guest-2`. O parâmetro `--falante-paciente Guest-2` permitiu analisar os termos críticos e os indicadores vocais atribuídos ao paciente, evitando atribuir automaticamente ao paciente as perguntas da médica. A correspondência entre identificador de voz e papel na consulta precisa ser confirmada na demonstração; o número `Guest-N` não identifica uma pessoa por si só.

A lógica de negação é uma heurística limitada. O teste demonstrou a supressão do alerta de **dor no peito** quando o paciente afirmou que não apresentava esse sintoma, sem impedir a detecção de **cansaço** e **falta de ar**.

### 4.3. Detecção de anomalias nos registros clínicos

Nos sinais vitais são usadas regras configuradas, análise de variações e **Isolation Forest** para padrões incomuns. O treinamento do modelo foi ajustado para usar uma parcela inicial cronológica da série e avaliar registros posteriores, evitando o uso simultâneo de todo o conjunto para treinamento e avaliação.

Prescrições e movimentação utilizam regras de negócio ilustrativas, como dose acima de limite, intervalo entre administrações, agitação noturna e inatividade prolongada. A análise de dose inclui verificação da unidade informada antes da comparação com o limite de referência.

## 5. Fusão multimodal e alertas

Cada alerta registra paciente, origem, tipo, descrição, data/hora e gravidade. O sistema consolida as ocorrências e calcula uma pontuação de prioridade por paciente utilizando regras e pesos internos. Essa pontuação é **operacional e experimental**, não uma escala clínica validada.

Além da consolidação por paciente, existe uma rotina de **correlação de alertas de múltiplas fontes em uma janela de cinco minutos**, com apresentação de episódios no relatório quando os registros atendem às condições temporais. A existência dessa rotina não implica que os eventos da demonstração tenham sido simultâneos: os CSVs de exemplo registram ocorrências em outubro, enquanto vídeo e áudio foram processados em novembro.

## 6. Experimentos e resultados

### 6.1. Procedimento da validação integrada

No ambiente Windows com Python 3.12.10 e o ambiente virtual ativo, foi executado:

```powershell
python main.py --audio dados/consulta.wav --paciente P001 --falante-paciente Guest-2 --sem-webhook
```

O arquivo `dados/video_fisioterapia.mp4` já havia sido renomeado para o nome padrão procurado pela aplicação, permitindo analisar vídeo e áudio na mesma execução sem informar `--video`. As integrações Azure Speech to Text e Azure Text Analytics foram apresentadas como configuradas e executaram normalmente.

### 6.2. Resultado por modalidade

| Fonte | Volume processado | Resultado observado |
|---|---:|---|
| Vídeo de fisioterapia | 129 quadros | 1 evento de inclinação postural fora do limite configurado. |
| Áudio de consulta | 1 arquivo WAV | 2 termos críticos atribuídos ao paciente; 0 alterações vocais pelas regras atuais. |
| Sinais vitais | 1.728 leituras | 11 alertas. |
| Prescrições | 70 administrações | 5 alertas. |
| Movimentação | 216 leituras | 3 alertas. |
| **Total** | **5 fontes** | **22 alertas.** |

**Vídeo:** foi registrado um evento denominado “Postura inadequada”, com detalhe de inclinação de tronco de **46°** no intervalo de **00:01.8 a 00:11.2**. A duração do intervalo é de **9,4 segundos**. O vídeo foi processado com `yolov8n-pose.pt`. O valor de 46° é o valor apresentado no registro do evento, não necessariamente o pico de toda a sequência.

**Áudio:** a transcrição e a análise textual utilizaram os serviços Azure. Os dois termos críticos detectados foram **“falta de ar”** (alerta de gravidade alta) e **“cansaço”** (alerta de gravidade média). A negativa de dor no peito não gerou alerta nessa execução.

Os indicadores vocais atribuídos ao paciente foram:

| Indicador | Resultado | Regra de alerta | Interpretação nesta execução |
|---|---:|---|---|
| Proporção de pausas | 2,0% | Maior que 45,0% | Não ultrapassou o limite. |
| Pausas por minuto | 2,9 | Maior que 25 | Não ultrapassou o limite. |
| Palavras por minuto | 126 | Menor que 90 | Não ultrapassou o limite. |

Portanto, **zero alertas de alteração vocal é compatível com os valores observados**. A lógica de identificação desses alertas também foi exercitada por testes automatizados; a consulta usada na demonstração não constitui um exemplo positivo de alteração vocal.

### 6.3. Consolidação por paciente

| Paciente | Pontos de prioridade | Fontes que produziram alertas | Alertas por gravidade |
|---|---:|---:|---|
| P001 | 27 | 5: vídeo, áudio, sinais vitais, prescrições e movimentação | 6 altas, 4 médias, 1 baixa. |
| P002 | 15 | 3: sinais vitais, prescrições e movimentação | 2 altas, 3 médias, 3 baixas. |
| P003 | 6 | 3: sinais vitais, prescrições e movimentação | 0 altas, 3 médias, 0 baixas. |

O paciente **P001** obteve a maior pontuação no cenário simulado. A soma dos alertas por paciente é de **22**, em concordância com o total apresentado na execução.

### 6.4. Testes automatizados

Foi executado o comando:

```powershell
python -m pytest testes/ -q
```

Resultado informado: **31 testes aprovados em 3,40 segundos**, no ambiente local com Python 3.12.10 e pytest 9.2.0. Os **11 testes em `testes/test_audio.py`** também foram executados individualmente e aprovados, abrangendo termos críticos, normalização de texto, casos simples de negação, uso de intervalos do falante, silêncio, pausas, fala lenta e normalização de resultados de transcrição.

Esse resultado demonstra que os testes automatizados existentes passaram; **não representa medida de sensibilidade, especificidade, precisão diagnóstica ou ausência de erros** em situações reais.

### 6.5. Evidências produzidas

- Registro de execução integrada no terminal, com status dos serviços Azure e contagens de cada módulo.
- `saida/relatorio_monitoramento.md` com os resultados do monitoramento.
- `saida/alertas.json` contendo os alertas gerados.
- Arquivo de vídeo anotado associado a `video_fisioterapia.mp4`.
- Saída do pytest indicando **31 testes aprovados**.

As informações de tempo de processamento disponíveis neste relatório são apenas as da suíte de testes; não foi realizada medição padronizada de latência, throughput ou desempenho dos serviços Azure.

## 7. Limitações

1. **Uso educacional:** os limiares e as prioridades não foram validados clinicamente. Os alertas indicam ocorrências segundo regras programadas.
2. **Vídeo:** um exercício legítimo pode ultrapassar o limite angular e produzir alerta. A solução não determina, por si só, se um movimento foi prescrito ou executado corretamente. A seleção postural utilizada no protótipo não faz rastreamento sofisticado de pessoas em cenas com múltiplos participantes.
3. **Áudio e diarização:** a transcrição automática pode errar palavras ou separar falantes incorretamente. A atribuição do papel de paciente depende de confirmação. Os indicadores de silêncio e ritmo são medidas heurísticas, não diagnósticos de fadiga, disartria ou dificuldade respiratória.
4. **Negação de sintomas:** o tratamento textual cobre casos simples e pode falhar com frases ambíguas, negativas distantes, transcrição imperfeita ou relatos contraditórios.
5. **Dados clínicos:** registros e limites usados na demonstração são ilustrativos. O Isolation Forest sinaliza padrões estatisticamente incomuns, que não são necessariamente anomalias clínicas.
6. **Correlação temporal:** os eventos dos CSVs de outubro e os alertas de áudio/vídeo de novembro não comprovam episódios simultâneos entre todas as fontes.
7. **Cobertura experimental:** a execução integrada apresentada validou o vídeo de fisioterapia, não um cenário completo de cirurgia. Não foram obtidas métricas formais de classificação com dados rotulados nem testes clínicos de campo.
8. **Privacidade e operação:** a apresentação deve utilizar dados simulados ou adequadamente desidentificados. Chaves Azure permanecem no `.env` local e não devem ser publicadas no repositório. O webhook foi desativado durante a demonstração.

## 8. Conclusão

A execução integrada demonstrou o processamento das **cinco fontes** previstas, com uso efetivo de **Azure Speech** e **Azure AI Language**, identificação de **um evento postural**, detecção de **dois termos críticos na consulta** e geração de **22 alertas** no total. A consolidação atribuiu **27 pontos** ao paciente P001, que apresentou alertas provenientes das cinco fontes.

A suíte automatizada concluiu com **31 testes aprovados**. Esses resultados sustentam a viabilidade técnica da integração no cenário acadêmico demonstrado, sem comprovar adequação para decisões clínicas reais.

Como próximas atividades de entrega, recomenda-se revisar o `README.md`, preservar as evidências da execução e apresentar em vídeo o fluxo de entrada de dados, processamento, vídeo anotado, priorização e relatório. Não é necessária a introdução de novos modelos para demonstrar os resultados já obtidos.

## 9. Referências

- Enunciado acadêmico — Tech Challenge Fase 4 (FIAP).
- Ultralytics — documentação YOLO: https://docs.ultralytics.com/
- Microsoft Learn — Azure AI Speech: https://learn.microsoft.com/azure/ai-services/speech-service/
- Microsoft Learn — Azure AI Language: https://learn.microsoft.com/azure/ai-services/language-service/
- `librosa` — documentação: https://librosa.org/doc/latest/
- scikit-learn — Isolation Forest: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html
- pytest — documentação: https://docs.pytest.org/
- PhysioNet — repositório de dados fisiológicos: https://physionet.org/