"""
Detecção de anomalias em sinais vitais (série temporal).

Arquivo de entrada (CSV), uma linha por leitura:
    data_hora, paciente_id, frequencia_cardiaca, pressao_sistolica,
    pressao_diastolica, saturacao_oxigenio
(colunas de sinais podem faltar; o código usa as que existirem)

Três técnicas, da mais simples para a mais sofisticada:
  1. Limites clínicos: valor fora da faixa normal (ex.: SpO2 < 92%).
  2. Variação brusca (z-score móvel): o valor mudou muito em relação às
     últimas leituras DO PRÓPRIO paciente, mesmo estando dentro da faixa
     (precisa passar do z-score E de uma variação mínima em unidades).
  3. Isolation Forest (machine learning): encontra combinações incomuns
     de vários sinais ao mesmo tempo.

As regras 1 e 2 funcionam leitura por leitura (função `verificar_leitura`),
por isso são usadas também na simulação em tempo real.
"""

from statistics import mean, pstdev

import pandas as pd
from sklearn.ensemble import IsolationForest

from monitoramento import config
from monitoramento.alertas import Alerta, remover_repetidos


def carregar_sinais_vitais(caminho_csv):
    tabela = pd.read_csv(caminho_csv, parse_dates=["data_hora"])
    tabela["paciente_id"] = tabela["paciente_id"].astype(str)
    return tabela.sort_values(["paciente_id", "data_hora"]).reset_index(drop=True)


def colunas_de_sinais(tabela):
    """Devolve só as colunas de sinais vitais que existem no arquivo."""
    return [coluna for coluna in config.LIMITES_SINAIS_VITAIS if coluna in tabela.columns]


def verificar_leitura(leitura, historico):
    """Analisa UMA leitura nova.

    leitura: dicionário com data_hora, paciente_id e os sinais
    historico: lista com as leituras anteriores deste mesmo paciente
    """
    alertas = []

    for sinal, (minimo, maximo) in config.LIMITES_SINAIS_VITAIS.items():
        valor = leitura.get(sinal)
        if valor is None or pd.isna(valor):
            continue
        nome = config.NOMES_SINAIS_VITAIS[sinal]

        # Regra 1: limites clínicos
        if valor < minimo or valor > maximo:
            situacao = "abaixo" if valor < minimo else "acima"
            alertas.append(Alerta(
                data_hora=leitura["data_hora"],
                paciente_id=leitura["paciente_id"],
                origem="sinais_vitais",
                tipo=f"{nome} fora do limite",
                gravidade="alta",
                descricao=f"{nome} = {valor:.0f}, {situacao} da faixa normal ({minimo} a {maximo}).",
            ))
            continue   # já alertou; não precisa testar variação brusca

        # Regra 2: variação brusca em relação às últimas leituras
        valores_anteriores = [h[sinal] for h in historico[-config.JANELA_HISTORICO:]
                              if h.get(sinal) is not None and not pd.isna(h[sinal])]
        if len(valores_anteriores) < config.JANELA_HISTORICO // 2:
            continue   # pouco histórico ainda para comparar

        media = mean(valores_anteriores)
        desvio = max(pstdev(valores_anteriores), 1.0)   # mínimo de 1 evita divisão por quase zero
        zscore = (valor - media) / desvio

        mudanca_relevante = abs(valor - media) >= config.VARIACAO_MINIMA[sinal]
        if abs(zscore) > config.LIMITE_ZSCORE and mudanca_relevante:
            direcao = "subiu" if zscore > 0 else "caiu"
            alertas.append(Alerta(
                data_hora=leitura["data_hora"],
                paciente_id=leitura["paciente_id"],
                origem="sinais_vitais",
                tipo=f"Variação brusca de {nome.lower()}",
                gravidade="media",
                descricao=(f"{nome} {direcao} para {valor:.0f} (média recente {media:.0f}; "
                           f"{abs(zscore):.1f} desvios-padrão)."),
            ))

    return alertas


def detectar_por_regras(tabela):
    """Aplica as regras 1 e 2 em todas as leituras, paciente por paciente."""
    alertas = []
    for _, leituras_do_paciente in tabela.groupby("paciente_id"):
        historico = []
        for leitura in leituras_do_paciente.to_dict("records"):
            alertas.extend(verificar_leitura(leitura, historico))
            historico.append(leitura)
    return alertas


def detectar_por_isolation_forest(tabela):
    """Regra 3: modelo Isolation Forest treinado separadamente para cada paciente.
    Leituras que já estão fora dos limites clínicos são ignoradas aqui
    (a regra 1 já gerou alerta para elas)."""
    colunas = colunas_de_sinais(tabela)
    alertas = []

    for paciente_id, leituras in tabela.groupby("paciente_id"):
        dados = leituras.dropna(subset=colunas)
        if len(dados) < 30:
            continue   # poucos dados para treinar o modelo

        modelo = IsolationForest(contamination=config.CONTAMINACAO_ISOLATION_FOREST, random_state=42)
        previsoes = modelo.fit_predict(dados[colunas])   # -1 = anomalia, 1 = normal

        for (_, leitura), previsao in zip(dados.iterrows(), previsoes):
            if previsao != -1 or fora_dos_limites(leitura, colunas):
                continue
            valores = ", ".join(f"{config.NOMES_SINAIS_VITAIS[c]}={leitura[c]:.0f}" for c in colunas)
            alertas.append(Alerta(
                data_hora=leitura["data_hora"],
                paciente_id=str(paciente_id),
                origem="sinais_vitais",
                tipo="Combinação incomum de sinais vitais",
                gravidade="baixa",
                descricao=f"Padrão diferente do habitual deste paciente ({valores}).",
            ))

    return alertas


def fora_dos_limites(leitura, colunas):
    for coluna in colunas:
        minimo, maximo = config.LIMITES_SINAIS_VITAIS[coluna]
        if leitura[coluna] < minimo or leitura[coluna] > maximo:
            return True
    return False


def detectar_anomalias_sinais_vitais(caminho_csv):
    tabela = carregar_sinais_vitais(caminho_csv)
    alertas = detectar_por_regras(tabela) + detectar_por_isolation_forest(tabela)
    alertas = remover_repetidos(alertas)
    print(f"Sinais vitais: {len(tabela)} leituras analisadas, {len(alertas)} alertas.")
    return alertas
