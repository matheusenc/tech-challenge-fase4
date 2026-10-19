"""
Gera dados SINTÉTICOS de exemplo na pasta dados/, com anomalias
colocadas de propósito (para saber o que o sistema deveria encontrar).

    python gerar_dados_exemplo.py

Arquivos gerados:
    dados/sinais_vitais.csv   -> 3 pacientes, 48 h, uma leitura a cada 5 min
    dados/prescricoes.csv     -> 3 pacientes, 3 dias de medicação
    dados/movimentacao.csv    -> 3 pacientes, 3 dias, uma leitura por hora
    dados/consulta.txt        -> transcrição de exemplo (usada sem Azure)

Anomalias inseridas:
    Sinais vitais
      P001: taquicardia + queda de saturação (dia 1, 20h)
      P002: pico de pressão arterial (dia 2, 06h)
      P003: salto de frequência cardíaca DENTRO do limite (dia 2, 14h)
            -> só a regra de variação brusca deve pegar
    Prescrições
      P001: dipirona 2000 mg (acima do máximo e aumento brusco)
      P002: morfina introduzida no dia 2 e dose dobrada depois
      P003: paracetamol repetido 1 hora depois (dose duplicada)
    Movimentação
      P001: 5 horas sem se mexer (dia 2, 10h às 15h)
      P002: agitação de madrugada (dia 2, 02h às 04h)
      P003: queda forte de mobilidade no dia 3
"""

import numpy as np
import pandas as pd

from monitoramento import config

INICIO = pd.Timestamp("2026-10-01 00:00")
PACIENTES = ["P001", "P002", "P003"]
gerador = np.random.default_rng(42)   # semente fixa: sempre gera os mesmos dados


def gerar_sinais_vitais():
    linhas = []
    horarios = pd.date_range(INICIO, periods=48 * 12, freq="5min")
    for paciente in PACIENTES:
        for horario in horarios:
            # Pequena variação ao longo do dia (ritmo circadiano) + ruído
            ciclo = np.sin(2 * np.pi * horario.hour / 24)
            linhas.append({
                "data_hora": horario,
                "paciente_id": paciente,
                "frequencia_cardiaca": 75 + 4 * ciclo + gerador.normal(0, 2),
                "pressao_sistolica": 120 + 5 * ciclo + gerador.normal(0, 3),
                "pressao_diastolica": 80 + 3 * ciclo + gerador.normal(0, 2),
                "saturacao_oxigenio": min(100, 97 + gerador.normal(0, 0.6)),
            })
    tabela = pd.DataFrame(linhas)

    def periodo(paciente, inicio, minutos):
        comeco = INICIO + pd.Timedelta(inicio)
        return (tabela["paciente_id"] == paciente) & (tabela["data_hora"] >= comeco) & \
               (tabela["data_hora"] < comeco + pd.Timedelta(minutes=minutos))

    # P001: taquicardia + dessaturação por 40 minutos
    p = periodo("P001", "20h", 40)
    tabela.loc[p, "frequencia_cardiaca"] = 128 + gerador.normal(0, 4, p.sum())
    tabela.loc[p, "saturacao_oxigenio"] = 88 + gerador.normal(0, 1, p.sum())

    # P002: crise hipertensiva por 30 minutos
    p = periodo("P002", "1 days 06h", 30)
    tabela.loc[p, "pressao_sistolica"] = 192 + gerador.normal(0, 4, p.sum())
    tabela.loc[p, "pressao_diastolica"] = 114 + gerador.normal(0, 2, p.sum())

    # P003: salto de FC de ~75 para ~105 (ainda dentro do limite de 120)
    p = periodo("P003", "1 days 14h", 20)
    tabela.loc[p, "frequencia_cardiaca"] = 105 + gerador.normal(0, 2, p.sum())

    colunas = list(config.LIMITES_SINAIS_VITAIS)
    tabela[colunas] = tabela[colunas].round(0)
    return tabela


def gerar_prescricoes():
    linhas = []
    rotina = [  # (medicamento, dose, unidade, de quantas em quantas horas)
        ("dipirona", 1000, "mg", 6),
        ("omeprazol", 40, "mg", 24),
        ("enoxaparina", 40, "mg", 24),
    ]
    for paciente in PACIENTES:
        for medicamento, dose, unidade, intervalo in rotina:
            for horario in pd.date_range(INICIO + pd.Timedelta(hours=8), INICIO + pd.Timedelta(days=3),
                                         freq=f"{intervalo}h"):
                linhas.append([horario, paciente, medicamento, dose, unidade])

    # P001: dipirona 2000 mg no dia 2 às 14h (troca uma dose normal)
    linhas = [l for l in linhas if not (l[1] == "P001" and l[2] == "dipirona"
                                         and l[0] == INICIO + pd.Timedelta("1 days 14h"))]
    linhas.append([INICIO + pd.Timedelta("1 days 14h"), "P001", "dipirona", 2000, "mg"])

    # P002: morfina introduzida no dia 2 (4 mg de 6/6h), dobrada para 8 mg no dia 3
    for horario in pd.date_range(INICIO + pd.Timedelta("1 days 10h"), INICIO + pd.Timedelta("2 days 22h"),
                                 freq="6h"):
        dose = 8 if horario >= INICIO + pd.Timedelta("2 days 10h") else 4
        linhas.append([horario, "P002", "morfina", dose, "mg"])

    # P003: paracetamol de 6/6h, mas uma dose repetida 1 hora depois
    for horario in pd.date_range(INICIO + pd.Timedelta(hours=8), INICIO + pd.Timedelta(days=3), freq="6h"):
        linhas.append([horario, "P003", "paracetamol", 750, "mg"])
    linhas.append([INICIO + pd.Timedelta("1 days 15h"), "P003", "paracetamol", 750, "mg"])

    tabela = pd.DataFrame(linhas, columns=["data_hora", "paciente_id", "medicamento", "dose", "unidade"])
    return tabela.sort_values(["paciente_id", "data_hora"])


def gerar_movimentacao():
    linhas = []
    for paciente in PACIENTES:
        for horario in pd.date_range(INICIO, periods=72, freq="h"):
            noite = horario.hour >= 22 or horario.hour < 6
            atividade = gerador.uniform(5, 15) if noite else gerador.uniform(20, 45)
            linhas.append({"data_hora": horario, "paciente_id": paciente, "nivel_atividade": atividade})
    tabela = pd.DataFrame(linhas)

    def periodo(paciente, inicio, fim):
        return (tabela["paciente_id"] == paciente) & \
               (tabela["data_hora"] >= INICIO + pd.Timedelta(inicio)) & \
               (tabela["data_hora"] < INICIO + pd.Timedelta(fim))

    p = periodo("P001", "1 days 10h", "1 days 15h")
    tabela.loc[p, "nivel_atividade"] = gerador.uniform(0, 2, p.sum())
    p = periodo("P002", "1 days 02h", "1 days 04h")
    tabela.loc[p, "nivel_atividade"] = gerador.uniform(70, 90, p.sum())
    p = periodo("P003", "2 days", "3 days")
    tabela.loc[p, "nivel_atividade"] = tabela.loc[p, "nivel_atividade"] * 0.3

    tabela["nivel_atividade"] = tabela["nivel_atividade"].round(1)
    return tabela


TRANSCRICAO_EXEMPLO = (
    "Bom dia doutor. Eu não estou me sentindo bem desde ontem. "
    "Sinto muito cansaço, qualquer esforço já me deixa sem forças. "
    "À noite eu tive falta de ar e precisei sentar na cama para respirar. "
    "Hoje de manhã senti uma dor no peito que foi para o braço. "
    "Também tive tontura quando levantei. "
    "Estou com medo de piorar."
)


def main():
    config.PASTA_DADOS.mkdir(parents=True, exist_ok=True)
    gerar_sinais_vitais().to_csv(config.PASTA_DADOS / "sinais_vitais.csv", index=False)
    gerar_prescricoes().to_csv(config.PASTA_DADOS / "prescricoes.csv", index=False)
    gerar_movimentacao().to_csv(config.PASTA_DADOS / "movimentacao.csv", index=False)
    (config.PASTA_DADOS / "consulta.txt").write_text(TRANSCRICAO_EXEMPLO, encoding="utf-8")
    print(f"Dados de exemplo gerados em {config.PASTA_DADOS}")


if __name__ == "__main__":
    main()
