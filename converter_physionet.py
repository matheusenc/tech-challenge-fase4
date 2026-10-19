"""
Converte dados REAIS do PhysioNet para o formato usado pelo projeto.

Dataset usado: BIDMC PPG and Respiration Dataset (acesso aberto)
    https://physionet.org/content/bidmc/1.0.0/
Baixe a pasta bidmc_csv/ e use os arquivos *_Numerics.csv, que trazem
uma leitura por segundo com as colunas: Time [s], HR, PULSE, RESP, SpO2.

    python converter_physionet.py caminho/bidmc_csv --saida dados/sinais_vitais_physionet.csv

Depois rode:
    python main.py --sinais dados/sinais_vitais_physionet.csv

Observação: esse dataset não tem pressão arterial; o sistema analisa
somente as colunas que existirem (frequência cardíaca e saturação).
"""

import argparse
from pathlib import Path

import pandas as pd

INICIO = pd.Timestamp("2026-10-01 08:00")


def converter_arquivo(caminho_csv, paciente_id):
    tabela = pd.read_csv(caminho_csv)
    tabela.columns = [coluna.strip() for coluna in tabela.columns]   # remove espaços dos nomes

    return pd.DataFrame({
        "data_hora": INICIO + pd.to_timedelta(tabela["Time [s]"], unit="s"),
        "paciente_id": paciente_id,
        "frequencia_cardiaca": tabela["HR"],
        "saturacao_oxigenio": tabela["SpO2"],
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pasta", type=Path, help="Pasta com os arquivos bidmc_XX_Numerics.csv")
    parser.add_argument("--saida", type=Path, default=Path("dados/sinais_vitais_physionet.csv"))
    args = parser.parse_args()

    arquivos = sorted(args.pasta.glob("*_Numerics.csv"))
    if not arquivos:
        print(f"Nenhum arquivo *_Numerics.csv encontrado em {args.pasta}")
        return

    tabelas = []
    for arquivo in arquivos:
        paciente_id = "BIDMC_" + arquivo.name.split("_")[1]   # bidmc_01_Numerics.csv -> BIDMC_01
        tabelas.append(converter_arquivo(arquivo, paciente_id))

    resultado = pd.concat(tabelas).dropna(subset=["frequencia_cardiaca", "saturacao_oxigenio"], how="all")
    args.saida.parent.mkdir(parents=True, exist_ok=True)
    resultado.to_csv(args.saida, index=False)
    print(f"{len(arquivos)} pacientes, {len(resultado)} leituras salvas em {args.saida}")


if __name__ == "__main__":
    main()
