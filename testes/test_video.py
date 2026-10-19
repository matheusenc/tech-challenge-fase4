"""Testes das regras de vídeo com pontos do corpo "de mentira" (sem rodar o YOLO)."""

import numpy as np

from monitoramento.analise_video import (agrupar_em_eventos, dentro_da_area_critica, inclinacao_tronco,
                                         verificar_postura)


def criar_pessoa(ombros_xy, quadril_xy, altura_corpo=400):
    """17 pontos no padrão COCO; só ombros (5, 6) e quadril (11, 12) importam aqui."""
    pontos = np.zeros((17, 2))
    pontos[5] = pontos[6] = ombros_xy
    pontos[11] = pontos[12] = quadril_xy
    return pontos, np.ones(17), altura_corpo


def test_inclinacao_em_pe_e_deitado():
    em_pe, _, _ = criar_pessoa(ombros_xy=(100, 100), quadril_xy=(100, 300))
    deitado, _, _ = criar_pessoa(ombros_xy=(100, 300), quadril_xy=(300, 300))
    assert inclinacao_tronco(em_pe) < 1
    assert 89 < inclinacao_tronco(deitado) < 91


def test_pessoa_em_pe_nao_gera_ocorrencia():
    assert verificar_postura(criar_pessoa((100, 100), (100, 300)), [], segundo=1.0) == []


def test_queda_rapida_gera_possivel_queda():
    historico = []
    verificar_postura(criar_pessoa((100, 100), (100, 300)), historico, segundo=1.0)       # em pé
    tipos = [tipo for tipo, _ in verificar_postura(criar_pessoa((100, 520), (300, 520)),  # 0,5 s depois, no chão
                                                   historico, segundo=1.5)]
    assert "Possível queda" in tipos
    assert "Postura inadequada" in tipos


def test_deitar_devagar_nao_e_queda():
    historico = []
    verificar_postura(criar_pessoa((100, 100), (100, 300)), historico, segundo=1.0)
    tipos = [tipo for tipo, _ in verificar_postura(criar_pessoa((100, 520), (300, 520)), historico, segundo=5.0)]
    assert "Possível queda" not in tipos


def test_area_critica():
    assert dentro_da_area_critica(320, 240, largura=640, altura=480)       # centro da imagem
    assert not dentro_da_area_critica(10, 10, largura=640, altura=480)     # canto


def test_ocorrencias_seguidas_viram_um_evento():
    ocorrencias = [(s / 10, "Postura inadequada", "tronco inclinado 50°") for s in range(10, 21)]  # 1,0 s a 2,0 s
    eventos = agrupar_em_eventos(ocorrencias)
    assert len(eventos) == 1
    assert eventos[0]["inicio_segundos"] == 1.0 and eventos[0]["fim_segundos"] == 2.0


def test_ocorrencia_isolada_e_ruido_exceto_queda():
    eventos = agrupar_em_eventos([(1.0, "Postura inadequada", "x"), (3.0, "Possível queda", "y")])
    assert [e["tipo"] for e in eventos] == ["Possível queda"]
