"""Cria base2_completa.parquet a partir de base_completa.parquet, removendo:
  - sensor_b_aux, temperatura e umidade (e as colunas _imputado e _origem delas)
  - score_saude (redundante com ind_quebra; troque em REDUNDANTE se preferir o contrário)
  - hora_extra
  - observacoes
  - turno
  - colunas testadas no notebook testes/12_imputacao_vibracao_operacao.ipynb, sem relação com as falhas:
      vibracao_mm_s (e _imputado/_origem), origem dos sensores (sensor_*_origem), planta,
      pecas_defeituosas, qtd_operadores, qualidade_leitura, pais_origem (redundante com o modelo)
    (as flags sensor_*_imputado ficam: o modelo usa para ignorar os valores imputados)
A base original não é alterada.

Uso (terminal do VS Code, na pasta T2_pj):
    python criar_base2.py
"""
import sys
from pathlib import Path
import pandas as pd

PASTA = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
ENTRADA = PASTA / "base_completa.parquet"
SAIDA = PASTA / "base2_completa.parquet"

# Colunas removidas por prefixo (pega também _imputado e _origem)
REMOVER_PREFIXO = ("sensor_b_aux", "temp_ambiente_c", "umidade_relativa",
                   "vibracao_mm_s")              # notebook 12: vibração não muda antes da falha

# Colunas removidas pelo final do nome. As flags sensor_*_imputado NÃO saem: o modelo usa.
REMOVER_SUFIXO = ("_origem",)                    # notebook 12: origem do valor não tem relação com falha

# Entre score_saude e ind_quebra, fica só um. Para manter score_saude, troque por "ind_quebra".
REDUNDANTE = "score_saude"

# Colunas removidas pelo nome exato
# Colunas removidas pelo nome exato
REMOVER_EXATO = (REDUNDANTE, "hora_extra", "observacoes", "turno",
                 "data_instalacao",                  # redundante com idade
                 # notebook 12: sem relação com as falhas
                 "tempo_setup_min",
                 "tempo_reconhecimento_medio_min",
                 "versao_firmware_ultima",
                 "capacidade_nominal_h",
                 "produto_fabricado",
                 "numero_serie",
                 "pecas_produzidas",       # tamanho da máquina + idade; sem sinal dentro da máquina
                 "meta_producao",
                 "consumo_energia_kwh",
                 "planta",               # todas as máquinas estão na SP-01
                 "pecas_defeituosas",    # só reflete o tipo de máquina; a taxa de defeito não muda
                 "qtd_operadores",       # só varia com o turno
                 "qualidade_leitura",    # AUC ~0,50
                 "pais_origem")          # mesma informação do modelo e do fabricante (1 país = 1 modelo)

base = pd.read_parquet(ENTRADA)

faltando = [c for c in REMOVER_EXATO if c not in base.columns]
if faltando:
    print("Aviso: estas colunas não existem na base e foram ignoradas:", faltando)

# Colunas que a versão fornecida de features.py usa diretamente.
NECESSARIAS = {
    "vibracao_mm_s", "qualidade_leitura", "tempo_reconhecimento_medio_min",
    "versao_firmware_ultima", "pecas_produzidas", "meta_producao", "pecas_defeituosas",
    "consumo_energia_kwh", "tempo_setup_min", "qtd_operadores", "capacidade_nominal_h",
    "data_instalacao",
}


def sai(c):
    if c in NECESSARIAS:
        return False
    return c.startswith(REMOVER_PREFIXO) or c.endswith(REMOVER_SUFIXO) or c in REMOVER_EXATO


tirar = [c for c in base.columns if sai(c)]
base2 = base.drop(columns=tirar)
base2.to_parquet(SAIDA, index=False)

# conferência: relê o arquivo salvo
check = pd.read_parquet(SAIDA)
assert len(check) == len(base), "número de linhas mudou!"
assert not any(sai(c) for c in check.columns), "sobrou coluna que devia sair!"

print("Colunas removidas (%d):" % len(tirar))
for c in tirar:
    print("  -", c)
print(f"Antes: {base.shape[0]:,} linhas x {base.shape[1]} colunas")
print(f"Depois: {check.shape[0]:,} linhas x {check.shape[1]} colunas")
print("Salvo em:", SAIDA)