"""Tratamento de telemetria e máquinas -> 2 arquivos parquet.

Uso:
    python tratamento.py                      # lê de Downloads, salva ao lado do script
    python tratamento.py PASTA_ENTRADA        # lê tele.csv e maqui.csv dessa pasta
    python tratamento.py PASTA_ENTRADA PASTA_SAIDA

Regras:
  - valor fora da faixa (outlier)            -> vira nulo e continua nulo
  - nulo de origem em COLS_MEDIA             -> média da máquina (ou média geral)
  - cada preenchimento é marcado em <coluna>_imputado
Requer: pip install pandas numpy pyarrow
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ======================= CONFIGURAÇÃO =======================
PASTA_PADRAO = Path(r"C:\Users\rafae\Downloads")

# faixas válidas: fora disso = outlier = nulo
LIM_TELE = {
    "sensor_a": (10_000, 60_000),      # REVISAR: pode estar cortando leituras boas (veja a mediana por máquina)
    "sensor_b": (500_000, 4_000_000),
    "sensor_c": (50_000, 350_000),
    "sensor_d": (50_000, 500_000),
    "sensor_b_aux": (50, 400),
    "temp_ambiente_c": (5, 40),
    "umidade_relativa": (0, 100),
    "vibracao_mm_s": (0.0001, 15),     # > 0 e <= 15
}
LIM_CAD = {
    "idade": (0, 50),
    "capacidade_nominal_h": (50, 250),
    "potencia_kw": (0, 10_000),
}

# colunas em que o nulo de origem é preenchido com a média
COLS_MEDIA = ["sensor_a", "sensor_b", "sensor_c", "sensor_d", "sensor_b_aux"]


# ======================= TELEMETRIA =======================
def limpar_milhar(s):
    """'1.974.539' / '1736,809' -> sem separador de milhar; '419,20' -> 419.20"""
    s = s.str.strip()
    s = s.str.replace(r"[.,](?=\d{3}(?:[.,]\d{3})*$)", "", regex=True)
    s = s.str.replace(r"^(\d+),(\d{1,2})$", r"\1.\2", regex=True)
    return pd.to_numeric(s, errors="coerce")


def limpar_telemetria(bruto):
    df = bruto.copy()
    rel = {"linhas_entrada": len(df)}

    # --- chave, datas, texto ---
    df["id_maquina"] = df["id_maquina"].str.strip().str.upper()
    datas = df["data_hora"].str.strip()
    iso = datas.str.match(r"^\d{4}-", na=False)
    df["data_hora"] = pd.to_datetime(datas.where(iso), format="ISO8601", errors="coerce").fillna(
        pd.to_datetime(datas.where(~iso), format="mixed", dayfirst=True, errors="coerce")
    )
    rel["linhas_removidas_data_invalida"] = int(df["data_hora"].isna().sum())
    rel["linhas_removidas_id_vazio"] = int((df["id_maquina"].isna() | (df["id_maquina"] == "")).sum())
    df = df.dropna(subset=["data_hora"])
    df = df[df["id_maquina"].notna() & (df["id_maquina"] != "")]

    df["qualidade_leitura"] = df["qualidade_leitura"].str.strip().str.lower()
    invalida = df["qualidade_leitura"].notna() & ~df["qualidade_leitura"].isin(["ok", "suspeita"])
    rel["qualidade_invalida_virou_nulo"] = int(invalida.sum())
    df["qualidade_leitura"] = df["qualidade_leitura"].mask(invalida)

    # --- converter números (sensores a-d têm separador de milhar) ---
    for c in ["sensor_a", "sensor_b", "sensor_c", "sensor_d"]:
        df[c] = limpar_milhar(df[c])
    for c in ["sensor_b_aux", "temp_ambiente_c", "umidade_relativa", "vibracao_mm_s"]:
        df[c] = pd.to_numeric(df[c].str.strip().str.replace(",", ".", regex=False), errors="coerce")

    # --- duplicatas ---
    antes = len(df)
    df = df.drop_duplicates()
    rel["duplicatas_removidas"] = antes - len(df)
    df = df.reset_index(drop=True)

    # --- 1) guarda quem JÁ era nulo (antes de zerar os outliers) ---
    nulo_original = {c: df[c].isna() for c in LIM_TELE}

    # --- 2) outlier -> nulo ---
    for c, (mn, mx) in LIM_TELE.items():
        fora = df[c].notna() & ((df[c] < mn) | (df[c] > mx))
        rel[f"outlier_virou_nulo_{c}"] = int(fora.sum())
        df[c] = df[c].mask(fora)

    # --- 3) nulo de origem -> média (só quem já era nulo, nunca o outlier) ---
    for c in COLS_MEDIA:
        media_maq = df.groupby("id_maquina")[c].transform("mean")
        media_geral = df[c].mean()
        preencher = nulo_original[c]
        df[f"{c}_imputado"] = preencher
        df[c] = df[c].where(~preencher, media_maq.fillna(media_geral))
        rel[f"preenchido_media_{c}"] = int(preencher.sum())

    df = df.sort_values(["id_maquina", "data_hora"]).reset_index(drop=True)
    rel["linhas_saida"] = len(df)
    return df, rel


# ======================= MÁQUINAS =======================
def limpar_cadastro(bruto):
    df = bruto.copy()
    rel = {"linhas_entrada": len(df)}

    df["id_maquina"] = df["id_maquina"].str.strip().str.upper()
    for c in ["modelo", "numero_serie", "fabricante", "planta", "linha_producao", "pais_origem"]:
        df[c] = df[c].str.strip().replace("", np.nan)

    # números
    for c in ["idade", "capacidade_nominal_h", "potencia_kw"]:
        df[c] = pd.to_numeric(df[c].str.strip().str.replace(",", ".", regex=False), errors="coerce")

    # outlier -> nulo
    for c, (mn, mx) in LIM_CAD.items():
        fora = df[c].notna() & ((df[c] < mn) | (df[c] > mx))
        rel[f"outlier_virou_nulo_{c}"] = int(fora.sum())
        df[c] = df[c].mask(fora)

    # data de instalação (AAAA-MM-DD, ano 1960-2026)
    di = df["data_instalacao"].str.strip()
    data = pd.to_datetime(di, format="%Y-%m-%d", errors="coerce")
    rel["data_instalacao_ilegivel_virou_nulo"] = int((di.notna() & data.isna()).sum())
    fora_ano = data.notna() & ~data.dt.year.between(1960, 2026)
    rel["data_instalacao_ano_fora_virou_nulo"] = int(fora_ano.sum())
    df["data_instalacao"] = data.mask(fora_ano)

    # padronização de categorias
    df["modelo"] = df["modelo"].str.lower().str.replace(" ", "_")
    df["linha_producao"] = df["linha_producao"].str.upper().replace({"I1": "L1"})   # confirme antes

    # preenchimentos derivados de outra coluna (não é chute)
    mapa_modelo = {22.0: "modelo_a", 18.5: "modelo_b", 30.0: "modelo_c"}
    df["modelo"] = df["modelo"].fillna(df["potencia_kw"].map(mapa_modelo))
    mapa_fab = {"FBM": "Fabrimaq", "MTF": "Metalforte", "IDT": "Indutec"}
    df["fabricante"] = df["fabricante"].fillna(df["numero_serie"].str[:3].str.upper().map(mapa_fab))

    # modelo fora da lista -> nulo
    invalido = df["modelo"].notna() & ~df["modelo"].isin(["modelo_a", "modelo_b", "modelo_c", "modelo_d"])
    rel["modelo_invalido_virou_nulo"] = int(invalido.sum())
    df["modelo"] = df["modelo"].mask(invalido)

    # ids duplicados: mantém o primeiro
    rel["ids_duplicados_removidos"] = int(df["id_maquina"].duplicated().sum())
    df = df.drop_duplicates("id_maquina", keep="first").reset_index(drop=True)

    rel["linhas_saida"] = len(df)
    return df, rel


# ======================= EXECUÇÃO =======================
def main():
    pasta = Path(sys.argv[1]) if len(sys.argv) > 1 else PASTA_PADRAO
    destino = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).parent
    destino.mkdir(parents=True, exist_ok=True)

    # leitura (tudo como texto, para o pandas não converter nada sozinho)
    def achar(*nomes):
        for n in nomes:
            if (pasta / n).exists():
                return pasta / n
        raise FileNotFoundError(f"nenhum de {nomes} em {pasta}")

    tele_bruto = pd.read_csv(achar("tele.csv", "telemetria.csv"), dtype=str)
    cad_bruto = pd.read_csv(achar("maqui.csv", "maquinas.csv"), dtype=str)

    tele, rel_tele = limpar_telemetria(tele_bruto)
    maq, rel_maq = limpar_cadastro(cad_bruto)

    # checagens
    sem_cadastro = set(tele["id_maquina"]) - set(maq["id_maquina"])
    print("=== Telemetria ===")
    print(pd.Series(rel_tele).to_string())
    print("\n=== Máquinas ===")
    print(pd.Series(rel_maq).to_string())
    print("\nIDs da telemetria sem cadastro:", sem_cadastro or "nenhum")
    print("Máquinas sem nenhuma leitura:", len(set(maq["id_maquina"]) - set(tele["id_maquina"])))

    # salvar
    tele.to_parquet(destino / "telemetria_tratada.parquet", index=False)
    maq.to_parquet(destino / "maquinas_tratado.parquet", index=False)
    print(f"\nSalvo em: {destino}")
    print("  telemetria_tratada.parquet", tele.shape)
    print("  maquinas_tratado.parquet", maq.shape)


if __name__ == "__main__":
    main()
