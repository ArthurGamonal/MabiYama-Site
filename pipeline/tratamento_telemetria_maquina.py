"""Tratamento de telemetria e máquinas -> 2 arquivos parquet.

Uso:
    python tratamento.py                      # lê de Downloads, salva ao lado do script
    python tratamento.py PASTA_ENTRADA        # lê tele.csv e maqui.csv dessa pasta
    python tratamento.py PASTA_ENTRADA PASTA_SAIDA

Regras da telemetria (nesta ordem):
  1. recuperar o que dá para recuperar (sensores a-d):
       - código de erro (-999000, -999, 0)            -> nulo
       - negativo com valor absoluto dentro da faixa   -> sinal trocado: vira positivo
       - fora da faixa por fator 10, 100, 1000...      -> escala errada: divide/multiplica
         (só se o valor corrigido cair perto da mediana da própria máquina)
  2. o que continua fora da faixa (outlier)            -> nulo
  3. mesma máquina e mesma hora repetida               -> UMA linha, com a MÉDIA (nulos ignorados)
  3b. sensor_b nulo com sensor_b_aux válido            -> sensor_b = sensor_b_aux / RAZAO (e o inverso para o aux)
      sensor_b_aux é uma cópia em outra escala do sensor_b (aux ≈ 1,047e-4 × sensor_b; r = 0,9998; erro ~0,1%),
      então é uma leitura real, não um chute: NÃO conta como imputado.
  4. TODO nulo (de origem, código de erro ou outlier)  -> preenchido pelo METODO_IMPUTACAO (padrão: mediana da máquina);
     o que sobrar (buraco longo, ponta da série)      -> mediana da máquina (ou mediana geral)
  Marcações: <coluna>_imputado (True/False) e <coluna>_origem
     (medido | corrigido_sinal | corrigido_escala | recuperado_aux | recuperado_b | imputado_nulo | imputado_outlier)
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
    "sensor_a": (10_000, 100_000),     # era 60_000: cortava a cauda real (pico logo acima de 60 mil).
                                       # Se a faixa 10-60 mil veio do enunciado, volte para 60_000.
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

# --- recuperação (só sensores a-d) ---
SENSORES = ["sensor_a", "sensor_b", "sensor_c", "sensor_d"]
CODIGOS_ERRO = [-999000, -999, 0]      # valores que significam "sensor sem leitura"
TOL_ESCALA = 0.3                       # valor corrigido pela escala tem que ficar a até 30% da
                                       # mediana da máquina; senão continua outlier (vira nulo)

# --- sensor_b <-> sensor_b_aux (passo 3b) ---
RECUPERAR_B_PELO_AUX = True            # False = comportamento antigo (sensor_b nulo vai para a mediana da máquina)
RAZAO_AUX_B_PADRAO = 1.0472e-4         # aux / sensor_b medido nos dados (mesmo valor em todas as máquinas: 1,0471e-4 a 1,0473e-4)
TOL_RAZAO = 0.02                       # só usa pares em que aux/b fica a até 2% da razão (descarta leituras quebradas)

# --- preenchimento de nulos ---
COLS_IMPUTAR = list(LIM_TELE)          # todas as colunas numéricas da telemetria
# "interpolacao"    = linha reta entre a leitura anterior e a seguinte da mesma máquina (usa o futuro:
#                     bom para EDA; para modelo preditivo prefira "ffill")
# "ffill"           = repete a última leitura da máquina (só usa o passado)
# "mediana_maquina" / "media_maquina" = valor fixo da máquina
# Escolha com o teste de imputação (esconder valores conhecidos e medir o erro).
# Teste (esconder valores conhecidos e medir o erro): mediana da máquina teve o menor erro
# (sensores ~0,79 desvio; vibração ~0,60). Interpolação (~0,92-1,00) e ffill (~1,06-1,13) foram piores:
# a leitura de uma hora quase não ajuda a prever a da hora seguinte.
METODO_IMPUTACAO = "mediana_maquina"
MAX_BURACO_H = 24                      # buraco maior que isso não é interpolado/repetido:
                                       # vai para a mediana da máquina

# --- cadastro de máquinas ---
# textos que significam "vazio" (comparados em minúsculas, sem espaços nas pontas)
TOKENS_VAZIO = {"", "na", "n/a", "nan", "none", "null", "-", "--", "?", "sem informacao",
                "sem informação", "nao informado", "não informado", "desconhecido"}

# grafia oficial do fabricante. A chave é o nome sem acento, minúsculo e só com letras/números;
# qualquer valor que COMECE com a chave vira o nome oficial ("Fabrimaq Ltda." -> "Fabrimaq").
# Se aparecer um fabricante que não está aqui, ele é listado no relatório: acrescente-o.
MAPA_FAB_CANON = {"fabrimaq": "Fabrimaq", "metalforte": "Metalforte", "indutec": "Indutec"}

# prefixo do número de série -> fabricante (usado para preencher vazios e conferir)
MAPA_FAB_PREFIXO = {"FBM": "Fabrimaq", "MTF": "Metalforte", "IDT": "Indutec"}

# potência fixa de cada modelo (usado para preencher modelo vazio)
MAPA_MODELO_POTENCIA = {22.0: "modelo_a", 18.5: "modelo_b", 30.0: "modelo_c"}
MODELOS_VALIDOS = ["modelo_a", "modelo_b", "modelo_c", "modelo_d"]

# "I1" (letra i) tratado como digitação errada de "L1". PENDENTE: confirmar.
TRATAR_I_COMO_L = True

# Data de referência para calcular a idade pela data de instalação.
# None = o script testa datas candidatas e escolhe a que mais bate com as idades já preenchidas
# (o valor escolhido e a % de acerto aparecem no relatório). Para fixar: pd.Timestamp("2023-01-01").
DATA_REF_IDADE = None


# ======================= TELEMETRIA =======================
def limpar_milhar(s):
    """'1.974.539' / '1736,809' -> sem separador de milhar; '419,20' -> 419.20"""
    s = s.str.strip()
    s = s.str.replace(r"[.,](?=\d{3}(?:[.,]\d{3})*$)", "", regex=True)
    s = s.str.replace(r"^(\d+),(\d{1,2})$", r"\1.\2", regex=True)
    return pd.to_numeric(s, errors="coerce")


def recuperar_sensores(df, rel):
    """Códigos de erro -> nulo; sinal trocado -> positivo; escala errada -> corrigida.
    Cria colunas auxiliares _corr_<sensor> ("" / "sinal" / "escala")."""
    maq = df["id_maquina"]
    for c in SENSORES:
        mn, mx = LIM_TELE[c]
        corr = pd.Series("", index=df.index, dtype=object)

        # 1) código de erro -> nulo (ANTES do sinal: |-999000| cairia dentro da faixa do sensor_b)
        codigo = df[c].isin(CODIGOS_ERRO)
        rel[f"codigo_erro_virou_nulo_{c}"] = int(codigo.sum())
        df[c] = df[c].mask(codigo)

        # 2) sinal trocado
        sinal = (df[c] < 0) & df[c].abs().between(mn, mx)
        df.loc[sinal, c] = df.loc[sinal, c].abs()
        corr[sinal] = "sinal"
        rel[f"corrigido_sinal_{c}"] = int(sinal.sum())

        # 3) escala errada (fator 10^k em relação à mediana da própria máquina)
        dentro = df[c].between(mn, mx)
        med = df[c].where(dentro).groupby(maq).transform("median")
        fora = (df[c] > 0) & ~dentro & med.notna()
        k = pd.Series(0.0, index=df.index)
        k[fora] = np.round(np.log10(df.loc[fora, c] / med[fora]))
        cand = df[c] / (10.0 ** k)
        escala = (fora & (k != 0) & cand.between(mn, mx)
                  & ((cand / med - 1).abs() <= TOL_ESCALA))
        df.loc[escala, c] = cand[escala]
        corr[escala] = "escala"
        rel[f"corrigido_escala_{c}"] = int(escala.sum())
        if escala.any():
            rel[f"corrigido_escala_{c}_fatores"] = (
                "; ".join(f"10^{int(f)}: {n}" for f, n in k[escala].value_counts().sort_index().items()))

        df[f"_corr_{c}"] = corr
    return df


def recuperar_b_pelo_aux(df, rel):
    """Preenche sensor_b nulo com sensor_b_aux / razão e sensor_b_aux nulo com sensor_b × razão.
    Razão estimada por máquina nas horas em que os dois são válidos (cai na razão geral se faltar)."""
    b, a = df["sensor_b"], df["sensor_b_aux"]
    r = a / b
    ok = b.notna() & a.notna() & ((r / RAZAO_AUX_B_PADRAO - 1).abs() <= TOL_RAZAO)
    razao_geral = float(r[ok].median()) if ok.any() else RAZAO_AUX_B_PADRAO
    razao = r.where(ok).groupby(df["id_maquina"]).transform("median").fillna(razao_geral)
    rel["razao_aux_sobre_b"] = f"{razao_geral:.4e} (pares válidos: {int(ok.sum())})"

    rec_b = b.isna() & a.notna()
    df.loc[rec_b, "sensor_b"] = a[rec_b] / razao[rec_b]
    rec_a = a.isna() & b.notna() & ~rec_b
    df.loc[rec_a, "sensor_b_aux"] = b[rec_a] * razao[rec_a]
    # o valor reconstruído tem que respeitar a faixa; senão volta a ser nulo
    ruim_b = rec_b & ~df["sensor_b"].between(*LIM_TELE["sensor_b"])
    ruim_a = rec_a & ~df["sensor_b_aux"].between(*LIM_TELE["sensor_b_aux"])
    df.loc[ruim_b, "sensor_b"], df.loc[ruim_a, "sensor_b_aux"] = np.nan, np.nan
    rec_b, rec_a = rec_b & ~ruim_b, rec_a & ~ruim_a
    df["_rec_sensor_b"], df["_rec_sensor_b_aux"] = rec_b, rec_a
    rel["sensor_b_recuperado_pelo_aux"] = int(rec_b.sum())
    rel["sensor_b_aux_recuperado_pelo_b"] = int(rec_a.sum())
    rel["sensor_b_sem_recuperacao (b e aux nulos)"] = int(df["sensor_b"].isna().sum())
    return df


def imputar(df, c):
    """Preenche os nulos de df[c] (df ordenado por máquina e hora, sem hora repetida).
    Devolve (série preenchida, máscara do que o método principal preencheu)."""
    s = df[c]
    maq = df["id_maquina"]
    if METODO_IMPUTACAO == "interpolacao":
        valores = s.to_numpy(dtype=float)
        datas = df["data_hora"].to_numpy()
        res = valores.copy()
        for idx in df.groupby("id_maquina", sort=False).indices.values():
            sub = pd.Series(valores[idx], index=pd.DatetimeIndex(datas[idx]))
            res[idx] = sub.interpolate(method="time", limit=MAX_BURACO_H,
                                       limit_area="inside").to_numpy()
        novo = pd.Series(res, index=s.index)
    elif METODO_IMPUTACAO == "ffill":
        novo = s.groupby(maq).ffill(limit=MAX_BURACO_H)
    elif METODO_IMPUTACAO == "media_maquina":
        novo = s.fillna(s.groupby(maq).transform("mean"))
    elif METODO_IMPUTACAO == "mediana_maquina":
        novo = s.fillna(s.groupby(maq).transform("median"))
    else:
        raise ValueError(f"METODO_IMPUTACAO desconhecido: {METODO_IMPUTACAO}")

    principal = s.isna() & novo.notna()
    # o que sobrou (buraco longo, começo/fim da série) -> mediana da máquina -> mediana geral
    novo = novo.fillna(s.groupby(maq).transform("median")).fillna(s.median())
    return novo, principal


def limpar_telemetria(bruto):
    df = bruto.copy()
    rel = {"linhas_entrada": len(df), "metodo_imputacao": METODO_IMPUTACAO}

    # --- chave, datas, texto ---
    df["id_maquina"] = df["id_maquina"].str.strip().str.upper()
    df["data_hora"] = pd.to_datetime(df["data_hora"], format="mixed", dayfirst=True, errors="coerce")
    rel["linhas_removidas_data_invalida"] = int(df["data_hora"].isna().sum())
    rel["linhas_removidas_id_vazio"] = int((df["id_maquina"].isna() | (df["id_maquina"] == "")).sum())
    df = df.dropna(subset=["data_hora"])
    df = df[df["id_maquina"].notna() & (df["id_maquina"] != "")]

    df["qualidade_leitura"] = df["qualidade_leitura"].str.strip().str.lower()
    invalida = df["qualidade_leitura"].notna() & ~df["qualidade_leitura"].isin(["ok", "suspeita"])
    rel["qualidade_invalida_virou_nulo"] = int(invalida.sum())
    df["qualidade_leitura"] = df["qualidade_leitura"].mask(invalida)

    # --- converter números (sensores a-d têm separador de milhar) ---
    for c in SENSORES:
        df[c] = limpar_milhar(df[c])
    for c in ["sensor_b_aux", "temp_ambiente_c", "umidade_relativa", "vibracao_mm_s"]:
        df[c] = pd.to_numeric(df[c].str.strip().str.replace(",", ".", regex=False), errors="coerce")

    # --- duplicatas exatas ---
    antes = len(df)
    df = df.drop_duplicates()
    rel["duplicatas_removidas"] = antes - len(df)
    df = df.reset_index(drop=True)

    # --- 1) recuperar sinal trocado / escala errada; código de erro -> nulo ---
    df = recuperar_sensores(df, rel)

    # --- 2) o que continua fora da faixa -> nulo (marcado em _out_<coluna>) ---
    for c, (mn, mx) in LIM_TELE.items():
        fora = df[c].notna() & ~df[c].between(mn, mx)
        rel[f"outlier_virou_nulo_{c}"] = int(fora.sum())
        df[f"_out_{c}"] = fora
        df[c] = df[c].mask(fora)

    # --- 3) mesma máquina e mesma hora mais de uma vez -> UMA linha com a MÉDIA ---
    # Motivo: as leituras repetidas são quase iguais e não há como saber qual está certa;
    # a média não depende da ordem do arquivo. Como os outliers já viraram nulo,
    # a média usa só leituras válidas. qualidade_leitura = "ok" se alguma for "ok".
    chave = ["id_maquina", "data_hora"]
    repetida = df.duplicated(chave, keep=False)
    rel["horas_repetidas_viraram_media"] = int(df.loc[repetida, chave].drop_duplicates().shape[0])
    if repetida.any():
        numericas = list(LIM_TELE)
        agregacao = {}
        for c in df.columns:
            if c in chave:
                continue
            if c in numericas:
                agregacao[c] = "mean"
            elif c.startswith("_out_") or c.startswith("_corr_"):
                agregacao[c] = "max"
            elif c == "qualidade_leitura":
                agregacao[c] = lambda s: ("ok" if (s == "ok").any()
                                          else (s.dropna().iloc[0] if s.notna().any() else np.nan))
            else:
                agregacao[c] = "first"
        agrupadas = (df[repetida].groupby(chave, sort=False).agg(agregacao)
                     .reset_index()[df.columns])
        df = pd.concat([df[~repetida], agrupadas], ignore_index=True)

    df = df.sort_values(chave).reset_index(drop=True)

    # --- 3b) sensor_b <-> sensor_b_aux (leitura real em outra escala, não é imputação) ---
    if RECUPERAR_B_PELO_AUX:
        df = recuperar_b_pelo_aux(df, rel)

    # --- 4) TODO nulo -> preenchido; origem de cada valor registrada ---
    for c in COLS_IMPUTAR:
        nulo = df[c].isna()
        era_outlier = df[f"_out_{c}"].astype(bool)
        origem = pd.Series("medido", index=df.index, dtype=object)
        if f"_corr_{c}" in df.columns:
            origem[df[f"_corr_{c}"] == "sinal"] = "corrigido_sinal"
            origem[df[f"_corr_{c}"] == "escala"] = "corrigido_escala"
        origem[nulo & era_outlier] = "imputado_outlier"
        origem[nulo & ~era_outlier] = "imputado_nulo"
        if f"_rec_{c}" in df.columns:
            origem[df[f"_rec_{c}"].astype(bool)] = "recuperado_aux" if c == "sensor_b" else "recuperado_b"

        df[c], principal = imputar(df, c)
        df[f"{c}_imputado"] = nulo
        df[f"{c}_origem"] = origem
        rel[f"imputado_{c}"] = int(nulo.sum())
        rel[f"imputado_{c}_pelo_metodo"] = int(principal.sum())
        rel[f"imputado_{c}_pela_mediana_maquina"] = int((nulo & ~principal).sum())

    df = df.drop(columns=[c for c in df.columns if c.startswith(("_out_", "_corr_", "_rec_"))])
    rel["linhas_saida"] = len(df)
    return df, rel


# ======================= MÁQUINAS =======================
def _chave(s):
    """'  Fabrimaq Ltda. ' -> 'fabrimaqltda' (sem acento, minúsculo, só letras e números)."""
    return (s.str.normalize("NFKD").str.encode("ascii", "ignore").str.decode("ascii")
             .str.lower().str.replace(r"[^a-z0-9]", "", regex=True))


def _vazio_para_nulo(s):
    s = s.str.strip().str.replace(r"\s+", " ", regex=True)
    return s.mask(s.str.lower().isin(TOKENS_VAZIO))


def padronizar_fabricante(s):
    chave = _chave(s)
    saida = pd.Series(np.nan, index=s.index, dtype=object)
    for k, nome in MAPA_FAB_CANON.items():
        saida = saida.mask(saida.isna() & chave.str.startswith(k, na=False), nome)
    # fora do mapa: mantém o texto, com a primeira letra maiúscula
    resto = saida.isna() & s.notna()
    saida[resto] = s[resto].str.title()
    return saida


def padronizar_linha(s):
    """'LINHA 3', 'linha-3', 'L 3', 'l3' -> 'L3'. O que não casar fica como estava (em maiúsculas)."""
    t = (s.str.normalize("NFKD").str.encode("ascii", "ignore").str.decode("ascii")
          .str.upper().str.strip())
    prefixos = "LINHA|LIN|L" + ("|I" if TRATAR_I_COMO_L else "")
    num = t.str.extract(rf"^(?:{prefixos})\s*[-_.]?\s*0*(\d+)$")[0]
    return ("L" + num).where(num.notna(), t)


def _idade_pela_data(data, ref, inteiro):
    anos = (ref - data).dt.days / 365.25
    return np.floor(anos) if inteiro else anos.round(1)


def estimar_data_ref(idade, data):
    """Testa o 1º dia de cada mês de 2015 a 2026 e devolve o que mais acerta as idades existentes."""
    ok = idade.notna() & data.notna()
    if ok.sum() == 0:
        return pd.Timestamp("2023-01-01"), np.nan, 0
    inteiro = bool((idade[ok] % 1 == 0).all())
    melhor, acerto = None, -1.0
    for ref in pd.date_range("2015-01-01", "2026-12-01", freq="MS"):
        calc = _idade_pela_data(data[ok], ref, inteiro)
        taxa = float(((calc - idade[ok]).abs() < (0.5 if inteiro else 0.15)).mean())
        if taxa > acerto:
            melhor, acerto = ref, taxa
    return melhor, acerto, int(ok.sum())


def limpar_cadastro(bruto):
    df = bruto.copy()
    rel = {"linhas_entrada": len(df)}

    df["id_maquina"] = df["id_maquina"].str.strip().str.upper()
    for c in ["modelo", "numero_serie", "fabricante", "planta", "linha_producao", "pais_origem",
              "data_instalacao", "idade", "capacidade_nominal_h", "potencia_kw"]:
        if c in df.columns:
            df[c] = _vazio_para_nulo(df[c])

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

    # padronização de categorias (guarda o "antes" para o relatório de-para)
    fab_bruto, linha_bruta = df["fabricante"].copy(), df["linha_producao"].copy()
    df["modelo"] = df["modelo"].str.lower().str.replace(r"[\s\-]+", "_", regex=True)
    df["linha_producao"] = padronizar_linha(df["linha_producao"])
    df["fabricante"] = padronizar_fabricante(df["fabricante"])

    # preenchimentos derivados de outra coluna (não é chute)
    vazio = df["modelo"].isna()
    df["modelo"] = df["modelo"].fillna(df["potencia_kw"].map(MAPA_MODELO_POTENCIA))
    rel["modelo_preenchido_pela_potencia"] = int((vazio & df["modelo"].notna()).sum())

    fab_prefixo = df["numero_serie"].str[:3].str.upper().map(MAPA_FAB_PREFIXO)
    vazio = df["fabricante"].isna()
    df["fabricante"] = df["fabricante"].fillna(fab_prefixo)
    rel["fabricante_preenchido_pelo_prefixo"] = int((vazio & df["fabricante"].notna()).sum())
    rel["fabricante_diverge_do_prefixo"] = int(
        (df["fabricante"].notna() & fab_prefixo.notna() & (df["fabricante"] != fab_prefixo)).sum())

    # idade vazia (de origem ou outlier) -> calculada pela data de instalação
    ref = DATA_REF_IDADE
    if ref is None:
        ref, acerto, n = estimar_data_ref(df["idade"], df["data_instalacao"])
        rel["idade_data_ref_estimada"] = ref.date().isoformat()
        rel["idade_data_ref_acerto_pct"] = round(100 * acerto, 1) if n else "sem base"
    else:
        rel["idade_data_ref_fixa"] = pd.Timestamp(ref).date().isoformat()
    existentes = df["idade"].dropna()
    inteiro = existentes.empty or bool((existentes % 1 == 0).all())
    calc = _idade_pela_data(df["data_instalacao"], pd.Timestamp(ref), inteiro)
    calc = calc.mask((calc < LIM_CAD["idade"][0]) | (calc > LIM_CAD["idade"][1]))
    vazio = df["idade"].isna()
    df["idade"] = df["idade"].fillna(calc)
    rel["idade_preenchida_pela_data"] = int((vazio & df["idade"].notna()).sum())
    rel["idade_continua_nula"] = int(df["idade"].isna().sum())

    # modelo fora da lista -> nulo
    invalido = df["modelo"].notna() & ~df["modelo"].isin(MODELOS_VALIDOS)
    rel["modelo_invalido_virou_nulo"] = int(invalido.sum())
    df["modelo"] = df["modelo"].mask(invalido)

    # de-para das categorias (para conferir no relatório)
    rel["_depara_fabricante"] = (pd.DataFrame({"bruto": fab_bruto, "final": df["fabricante"]})
                                 .value_counts(dropna=False).sort_index())
    rel["_depara_linha"] = (pd.DataFrame({"bruto": linha_bruta, "final": df["linha_producao"]})
                            .value_counts(dropna=False).sort_index())

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
    origem = pd.DataFrame({c: tele[f"{c}_origem"].value_counts() for c in COLS_IMPUTAR}).fillna(0).astype(int)
    print("\n--- origem dos valores (quantas linhas)")
    print(origem.to_string())
    print("\n=== Máquinas ===")
    depara = {k: rel_maq.pop(k) for k in [k for k in rel_maq if k.startswith("_depara")]}
    print(pd.Series(rel_maq).to_string())
    for nome, tabela in depara.items():
        print(f"\n--- {nome[1:]} (bruto -> final, quantidade)")
        print(tabela.to_string())
    print("\nIDs da telemetria sem cadastro:", sem_cadastro or "nenhum")
    print("Máquinas sem nenhuma leitura:", len(set(maq["id_maquina"]) - set(tele["id_maquina"])))

    # checagens das categorias (avisam, não param a execução)
    fabs = sorted(maq["fabricante"].dropna().unique())
    linhas = sorted(maq["linha_producao"].dropna().unique())
    fora_linha = [x for x in linhas if not pd.Series([x]).str.fullmatch(r"L\d+").iloc[0]]
    print("\nfabricantes:", len(fabs), fabs)
    print("linhas:", linhas)
    if fora_linha:
        print("AVISO: linha fora do padrão L<n>:", fora_linha)
    print("idade nula: %.1f%%" % (100 * maq["idade"].isna().mean()))

    # salvar
    tele.to_parquet(destino / "telemetria_tratada.parquet", index=False)
    maq.to_parquet(destino / "maquinas_tratado.parquet", index=False)
    print(f"\nSalvo em: {destino}")
    print("  telemetria_tratada.parquet", tele.shape)
    print("  maquinas_tratado.parquet", maq.shape)


if __name__ == "__main__":
    main()