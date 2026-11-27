"""
Construção da base de modelagem: uma linha por máquina por dia, com TUDO o que se
sabe até as 05:00 daquele dia (corte). Nenhuma informação posterior ao corte entra.

Uso:
    from features import construir_features
    X = construir_features(df_bruto)        # df_bruto no formato da base2_completa
"""
import warnings
import numpy as np
import pandas as pd
from scipy.signal import lfilter

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

SENS = ["sensor_a", "sensor_b", "sensor_c", "sensor_d", "vibracao_mm_s"]
JANELAS = [6, 24, 72, 168]            # horas
CODIGOS = ["E1", "E2", "E3", "E4", "E5"]
SEV = {"baixa": 1, "media": 2, "alta": 3}
OP_COLS = ["pecas_produzidas", "meta_producao", "pecas_defeituosas",
           "consumo_energia_kwh", "utilizacao_pct", "tempo_setup_min", "qtd_operadores"]
HORA_CORTE = 5
# Filtro casado: a rampa do sensor-sentinela sobe linearmente de ~0 (66h antes da falha) a ~1 (12h antes) e estabiliza.
# Falhas são carimbadas às 06:00 e o corte é 05:00, então o instante da falha em relação ao corte é 1h, 25h, 49h...
TAUS = (1, 25, 49)


def _rampa(h):
    return np.clip((66.0 - h) / 54.0, 0, 1)


# --------------------------------------------------------------------------- limpeza
def limpar(df: pd.DataFrame) -> pd.DataFrame:
    """Limpeza mínima e reprodutível (a base já vem com sensores corrigidos/imputados)."""
    df = df.copy()
    df["id_maquina"] = df["id_maquina"].astype(str).str.strip().str.upper()
    df["data_hora"] = pd.to_datetime(df["data_hora"])
    df = df.drop_duplicates(["id_maquina", "data_hora"]).sort_values(["id_maquina", "data_hora"])
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- helpers
def _wide(df, col, idx):
    """tabela tempo x máquina em grade horária regular (horas sem leitura viram NaN)."""
    return df.pivot(index="data_hora", columns="id_maquina", values=col).reindex(idx)


def _long(frame, nome, cortes):
    """amostra a tabela larga nos cortes e empilha em (dia, maquina)."""
    s = frame.reindex(cortes)
    s.index = s.index.normalize()
    s.index.name = "dia"
    out = s.stack(future_stack=True).rename(nome)
    out.index.names = ["dia", "id_maquina"]
    return out


# --------------------------------------------------------------------------- principal
def construir_features(df: pd.DataFrame, primeiro_dia=None, ultimo_dia=None) -> pd.DataFrame:
    df = limpar(df)
    idx = pd.date_range(df["data_hora"].min().floor("h"), df["data_hora"].max().floor("h"), freq="h")
    maquinas = sorted(df["id_maquina"].unique())

    t0 = idx.min().normalize() + pd.Timedelta(days=3)        # 3 dias de histórico mínimo
    primeiro_dia = pd.Timestamp(primeiro_dia) if primeiro_dia is not None else t0
    ultimo_dia = pd.Timestamp(ultimo_dia) if ultimo_dia is not None else (idx.max() - pd.Timedelta(hours=HORA_CORTE)).normalize()
    dias = pd.date_range(primeiro_dia, ultimo_dia, freq="D")
    cortes = dias + pd.Timedelta(hours=HORA_CORTE)
    cortes = cortes[cortes <= idx.max()]

    cols = {}

    # ---------------- sensores: média/desvio/máximo por janela + z em relação à própria máquina
    for s in SENS:
        W = _wide(df, s, idx)[maquinas]
        base_m = W.expanding(min_periods=48).mean()
        base_s = W.expanding(min_periods=48).std().replace(0, np.nan)
        for w in JANELAS:
            r = W.rolling(w, min_periods=max(2, w // 2))
            m = r.mean()
            cols[f"{s}_med{w}"] = _long(m, f"{s}_med{w}", cortes)
            cols[f"{s}_dp{w}"] = _long(r.std(), f"{s}_dp{w}", cortes)
            cols[f"{s}_max{w}"] = _long(r.max(), f"{s}_max{w}", cortes)
            if w <= 72:
                cols[f"{s}_z{w}"] = _long((m - base_m) / base_s, f"{s}_z{w}", cortes)
            if w == 6:
                z6s = (m - base_m) / base_s
        # onde estava a "rampa" do sensor 12, 24, 36 e 48h antes do corte (só passado)
        for lag in (12, 24, 36, 48):
            cols[f"{s}_z6_lag{lag}"] = _long(z6s.shift(lag), f"{s}_z6_lag{lag}", cortes)
        # correlação das últimas 72h do z horário com a rampa esperada para "falha em tau horas"
        zh = ((W - base_m) / base_s).fillna(0).to_numpy()
        for tau in TAUS:
            w = _rampa(tau + np.arange(72))
            w = w / np.sqrt((w ** 2).sum())
            S = pd.DataFrame(lfilter(w, [1.0], zh, axis=0), index=idx, columns=maquinas)
            cols[f"{s}_mf{tau}"] = _long(S, f"{s}_mf{tau}", cortes)
        cols[f"{s}_tend"] = cols[f"{s}_z6"] - cols[f"{s}_z72"]
        if s == "sensor_a":
            cols["horas_sem_leitura_24h"] = _long(W.isna().astype(float).rolling(24).sum(), "horas_sem_leitura_24h", cortes)

    # ---------------- qualidade da leitura
    susp = _wide(df.assign(x=(df["qualidade_leitura"] == "suspeita").astype(float)), "x", idx)[maquinas]
    cols["leituras_suspeitas_24h"] = _long(susp.rolling(24, min_periods=1).sum(), "leituras_suspeitas_24h", cortes)
    imp_cols = [c for c in df.columns if c.endswith("_imputado")]
    df["_imp"] = df[imp_cols].any(axis=1).astype(float)
    imp = _wide(df, "_imp", idx)[maquinas]
    cols["leituras_imputadas_72h"] = _long(imp.rolling(72, min_periods=1).sum(), "leituras_imputadas_72h", cortes)

    # ---------------- erros / alertas
    E = _wide(df, "n_erros_hora", idx)[maquinas].fillna(0)
    sev_num = df["severidade_max_hora"].astype(object).map(SEV)
    df["_sev"] = sev_num
    SV = _wide(df, "_sev", idx)[maquinas]
    flags = {f"sev_{k}": (SV == v).astype(float) for k, v in SEV.items()}
    for c in CODIGOS:
        df[f"_c_{c}"] = (df["codigo_erro_ultimo"] == c).astype(float)
        flags[f"cod_{c}"] = _wide(df, f"_c_{c}", idx)[maquinas].fillna(0)
    flags["n"] = E
    for w in [24, 72, 168]:
        for nome, F in flags.items():
            cols[f"err_{nome}_{w}h"] = _long(F.rolling(w, min_periods=1).sum(), f"err_{nome}_{w}h", cortes)
    rec = _wide(df, "tempo_reconhecimento_medio_min", idx)[maquinas]
    cols["err_rec_med_72h"] = _long(rec.rolling(72, min_periods=1).mean(), "err_rec_med_72h", cortes)

    # horas desde o último erro, e seu código/severidade
    horas = np.arange(len(idx), dtype=float)[:, None]
    ult = pd.DataFrame(np.where(E.values > 0, horas, np.nan), index=idx, columns=maquinas).ffill()
    desde = (pd.DataFrame(np.repeat(horas, len(maquinas), axis=1), index=idx, columns=maquinas) - ult).fillna(999).clip(upper=999)
    cols["horas_desde_ultimo_erro"] = _long(desde, "horas_desde_ultimo_erro", cortes)
    cod_num = sum(flags[f"cod_{c}"] * (i + 1) for i, c in enumerate(CODIGOS))
    cod_ult = cod_num.where(E > 0).ffill()
    cols["ult_codigo_erro"] = _long(cod_ult, "ult_codigo_erro", cortes)
    cols["ult_sev_erro"] = _long(SV.where(E > 0).ffill(), "ult_sev_erro", cortes)

    # tempo desde o último alerta das 06:00 (proxy do histórico de falhas, disponível também em out-dez/2023):
    # o alerta das 06:00 antecede a falha em 24h, e a máquina tem período refratário de ~14 dias após falhar
    H = pd.DataFrame(np.repeat(horas, len(maquinas), axis=1), index=idx, columns=maquinas)
    h6 = (idx.hour == 6).astype(float)
    def _desde(flag):
        ult = pd.DataFrame(np.where(flag.values > 0, horas, np.nan), index=idx, columns=maquinas).ffill()
        return (H - ult).fillna(3000).clip(upper=3000)
    A6 = E.mul(h6, axis=0)
    cols["horas_desde_alerta06h"] = _long(_desde(A6), "horas_desde_alerta06h", cortes)
    cols["n_alerta06h_30d"] = _long((A6 > 0).astype(float).rolling(720, min_periods=1).sum(), "n_alerta06h_30d", cortes)
    cols["n_alerta06h_90d"] = _long((A6 > 0).astype(float).rolling(2160, min_periods=1).sum(), "n_alerta06h_90d", cortes)
    for c in CODIGOS:
        cols[f"horas_desde_alerta06h_{c}"] = _long(_desde(flags[f"cod_{c}"].mul(h6, axis=0)), f"horas_desde_alerta06h_{c}", cortes)

    # alerta das 06:00 do dia anterior (23h antes do corte): padrão visto na EDA
    # (um erro às 06:00 precede a falha em ~24h, e o código indica a peça)
    for nome, F in flags.items():
        cols[f"alerta_ontem06h_{nome}"] = _long(F.shift(23), f"alerta_ontem06h_{nome}", cortes)
    cols["alerta06h_ult72h"] = _long(E.mul((idx.hour == 6).astype(float), axis=0).rolling(72, min_periods=1).sum(),
                                      "alerta06h_ult72h", cortes)

    # ---------------- operação: apenas turnos COMPLETOS (turno D-2 e anteriores)
    dd = df.groupby(["id_maquina", "data_turno"])[OP_COLS].mean().reset_index()
    dias_op = pd.date_range(dd["data_turno"].min(), dd["data_turno"].max() + pd.Timedelta(days=3), freq="D")
    for c in OP_COLS:
        Wo = dd.pivot(index="data_turno", columns="id_maquina", values=c).reindex(dias_op)[maquinas]
        Wo = Wo.astype(float).rolling(3, min_periods=1).mean().shift(2)     # média dos 3 últimos turnos fechados
        s = Wo.reindex(dias).stack(future_stack=True).rename(f"op_{c}")
        s.index.names = ["dia", "id_maquina"]
        cols[f"op_{c}"] = s

    X = pd.concat(cols.values(), axis=1)
    X.columns = list(cols.keys())
    X["op_prod_vs_meta"] = X["op_pecas_produzidas"] / X["op_meta_producao"]
    X["op_taxa_defeito"] = X["op_pecas_defeituosas"] / X["op_pecas_produzidas"]
    X = X.reset_index()

    # ---------------- cadastro da máquina e idade na data do corte
    cad = df.sort_values("data_hora").groupby("id_maquina").agg(
        modelo=("modelo", "first"), linha_producao=("linha_producao", "first"),
        capacidade_nominal_h=("capacidade_nominal_h", "first"), potencia_kw=("potencia_kw", "first"),
        data_instalacao=("data_instalacao", lambda s: s.dropna().iloc[0] if s.notna().any() else pd.NaT),
        idade_ini=("idade", lambda s: s.dropna().iloc[0] if s.notna().any() else np.nan),
        ini=("data_hora", "min")).reset_index()
    est = cad["ini"] - pd.to_timedelta(cad["idade_ini"].astype(float) * 365.25, unit="D")
    cad["inst"] = pd.to_datetime(cad["data_instalacao"]).fillna(est)
    X = X.merge(cad.drop(columns=["data_instalacao", "idade_ini", "ini"]), on="id_maquina", how="left")
    X["idade_anos"] = (X["dia"] - X["inst"]).dt.days / 365.25
    X = X.drop(columns="inst")

    # firmware vigente no corte
    fw = df[["id_maquina", "data_hora", "versao_firmware_ultima"]].dropna()
    fw = fw.pivot_table(index="data_hora", columns="id_maquina", values="versao_firmware_ultima", aggfunc="last")
    fw = fw.reindex(index=idx, columns=maquinas).ffill()
    fws = fw.reindex(cortes)
    fws.index = fws.index.normalize()
    fws = fws.stack().rename("firmware").reset_index()
    fws.columns = ["dia", "id_maquina", "firmware"]
    X = X.merge(fws, on=["dia", "id_maquina"], how="left")

    X = X.replace([np.inf, -np.inf], np.nan)
    X["dia_semana"] = X["dia"].dt.dayofweek
    for c in ["modelo", "linha_producao", "firmware"]:
        X[c] = X[c].astype("category")
    return X.sort_values(["dia", "id_maquina"]).reset_index(drop=True)


if __name__ == "__main__":
    import os
    dados = os.environ.get("MABIYAMA_DADOS", "./dados").rstrip("/")
    saida = os.environ.get("MABIYAMA_SAIDA", "./saida").rstrip("/")
    os.makedirs(saida, exist_ok=True)
    bruto = pd.read_parquet(f"{dados}/base2_completa.parquet")
    X = construir_features(bruto)
    X.to_parquet(f"{saida}/features_maquina_dia.parquet", index=False)
    print(X.shape, X["dia"].min(), X["dia"].max())
