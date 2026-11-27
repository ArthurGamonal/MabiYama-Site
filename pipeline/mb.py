"""Funções compartilhadas de modelagem (rótulo, split temporal, features por peça, métricas)."""
import os
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss

# pasta com base2_completa.parquet e falhas.parquet / pasta de saída (mude por variável de ambiente)
U = os.environ.get("MABIYAMA_DADOS", "./dados").rstrip("/") + "/"
OUT = os.environ.get("MABIYAMA_SAIDA", "./saida").rstrip("/") + "/"
os.makedirs(OUT, exist_ok=True)
PECAS = ["peca_A", "peca_B", "peca_C", "peca_D"]
SENS = ["sensor_a", "sensor_b", "sensor_c", "sensor_d", "vibracao_mm_s"]
CODIGOS = ["E1", "E2", "E3", "E4", "E5"]
HORA_CORTE = 5
ULTIMO_DIA_ROTULADO = pd.Timestamp("2023-09-29")   # janela de D vai até D+1 06:00; falhas conhecidas até 30/09 06:00
CATS = ["modelo", "linha_producao", "firmware", "peca"]


# --------------------------------------------------------------------------- rótulo
# Modos (variável MABIYAMA_ROTULO):
#   vespera   (padrão): falha em (D 06:00, D+1 06:00] conta UMA vez, no dia D. A falha carimbada às 06:00 de X é do dia X-1.
#                       É o alvo do Kaggle: os ids do sample_submission vão de 30/09 a 30/12, e um modelo treinado com
#                       o rótulo "duplo" pontua ~0,10 contra ele (igual à nota 0,12 observada) -- ver diag_rotulo.py.
#                       O alerta das 06:00 de D (que antecede a falha em 24h) chega 1h DEPOIS do corte: não é visível.
#   mesmo_dia          : falha em [D 06:00, D+1 06:00) conta no dia D (alerta visível; ~0,86 de AP, mas NÃO é o alvo do Kaggle).
#   duplo              : janela literal (D 05:00, D+1 06:00]; conta em dois dias.
MODO_ROTULO = os.environ.get("MABIYAMA_ROTULO", "vespera")


def rotular(falhas: pd.DataFrame, modo: str | None = None) -> pd.DataFrame:
    modo = modo or MODO_ROTULO
    f = falhas.copy()
    f["data_hora"] = pd.to_datetime(f["data_hora"])
    t = f["data_hora"]
    if modo in ("vespera", "mesmo_dia"):
        desl = pd.Timedelta(hours=6, seconds=1) if modo == "vespera" else pd.Timedelta(hours=6)
        d = (t - desl).dt.normalize()
        x = f[["id_maquina", "peca"]].copy()
        x["dia"] = d
        x["antecedencia_h"] = (t - (d + pd.Timedelta(hours=HORA_CORTE))).dt.total_seconds() / 3600
        r = x
    elif modo == "duplo":
        linhas = []
        for off in (0, 1):
            d = t.dt.normalize() - pd.Timedelta(days=off)
            ini, fim = d + pd.Timedelta(hours=HORA_CORTE), d + pd.Timedelta(days=1, hours=6)
            ok = (t > ini) & (t <= fim)
            x = f.loc[ok, ["id_maquina", "peca"]].copy()
            x["dia"] = d[ok]
            x["antecedencia_h"] = (t[ok] - ini[ok]).dt.total_seconds() / 3600
            linhas.append(x)
        r = pd.concat(linhas)
    else:
        raise ValueError(f"MABIYAMA_ROTULO inválido: {modo}")
    return r.sort_values("antecedencia_h").drop_duplicates(["id_maquina", "peca", "dia"], keep="first")


def rotular_iminente(falhas: pd.DataFrame) -> pd.DataFrame:
    """Falhas que acontecem na hora seguinte ao corte (D 05:00 < t <= D 06:00).
    NÃO é alvo: serve só para descobrir, no treino, quais códigos de alerta apontam para cada peça."""
    f = falhas.copy()
    f["data_hora"] = pd.to_datetime(f["data_hora"])
    t = f["data_hora"]
    linhas = []
    for off in (0, 1):
        d = t.dt.normalize() - pd.Timedelta(days=off)
        ok = (t > d + pd.Timedelta(hours=HORA_CORTE)) & (t <= d + pd.Timedelta(hours=6))
        x = f.loc[ok, ["id_maquina", "peca"]].copy()
        x["dia"] = d[ok]
        linhas.append(x)
    r = pd.concat(linhas).drop_duplicates()
    r["iminente"] = 1
    return r


def montar_longo(X: pd.DataFrame, falhas: pd.DataFrame | None, modo: str | None = None) -> pd.DataFrame:
    """máquina x dia  ->  máquina x dia x peça, com alvo y, antecedência e flag auxiliar `iminente`."""
    L = pd.concat([X.assign(peca=p) for p in PECAS], ignore_index=True)
    if falhas is not None:
        r = rotular(falhas, modo)
        L = L.merge(r, on=["id_maquina", "peca", "dia"], how="left")
        L["y"] = L["antecedencia_h"].notna().astype(int)
        im = rotular_iminente(falhas)
        L = L.merge(im, on=["id_maquina", "peca", "dia"], how="left")
        L["iminente"] = L["iminente"].fillna(0).astype(int)
    L["peca"] = pd.Categorical(L["peca"], categories=PECAS)
    return L


# --------------------------------------------------------------------------- split temporal 70/15/15
def split_temporal(dias_rotulados, p_tr=0.70, p_va=0.15, purga=1):
    d = pd.DatetimeIndex(sorted(pd.unique(dias_rotulados)))
    n = len(d)
    n_tr, n_va = int(round(n * p_tr)), int(round(n * p_va))
    tr = d[:n_tr]
    va = d[n_tr + purga: n_tr + n_va]
    te = d[n_tr + n_va + purga:]
    return tr, va, te


# --------------------------------------------------------------------------- features específicas da peça
def derivar_sentinelas(L_tr: pd.DataFrame):
    """Descobre, só com o TREINO, qual sensor (e em que direção) denuncia cada peça,
    e quais códigos de alerta apontam para ela."""
    sent, alerta = {}, {}
    for p in PECAS:
        g = L_tr[L_tr["peca"] == p]
        melhor = None
        for s in SENS:
            z = g[f"{s}_z24"]
            d = (z[g["y"] == 1].mean() - z[g["y"] == 0].mean()) / z.std()
            if melhor is None or abs(d) > abs(melhor[1]):
                melhor = (s, d)
        sent[p] = (melhor[0], 1.0 if melhor[1] > 0 else -1.0)
        taxa = {}
        for c in CODIGOS:
            m = g[f"alerta_ontem06h_cod_{c}"] == 1
            taxa[c] = g.loc[m, "iminente"].mean() if m.sum() >= 10 else 0.0
        top = max(taxa.values())
        cods = [c for c, v in taxa.items() if top > 0 and v >= 0.3 * top]
        alerta[p] = cods
    return sent, alerta


SENT_ASSINADAS = ["z6", "z24", "z72", "tend", "z6_lag12", "z6_lag24", "z6_lag36", "z6_lag48", "mf1", "mf25", "mf49"]
SENT_BRUTAS = ["med6", "med24", "dp24", "max24"]


def add_peca_features(L: pd.DataFrame, sent, alerta) -> pd.DataFrame:
    L = L.copy()
    for nome in SENT_ASSINADAS + SENT_BRUTAS:
        L[f"sent_{nome}"] = np.nan
    L["alerta_peca"] = 0.0
    for w in (24, 72, 168):
        L[f"err_peca_{w}h"] = 0.0
    for p in PECAS:
        m = (L["peca"] == p).to_numpy()
        s, sg = sent[p]
        for nome in SENT_ASSINADAS:
            L.loc[m, f"sent_{nome}"] = sg * L.loc[m, f"{s}_{nome}"].to_numpy()
        for nome in SENT_BRUTAS:
            L.loc[m, f"sent_{nome}"] = L.loc[m, f"{s}_{nome}"].to_numpy()
        for c in alerta[p]:
            L.loc[m, "alerta_peca"] += L.loc[m, f"alerta_ontem06h_cod_{c}"].to_numpy()
            for w in (24, 72, 168):                       # erros com o código desta peça
                L.loc[m, f"err_peca_{w}h"] += L.loc[m, f"err_cod_{c}_{w}h"].to_numpy()
    # tempo desde o último alerta das 06:00 com o código desta peça (proxy de "dias desde a última falha da peça")
    L["horas_desde_alerta_peca"] = 3000.0
    for p in PECAS:
        if not alerta[p]:
            continue
        k = (L["peca"] == p).to_numpy()
        L.loc[k, "horas_desde_alerta_peca"] = np.minimum.reduce([L.loc[k, f"horas_desde_alerta06h_{c}"].to_numpy() for c in alerta[p]])
    # alerta de ontem 06h: quantos erros (1/2/3 => 9% / 57% / 81% de chance) e se o código é o desta peça
    n = L["alerta_ontem06h_n"].fillna(0)
    L["alerta_peca_n"] = (L["alerta_peca"] > 0) * n
    L["alerta_outro_cod"] = ((n > 0) & (L["alerta_peca"] == 0)) * n
    # idade da "rampa" do sensor: quanto subiu nas últimas 24h / 48h
    L["sent_sobe_24h"] = L["sent_z6"] - L["sent_z6_lag24"]
    L["sent_sobe_48h"] = L["sent_z6"] - L["sent_z6_lag48"]
    # a rampa parece mais "falha em 25h" do que "falha em 1h" ou "em 49h"?
    L["sent_mf25_vs"] = L["sent_mf25"] - np.maximum(L["sent_mf1"], L["sent_mf49"])
    return L


def add_historico(L: pd.DataFrame, falhas: pd.DataFrame) -> pd.DataFrame:
    """Propensão histórica de falha (SÓ PASSADO): falhas da máquina / da peça até o corte das 05:00,
    por 1000 h de exposição, com encolhimento para a média da frota. Na previsão (sem falhas novas
    conhecidas) o histórico fica congelado na última falha conhecida.
    DESLIGADO por padrão (na validação não melhorou o AP); ligue com MABIYAMA_HIST=1."""
    if os.environ.get("MABIYAMA_HIST", "0") != "1":
        return L
    L = L.copy()
    f = falhas.copy()
    f["data_hora"] = pd.to_datetime(f["data_hora"])
    t_ult = f["data_hora"].max()
    t0 = pd.Timestamp("2022-10-01 06:00")
    corte = (L["dia"] + pd.Timedelta(hours=HORA_CORTE)).clip(upper=t_ult)
    c_ns = corte.to_numpy().astype("datetime64[ns]")
    expos_k = np.clip((c_ns - np.datetime64(t0, "ns")).astype("timedelta64[s]").astype(float) / 3600, 24, None) / 1000
    maq = L["id_maquina"].to_numpy()
    pec = L["peca"].astype(str).to_numpy()
    cnt_m, cnt_p = np.zeros(len(L)), np.zeros(len(L))
    for m, g in f.groupby("id_maquina"):
        km = maq == m
        cnt_m[km] = np.searchsorted(np.sort(g["data_hora"].to_numpy().astype("datetime64[ns]")), c_ns[km], side="right")
        for p, gp in g.groupby("peca"):
            kp = km & (pec == p)
            cnt_p[kp] = np.searchsorted(np.sort(gp["data_hora"].to_numpy().astype("datetime64[ns]")), c_ns[kp], side="right")
    horas_tot = (t_ult - t0).total_seconds() / 3600 / 1000
    n_maq = L["id_maquina"].nunique()
    r_all = len(f) / (n_maq * horas_tot)                                  # falhas por 1000 h por máquina
    r_pec = (f["peca"].value_counts() / (n_maq * horas_tot)).to_dict()
    prior_p = pd.Series(pec).map(r_pec).fillna(r_all / 4).to_numpy()
    L["hist_taxa_maq"] = (cnt_m + r_all) / (expos_k + 1)
    L["hist_taxa_peca"] = (cnt_p + prior_p) / (expos_k + 1)
    L["hist_taxa_peca_rel"] = L["hist_taxa_peca"] / prior_p
    return L


def colunas_modelo(L: pd.DataFrame, unificado: bool):
    excl = {"dia", "id_maquina", "y", "antecedencia_h", "iminente"}
    cols = [c for c in L.columns if c not in excl]
    if not unificado:
        cols = [c for c in cols if c != "peca" and not c.startswith("sent_") and c != "alerta_peca"]
    return cols


# --------------------------------------------------------------------------- métricas
def metricas(y, p):
    y, p = np.asarray(y), np.asarray(p)
    return dict(ap=average_precision_score(y, p), auc=roc_auc_score(y, p),
                brier=brier_score_loss(y, p), taxa_base=float(y.mean()), n_pos=int(y.sum()), n=int(len(y)))


def melhor_limiar(y, p):
    from sklearn.metrics import precision_recall_curve
    pr, rc, th = precision_recall_curve(y, p)
    f1 = 2 * pr[:-1] * rc[:-1] / np.clip(pr[:-1] + rc[:-1], 1e-12, None)
    i = int(np.nanargmax(f1))
    return float(th[i]), float(f1[i])


def pr_rec_f1(y, p, limiar):
    a = np.asarray(p) >= limiar
    y = np.asarray(y)
    vp, fp, fn = int((a & (y == 1)).sum()), int((a & (y == 0)).sum()), int((~a & (y == 1)).sum())
    pr = vp / max(vp + fp, 1)
    rc = vp / max(vp + fn, 1)
    return dict(limiar=limiar, vp=vp, fp=fp, fn=fn, precisao=pr, recall=rc,
                f1=2 * pr * rc / max(pr + rc, 1e-12), alertas=int(a.sum()))


def topk_diario(L: pd.DataFrame, p, ks=(5, 10, 20, 40)):
    """Plano do dia: inspecionar as K peças (máquina x peça) de maior risco por dia."""
    d = L[["dia", "y"]].copy()
    d["p"] = p
    d["rank"] = d.groupby("dia")["p"].rank(ascending=False, method="first")
    tot = d["y"].sum()
    return {k: dict(recall=float(d.loc[d["rank"] <= k, "y"].sum() / tot),
                    precisao=float(d.loc[d["rank"] <= k, "y"].mean())) for k in ks}


# --------------------------------------------------------------------------- predição (reutilizável pela interface)
def prever_prob(art: dict, L: pd.DataFrame) -> np.ndarray:
    """art: artefato salvo por final.py. L: base longa (máquina x dia x peça) já com add_peca_features."""
    ps = []
    if art["tipo"] in ("unificado", "ensemble"):
        ps.append(art["modelo_unificado"].predict_proba(L[art["cols_unificado"]])[:, 1])
    if art["tipo"] in ("por_peca", "ensemble"):
        p = np.zeros(len(L))
        for peca, m in art["modelos_peca"].items():
            k = (L["peca"] == peca).to_numpy()
            p[k] = m.predict_proba(L.loc[k, art["cols_peca"]])[:, 1]
        ps.append(p)
    return np.mean(ps, axis=0)


def ap_por_peca(L: pd.DataFrame, p):
    """Métrica do Kaggle: average precision de cada peça e a média das quatro."""
    p = np.asarray(p)
    out = {}
    for pc in PECAS:
        k = (L["peca"] == pc).to_numpy()
        yk = L.loc[k, "y"]
        out[pc] = float(average_precision_score(yk, p[k])) if yk.sum() > 0 else float("nan")
    return out, float(np.nanmean(list(out.values())))


def montar_submissao(out: pd.DataFrame, caminho_sample: str | None = None) -> pd.DataFrame:
    """Formato do Kaggle: id = MAQUINA_AAAA-MM-DD; colunas peca_A..peca_D. Segue a ordem/ids do sample_submission."""
    sub = (out.assign(id=out["id_maquina"].astype(str) + "_" + out["data_previsao"].dt.strftime("%Y-%m-%d"),
                      peca=out["peca"].astype(str))
              .pivot(index="id", columns="peca", values="prob_falha")[PECAS])
    if caminho_sample and os.path.exists(caminho_sample):
        ex = pd.read_csv(caminho_sample)
        falta = set(ex["id"]) - set(sub.index)
        if falta:
            raise ValueError(f"{len(falta)} ids do sample_submission sem previsão (ex.: {sorted(falta)[:3]})")
        sub = sub.loc[ex["id"]]
    return sub.reset_index()
