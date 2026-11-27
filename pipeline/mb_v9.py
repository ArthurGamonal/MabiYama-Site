"""mb_v9.py -- funções dos 3 experimentos v9 (usadas por treinar_v9.py).

  A) add_sinal_v9          : features novas do sensor-sentinela (filtro casado com baseline LOCAL, correlação normalizada,
                             resíduo da rampa). Precisa de features_extra_v9.parquet (gerado por features_v9.py).
  B) ablacao_familias      : seleção de colunas por FAMÍLIA com CV temporal (remove famílias que pioram o AP).
  C) add_candidatos_v9     : features calculadas SÓ entre as linhas candidatas (grade ligada) + evidência entre sensores,
                             combinadas com a poda/duplicação do selecao.py.
Nada aqui lê rótulo do futuro: tudo usa só colunas que já existem no corte das 05:00.
"""
import re
import numpy as np
import pandas as pd
import xgboost as xgb
from mb import PECAS, SENS, ap_por_peca

# ============================================================================ A) sinal
SV9_NOMES = ["lmf1", "lmf13", "lmf25", "lmf37", "lmf49", "lncc1", "lncc25", "lncc49", "lres25",
             "lz6", "lz24", "lz6_lag24", "lz6_lag48", "lsobe24", "lsobe48"]
SV9_SEM_SINAL = {"lres25"}                        # resíduo é sempre >= 0: não multiplica pelo sinal do sentinela
SV9_AUX = ["lmf25", "lncc25", "lz6", "lsobe24"]   # sentinela auxiliar (sensor_b etc.)


def add_sinal_v9(d: pd.DataFrame, sent: dict, ex: pd.DataFrame) -> pd.DataFrame:
    """Junta as colunas por sensor de features_extra_v9 e escolhe, por peça, a do sensor-sentinela (com o sinal).
    Sai com sv9_* / sv9b_* e SEM as colunas brutas por sensor (para não inchar o modelo)."""
    d = d.copy()
    d["_m"] = d["id_maquina"].astype(str)
    e = ex.rename(columns={"id_maquina": "_m"})
    raw = [c for c in e.columns if c not in ("_m", "dia")]
    q = d.merge(e, on=["_m", "dia"], how="left")
    assert len(q) == len(d), "features_extra_v9 tem chave (dia, máquina) duplicada"
    pec = q["peca"].astype(str).to_numpy()
    new = {}
    for p in PECAS:
        m = pec == p
        if not m.any():
            continue
        for chave, nomes, pref in ((p, SV9_NOMES, "sv9_"), (p + "__aux", SV9_AUX, "sv9b_")):
            if chave not in sent:
                continue
            s, sg = sent[chave]
            for nome in nomes:
                col = f"{s}_{nome}"
                if col not in q.columns:
                    continue
                f = 1.0 if nome in SV9_SEM_SINAL else float(sg)
                arr = new.setdefault(pref + nome, np.full(len(q), np.nan))
                arr[m] = f * q.loc[m, col].to_numpy(dtype=float)
    # contrastes: "parece falha em 25h" vs "em 1h/49h" (timing) e vs "em 13h/37h" (nitidez do timing)
    if {"sv9_lmf25", "sv9_lmf1", "sv9_lmf49"} <= set(new):
        new["sv9_lmf25_vs"] = new["sv9_lmf25"] - np.fmax(new["sv9_lmf1"], new["sv9_lmf49"])
    if {"sv9_lmf25", "sv9_lmf13", "sv9_lmf37"} <= set(new):
        new["sv9_lmf25_vs2"] = new["sv9_lmf25"] - np.fmax(new["sv9_lmf13"], new["sv9_lmf37"])
    q = q.drop(columns=["_m"] + raw)
    return pd.concat([q, pd.DataFrame(new, index=q.index).astype("float32")], axis=1)


# ============================================================================ C) candidatos
def add_candidatos_v9(d: pd.DataFrame, mask) -> pd.DataFrame:
    """mask = linhas candidatas (grade ligada). O treino só vê candidatas, então ranks/z calculados contra o dia INTEIRO
    (inclui ~93% de linhas que nunca falham) ficam numa escala diferente da que o modelo realmente disputa."""
    d = d.copy()
    if "sent_mf25" not in d.columns:
        return d
    mk = np.asarray(mask, dtype=bool)
    x = d["sent_mf25"].astype(float)
    xc = x.where(mk)
    dia, pec, maq = d["dia"], d["peca"].astype(str), d["id_maquina"].astype(str)
    new = {}
    g_dia = xc.groupby(dia)
    new["cv_mf25_rk_cand"] = g_dia.rank(pct=True)
    new["cv_mf25_z_cand"] = (xc - g_dia.transform("mean")) / g_dia.transform("std").replace(0, np.nan)
    new["cv_mf25_rk_cand_peca"] = xc.groupby([dia, pec]).rank(pct=True)
    if "sx_ev_media" in d.columns:                       # evidência de 2 dias seguidos (item 5 do v8a)
        e = d["sx_ev_media"].astype(float).where(mk)
        ge = e.groupby(dia)
        new["cv_ev_rk_cand"] = ge.rank(pct=True)
        new["cv_ev_z_cand"] = (e - ge.transform("mean")) / ge.transform("std").replace(0, np.nan)
    # qual das 4 peças da máquina carrega o evento (softmax suave do sentinela dentro da máquina-dia)
    mx = x.groupby([maq, dia]).transform("max")
    new["cv_maq_max_mf25"] = mx
    ee = np.exp(np.clip(2.0 * (x - mx), -30, 0))
    new["cv_share_mf25"] = ee / ee.groupby([maq, dia]).transform("sum")
    # a máquina tem algum evento em QUALQUER sensor? e o sentinela é o sensor mais forte?
    sc = [f"{s}_mf25" for s in SENS if f"{s}_mf25" in d.columns]
    if sc:
        anyv = d[sc].astype(float).abs().max(axis=1)
        new["cv_any_mf25"] = anyv
        new["cv_sent_menos_any"] = x - anyv
    if "grade_conf_m" in d.columns:
        new["cv_mf25_x_conf"] = x * d["grade_conf_m"].astype(float)
    return pd.concat([d.drop(columns=[c for c in new if c in d.columns]),
                      pd.DataFrame(new, index=d.index).astype("float32")], axis=1)


# ============================================================================ B) ablação por família
PROTEGIDAS = {"peca", "grade", "gt"}


def familia(c: str) -> str:
    c = c.split("__dup")[0]
    if c == "peca":
        return "peca"
    if c.startswith("nbd_"):
        return "nbd"
    if c.startswith("nbp_"):
        return "nbp"
    if c.startswith("nb_"):
        return "nb"
    if re.search(r"_(rk|dmax|dmed)$", c):
        return "competicao"
    if c.startswith("sv9"):
        return "sinal_v9"
    if c.startswith("cv_"):
        return "cand_v9"
    if c.startswith("sx_"):
        return "sx"
    if c.startswith("fx_"):
        return "fx"
    if c.startswith("sentb_"):
        return "sentb"
    if c.startswith("sent_"):
        return "sent"
    if c.startswith(("err_peca_", "alerta_peca", "alerta_outro_cod", "horas_desde_alerta_peca", "maq_n_alerta")):
        return "alerta_peca"
    if c.startswith("grade_"):
        return "grade"
    if c.startswith("gt_"):
        return "gt"
    if "_z6_lag" in c:
        return "sens_zlag"
    m = re.match(r"^(sensor_[a-d]|vibracao_mm_s)_(med|dp|max|z|tend|mf)", c)
    if m:
        return "sens_" + m.group(2)
    if c.startswith("alerta_ontem06h"):
        return "alerta_ontem"
    if c.startswith(("horas_desde_alerta06h", "n_alerta06h", "alerta06h")):
        return "alerta_hist"
    if c.startswith("err_"):
        return "err_" + c.split("_")[1]
    if c in ("horas_desde_ultimo_erro", "ult_codigo_erro", "ult_sev_erro"):
        return "ult_erro"
    if c.startswith(("horas_sem_leitura", "leituras_")):
        return "leitura"
    if c.startswith("op_"):
        return "op"
    if c.startswith("hist_"):
        return "hist"
    return "cadastro_outras"


def ablacao_familias(trva, cols, params, mascara_fn, protegidas=PROTEGIDAS, n_folds=3, janela=45, purga=2,
                     trees=250, seeds=(0, 1), min_gain=0.002, min_folds=2, max_rounds=4):
    """Eliminação regressiva por família, medida em CV temporal (janelas deslizantes de `janela` dias dentro de treino+validação).
    Só remove uma família se o ganho médio >= min_gain E melhorar em >= min_folds folds (para não perseguir ruído).
    Treina só nas linhas candidatas (igual ao treino real). Devolve (colunas_finais, relatorio)."""
    d = trva[mascara_fn(trva)].reset_index(drop=True)
    dias = np.sort(d["dia"].unique())
    fim = len(dias)
    folds = []
    for k in range(n_folds):
        v_fim = fim - janela * (n_folds - 1 - k)
        v_ini = v_fim - janela
        a = d[d["dia"].isin(dias[: v_ini - purga])]
        b = d[d["dia"].isin(dias[v_ini:v_fim])]
        folds.append((a, b))
        print(f"[ablação] fold {k}: treino {len(a)} linhas ({int(a['y'].sum())} pos) | val {len(b)} linhas ({int(b['y'].sum())} pos)", flush=True)
    pr = dict(params, learning_rate=max(float(params.get("learning_rate", 0.03)) * 2, 0.05))
    base_kw = dict(tree_method="hist", enable_categorical=True, n_jobs=-1, verbosity=0, n_estimators=trees)

    def score(cs):
        out = []
        for a, b in folds:
            v = []
            for s in seeds:
                m = xgb.XGBClassifier(**base_kw, **pr, random_state=11 + s).fit(a[cs], a["y"])
                v.append(ap_por_peca(b, m.predict_proba(b[cs])[:, 1])[1])
            out.append(float(np.mean(v)))
        return np.array(out)

    fam = {c: familia(c) for c in cols}
    fams = sorted(set(fam.values()) - set(protegidas))
    atual = list(cols)
    base = score(atual)
    rel = {"base_folds": base.round(4).tolist(), "passos": []}
    print(f"[ablação] base: folds {base.round(4).tolist()} média {base.mean():.4f}", flush=True)
    for r in range(max_rounds):
        res = {}
        for f in fams:
            sub = [c for c in atual if fam[c] != f]
            if len(sub) == len(atual) or not sub:
                continue
            res[f] = score(sub) - base
            print(f"[ablação] r{r} sem {f:16s} ({len(atual) - len(sub):3d} cols): Δ média {res[f].mean():+.4f} | Δ folds {res[f].round(4).tolist()}", flush=True)
        ok = {f: v for f, v in res.items() if v.mean() >= min_gain and (v > 0).sum() >= min_folds}
        if not ok:
            print(f"[ablação] r{r}: nenhuma família passa na regra (>= {min_gain} e {min_folds} folds). Fim.", flush=True)
            break
        melhor = max(ok, key=lambda f: ok[f].mean())
        atual = [c for c in atual if fam[c] != melhor]
        fams.remove(melhor)
        base = base + res[melhor]
        rel["passos"].append({"removida": melhor, "delta_medio": float(res[melhor].mean()), "colunas_restantes": len(atual)})
        print(f"[ablação] r{r}: REMOVE {melhor} | novo base {base.mean():.4f} | {len(atual)} colunas", flush=True)
    rel["familias_removidas"] = [p["removida"] for p in rel["passos"]]
    rel["colunas_finais"] = len(atual)
    return atual, rel
