"""treinar_v8a.py -- VERSÃO 8 (itens 2, 5, 6).
Base: treinar.py v7 (AP teste 0,755 / Kaggle 0,65). Mudanças, numeradas como na lista de sugestões:
  2) fase de 15 dias + REANCORAGEM por picos do sensor-sentinela (dias/visitas desde o último pico, nº de picos)   [FASE15]
  5) reforço do sentinela: evidência de 2 dias seguidos (mf25 de hoje + mf49 de ontem), diferenças, ranking no dia [SENT_EXTRA]
  6) estabilidade: 9 sementes em 3 variantes de hiperparâmetros e nº de árvores FIXO por variante (mediana)        [ESTABILIDADE]
  3) poda: remove colunas de importância ~0 (cumulativa > PODA_ACUM no CSV do v7)                                  [PODAR = False]
  4) sem colunas "impressão digital" da máquina (idade_anos, taxas históricas, fornecedor/técnico, firmware...)   [SEM_IDENTIDADE = False]
Arquivos na mesma pasta: mb.py, mb_extra.py, mb_troca.py, mb_nb26.py | saída: *_v8a.* (não sobrescreve o v7).
Para a poda é preciso existir OUT/importancia_features_v7.csv (gerado pelo treinar v7).
"""
import json, time, warnings, joblib
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.metrics import average_precision_score
from mb import *
from mb_extra import add_competicao, COMP_COLS, mascara_grade
from mb_troca import add_troca, diagnostico_troca
from mb_nb26 import NB26
from pathlib import Path
from selecao import SelecaoFeatures, SUF

warnings.filterwarnings("ignore")
T0 = time.time()
def marca(msg): print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)

# ================================ CONFIGURAÇÃO ================================
VERSAO = "v8b"
PARAMS = dict(learning_rate=0.03, max_depth=4, min_child_weight=10, subsample=0.8, colsample_bytree=0.5,
              reg_lambda=5, reg_alpha=0.1, gamma=0.5)
ESTABILIDADE = True                                   # item 6
N_SEEDS = 9 if ESTABILIDADE else 3
PARAMS_VAR = [PARAMS,
              dict(PARAMS, max_depth=3, colsample_bytree=0.4, min_child_weight=12),
              dict(PARAMS, max_depth=5, colsample_bytree=0.6, min_child_weight=8, gamma=1.0)] if ESTABILIDADE else [PARAMS]
EARLY_STOP = True
USAR_GRADE, USAR_TROCA, USAR_COMPETICAO = True, True, True
FILTRO_GRADE, TREINAR_SO_GRADE, MASCARA, PISO_GRADE = True, True, "grade", 1e-3
EXTRA_COMP = ["gt_idade_dias"]
USAR_NB_ESTATICO = True
USAR_NB_DEP = True
USAR_RISCO = False        # suspeito de vazamento: manter False
COMPARAR_BASELINE = False # o v7 já foi medido; evita 1 treino extra
FASE15 = True             # item 2
SENT_EXTRA = True         # item 5
PODAR = False         # item 3
PODA_ACUM = 0.90          # mantém as colunas que formam 90% da importância do v7 (~280 de 502)
SEM_IDENTIDADE = False  # item 4
SELECAO = True            # poda por importância (CSV do v8a) + duplicação das features mais importantes
IMP_MIN, N_DUP, COPIAS = 5e-4, 0, 0   # varrer: (0,0,0) controle | (5e-4,0,0) só poda | (5e-4,20,2) | (1e-3,30,1)
IDENT = ["idade_anos", "nbd_hist_taxa", "nbd_hist_taxa_total", "nbd_inst_taxa_forn", "nbd_inst_taxa_tec",
         "firmware", "modelo", "linha_producao"]
# (nbd_inst_idade = dias desde a última troca/manutenção DA PEÇA: é sinal legítimo, NÃO entra em IDENT)
# ==============================================================================
if not USAR_GRADE:
    FILTRO_GRADE = TREINAR_SO_GRADE = USAR_TROCA = False
USAR_NB = USAR_NB_ESTATICO or USAR_NB_DEP
USAR_EXTRA = FASE15 or SENT_EXTRA
KEY = ["id_maquina", "peca", "dia"]
NOVAS = []

def mascara(L):
    m = np.ones(len(L), dtype=bool)
    if MASCARA in ("grade", "ambas") and "grade_on_m" in L.columns:
        m &= mascara_grade(L)
    if MASCARA in ("troca", "ambas") and "gt_on_m" in L.columns:
        m &= (L["gt_on_m"] != 0).to_numpy()
    return m

def aplicar(L, p):
    p = np.asarray(p, dtype=float)
    return np.where(mascara(L), p, p * PISO_GRADE) if FILTRO_GRADE else p

def so_grade(d):
    return d[mascara(d)] if TREINAR_SO_GRADE else d

def features(d, s, a):
    d = add_peca_features(d, s, a)
    if USAR_COMPETICAO:
        d = add_competicao(d, COMP_COLS + EXTRA_COMP)
    return d

CHAVE = ["id_maquina", "dia", "peca"]
def add_dep(d, corte, extra):
    if not (USAR_NB and USAR_NB_DEP):
        return d
    keys = d[["id_maquina", "dia"]].drop_duplicates().reset_index(drop=True)
    return d.merge(nb.dep_long(keys, corte=corte, extra=extra), on=CHAVE, how="left")

# >>> EXTRA (itens 2 e 5) -------------------------------------------------------------------------------
def limiares_pico(d):
    """Por peça: limiares de sent_mf1 que marcam 'a falha acabou de acontecer' (06:00 de hoje).
    Calibrados só com rótulos do bloco passado (y de ontem == 1 => mf1 de hoje é o pico da falha)."""
    out = {}
    if "sent_mf1" not in d.columns:
        return out
    t = d[["id_maquina", "peca", "dia", "y", "sent_mf1"]].sort_values(["id_maquina", "peca", "dia"]).copy()
    t["yp"] = t.groupby(["id_maquina", "peca"], observed=True)["y"].shift(1)
    t["pc"] = t["peca"].astype(str)
    for p, s in t.groupby("pc"):
        v, ref = s.loc[s["yp"] == 1, "sent_mf1"].dropna(), s["sent_mf1"].dropna()
        lo, hi = (v.quantile(0.10), v.quantile(0.50)) if len(v) >= 15 else (ref.quantile(0.99), ref.quantile(0.995))
        out[p] = (float(lo), float(hi))
    return out

def add_extra(parts, thr):
    """parts: [tr, va, te] (ou [Lf, Q]) em ordem cronológica, já com sent_*/gt_*. Calcula features que precisam de
    continuidade entre blocos usando SÓ o passado (sensores existem em out-dez), e devolve a lista com colunas novas."""
    global NOVAS
    base = pd.concat([p.assign(_b=i) for i, p in enumerate(parts)], ignore_index=True)
    core = base.sort_values(KEY + ["_b"], kind="stable").drop_duplicates(KEY).reset_index(drop=True)
    gk = [core["id_maquina"], core["peca"]]
    g = core.groupby(["id_maquina", "peca"], sort=False, observed=True)
    ok = (core["dia"] - g["dia"].shift(1)).dt.days == 1          # lag de 1 linha só vale se for o dia anterior
    new = {}
    if FASE15:
        for nome, col in (("fx_idade", "gt_idade_dias"), ("fx_idmaq", "gt_idade_maq_dias")):
            if col in core.columns:
                a = core[col].astype(float)
                fase = np.mod(a, 15)
                new[f"{nome}_fase15"] = fase
                new[f"{nome}_dist15"] = np.mod(15 - fase, 15)
                new[f"{nome}_perto15"] = np.minimum(fase, np.mod(15 - fase, 15))
        if "sent_mf1" in core.columns and thr:
            lo = core["peca"].astype(str).map(lambda p: thr.get(p, (np.nan, np.nan))[0])
            hi = core["peca"].astype(str).map(lambda p: thr.get(p, (np.nan, np.nan))[1])
            for nome, lim in (("lo", lo), ("hi", hi)):
                flag = (core["sent_mf1"] >= lim)
                ult = core["dia"].where(flag).groupby(gk, observed=True).ffill()
                dsp = (core["dia"] - ult).dt.days.astype(float)  # NaN = nunca visto
                new[f"fx_dsp_{nome}"] = dsp
                new[f"fx_vis_{nome}"] = np.ceil(dsp / 15)         # mesmo conceito de nbd_rec_*: visitas desde o pico
                new[f"fx_fase_pico_{nome}"] = np.mod(dsp + 1, 15)
                if nome == "lo":
                    new["fx_npicos_30"] = flag.astype(float).groupby(gk, observed=True).transform(
                        lambda s: s.rolling(30, min_periods=1).sum())
    if SENT_EXTRA and {"sent_mf1", "sent_mf25", "sent_mf49"} <= set(core.columns):
        m1, m25, m49 = core["sent_mf1"], core["sent_mf25"], core["sent_mf49"]
        m49l = g["sent_mf49"].shift(1).where(ok)                  # mf49 de ontem = mesma falha prevista 24h antes
        new["sx_mf_max"] = pd.concat([m1, m25, m49], axis=1).max(axis=1, skipna=False)
        new["sx_mf49_lag1"] = m49l
        new["sx_ev_media"] = (m25 + m49l) / 2
        new["sx_ev_min"] = pd.concat([m25, m49l], axis=1).min(axis=1, skipna=False)
        new["sx_mf25_d1"] = m25 - g["sent_mf25"].shift(1).where(ok)
        new["sx_mf25_vs_hist"] = m25 - g["sent_mf25"].transform(lambda s: s.shift(1).rolling(30, min_periods=5).mean())
        gd = core.groupby(["dia", "peca"], observed=True)["sent_mf25"]
        new["sx_mf25_rk_dia"] = gd.rank(pct=True)
        new["sx_mf49_rk_dia"] = core.groupby(["dia", "peca"], observed=True)["sent_mf49"].rank(pct=True)
        new["sx_mf25_z_dia"] = (m25 - gd.transform("mean")) / gd.transform("std").replace(0, np.nan)
    feat = pd.DataFrame(new).astype("float32")
    for k in KEY:
        feat[k] = core[k]
    NOVAS = list(new)
    out = []
    for p in parts:
        q = p.drop(columns=[c for c in NOVAS if c in p.columns]).merge(feat, on=KEY, how="left")
        assert len(q) == len(p)
        out.append(q)
    return out
# <<< EXTRA ---------------------------------------------------------------------------------------------

# ---------------------------------------------------------------- dados
X = pd.read_parquet(OUT + "features_maquina_dia.parquet")
falhas = pd.read_parquet(U + "falhas.parquet")
Lall = montar_longo(X, falhas)
Lall = add_historico(Lall, falhas)
if USAR_GRADE:
    Lall = add_grade(Lall, falhas); marca("grade calculada")
if USAR_TROCA:
    manut = pd.read_parquet(U + "manutencao.parquet")
    Lall = add_troca(Lall, manut, falhas); marca("features de troca calculadas")
L = Lall[Lall["dia"] <= ULTIMO_DIA_ROTULADO].reset_index(drop=True)
tr_d, va_d, te_d = split_temporal(L["dia"].unique())

nb = None
if USAR_NB:
    nb = NB26(U, corte_z=va_d[0], usar_risco=USAR_RISCO)
    keys = Lall[["id_maquina", "dia"]].drop_duplicates().reset_index(drop=True)
    W = nb.estatico_wide(keys); marca(f"nb26: {W.shape[1]} colunas estáticas")
    Ltr = L[L["dia"].isin(tr_d)]
    nb.aprender(keys, W, Ltr[mascara(Ltr)][["id_maquina", "dia", "peca", "y"]])
    S = nb.estatico_long(keys, W) if USAR_NB_ESTATICO else None
    if S is not None:
        Lall = Lall.merge(S, on=CHAVE, how="left"); del S
    L = Lall[Lall["dia"] <= ULTIMO_DIA_ROTULADO].reset_index(drop=True)
    marca(f"base unificada: {L.shape[1]} colunas")

tr, va, te = (L[L["dia"].isin(d)].reset_index(drop=True) for d in (tr_d, va_d, te_d))
sent, alerta = derivar_sentinelas(tr)
tr, va, te = (features(d, sent, alerta) for d in (tr, va, te))
if USAR_NB and USAR_NB_DEP:
    tr = add_dep(tr, None, None)
    va = add_dep(va, va_d[0], nb.pseudo_por_erros(va_d[0], va_d[-1]))
    te = add_dep(te, te_d[0], nb.pseudo_por_erros(te_d[0], te_d[-1]))
if USAR_EXTRA:
    tr, va, te = add_extra([tr, va, te], limiares_pico(tr)); marca(f"features novas (itens 2/5): {len(NOVAS)}")
cols_u = colunas_modelo(tr, True)
if USAR_EXTRA:
    cols_u = cols_u + [c for c in NOVAS if c in tr.columns and c not in cols_u]
if SEM_IDENTIDADE:                                                                   # item 4
    cols_u = [c for c in cols_u if c not in IDENT]
if PODAR:                                                                            # item 3
    ip = pd.read_csv(OUT + "importancia_features_v7.csv", index_col=0)["importancia"].sort_values(ascending=False)
    ok_poda = set(ip.index[(ip.cumsum() - ip) < PODA_ACUM])
    cols_u = [c for c in cols_u if (c in ok_poda) or (c not in ip.index)]            # colunas novas (fora do CSV) sempre ficam
sel = None
if SELECAO:
    sel = SelecaoFeatures(str(Path(__file__).with_name("importancia_features_v8a.csv")), imp_min=IMP_MIN, n_dup=N_DUP, copias=COPIAS)
    n0 = len(cols_u)
    cols_u = sel.colunas(cols_u)
    tr, va, te = (sel.adicionar_dups(d) for d in (tr, va, te))
    marca(f"seleção: {n0} -> {len(cols_u)} colunas ({len([c for c in cols_u if SUF in c])} cópias)")
NB_PREF = ("nb_", "nbp_", "nbd_")
cols_base = [c for c in cols_u if not c.startswith(NB_PREF)]
marca(f"dados prontos | treino {len(tr)} linhas | {len(cols_u)} colunas ({len(cols_u) - len(cols_base)} do nb26)")
for nome, d in (("treino", tr), ("validação", va), ("teste", te)):
    m, pos = mascara(d), (d["y"] == 1).to_numpy()
    print(f"máscara {nome}: linhas {m.mean():.3f} | positivos mantidos {m[pos].mean():.4f} ({int((m & pos).sum())}/{int(pos.sum())})", flush=True)

# ---------------------------------------------------------------- treino
BASE = dict(tree_method="hist", enable_categorical=True, eval_metric="aucpr", n_jobs=-1)
NV = len(PARAMS_VAR)
def kw(seed, **extra):
    k = dict(BASE, random_state=7 + seed, **PARAMS_VAR[seed % NV])
    k.update(dict(n_estimators=3000, early_stopping_rounds=100) if EARLY_STOP else dict(n_estimators=600))
    k.update(extra)
    return k

def fit_bag(A, B, cols, n_seeds):
    A, B = so_grade(A), so_grade(B)
    return [xgb.XGBClassifier(**kw(s)).fit(A[cols], A["y"], eval_set=[(B[cols], B["y"])], verbose=False) for s in range(n_seeds)]

def prever(ms, d, cols):
    return aplicar(d, np.mean([m.predict_proba(d[cols])[:, 1] for m in ms], axis=0))

ap = lambda d, p: ap_por_peca(d, p)[1]
ms = fit_bag(tr, va, cols_u, N_SEEDS)
best = [int(m.best_iteration) for m in ms]
marca(f"modelo treinado | árvores por semente: {best}")
p_va, p_te = prever(ms, va, cols_u), prever(ms, te, cols_u)
print(f"{VERSAO}: AP validação {ap(va, p_va):.4f} | AP teste {ap(te, p_te):.4f}", flush=True)
print("AP por peça no teste:", {k: round(v, 4) for k, v in ap_por_peca(te, p_te)[0].items()}, flush=True)
limiar, f1_va = melhor_limiar(va["y"], p_va)
R = {"versao": VERSAO, "ap_validacao": ap(va, p_va), "ap_teste": ap(te, p_te), "ap_por_peca_teste": ap_por_peca(te, p_te)[0],
     "limiar": limiar, "teste_topk_diario": topk_diario(te, p_te), "modo_rotulo": MODO_ROTULO,
     "config": dict(seeds=N_SEEDS, params_var=PARAMS_VAR, fase15=FASE15, sent_extra=SENT_EXTRA, estabilidade=ESTABILIDADE,
                    podar=PODAR, poda_acum=PODA_ACUM, sem_identidade=SEM_IDENTIDADE, selecao=SELECAO, imp_min=IMP_MIN, n_dup=N_DUP, copias=COPIAS, n_colunas=len(cols_u), arvores=best),
     "colunas_novas": NOVAS}
imp = pd.Series(np.mean([m.feature_importances_ for m in ms], axis=0), index=cols_u)
imp.index = [c.split(SUF)[0] for c in imp.index]          # soma as cópias de volta na coluna original
imp = imp.groupby(level=0).sum().sort_values(ascending=False)
imp.to_csv(OUT + f"importancia_features_{VERSAO}.csv", header=["importancia"])
R["top20_importancia"] = imp.head(20).round(4).to_dict()
R["importancia_novas"] = imp.reindex([c for c in NOVAS if c in imp.index]).round(4).to_dict()
json.dump(R, open(OUT + f"resultados_{VERSAO}.json", "w"), indent=2, default=float)
print(json.dumps({k: R[k] for k in ["top20_importancia", "importancia_novas"]}, indent=1, default=float), flush=True)

# ---------------------------------------------------------------- refit com todos os rótulos + previsão 4T/2023
sent_f, alerta_f = derivar_sentinelas(L)
Lf = add_dep(features(L, sent_f, alerta_f), None, None)
Q = Lall[(Lall["dia"] >= "2023-09-30") & (Lall["dia"] <= "2023-12-30")].reset_index(drop=True)
Q = features(Q, sent_f, alerta_f)
if USAR_NB and USAR_NB_DEP:
    Q = add_dep(Q, None, nb.pseudo_por_erros("2023-09-30", "2023-12-30"))      # falhas só até 30/09 + pseudo-falhas dos erros
if USAR_EXTRA:
    Lf, Q = add_extra([Lf, Q], limiares_pico(Lf))                               # Q herda o passado de Lf (sensores)
if sel is not None:
    Lf, Q = sel.adicionar_dups(Lf), sel.adicionar_dups(Q)
Lfit = so_grade(Lf)
n_fin = {v: int(np.median([b for s, b in enumerate(best) if s % NV == v]) * 1.2) + 1 for v in range(NV)} if ESTABILIDADE else None
mods = [xgb.XGBClassifier(**{**kw(s), "n_estimators": (n_fin[s % NV] if ESTABILIDADE else int(m.best_iteration * 1.2) + 1),
                              "early_stopping_rounds": None}).fit(Lfit[cols_u], Lfit["y"]) for s, m in enumerate(ms)]
joblib.dump(dict(modelos=mods, cols=cols_u, sentinelas=sent_f, alerta=alerta_f, limiar=limiar), OUT + f"modelo_final_{VERSAO}.joblib")
marca("refit final concluído")

Q["prob_falha"] = prever(mods, Q, cols_u)
Q["janela_ate"] = Q["dia"] + pd.Timedelta(days=1, hours=6)
Q["rank_no_dia"] = Q.groupby("dia")["prob_falha"].rank(ascending=False, method="first").astype(int)
Q["alerta"] = (Q["prob_falha"] >= limiar).astype(int)
out = Q[["dia", "janela_ate", "id_maquina", "peca", "prob_falha", "rank_no_dia", "alerta"]].rename(columns={"dia": "data_previsao"})
out = out.sort_values(["data_previsao", "rank_no_dia"]).reset_index(drop=True)
out.to_csv(OUT + f"previsoes_out_dez_2023_{VERSAO}.csv", index=False)
sub = montar_submissao(out, U + "sample_submission.csv")
sub.to_csv(OUT + f"submission_{VERSAO}.csv", index=False)
print(f"submission_{VERSAO}.csv:", sub.shape, "<- ENVIE ESTE ao Kaggle")
print("alertas/dia (média):", round(out.groupby('data_previsao').alerta.sum().mean(), 2), "| taxa média prevista Q4:", round(out.prob_falha.mean(), 4),
      "| taxa real rotulada:", round(L.y.mean(), 4), "| linhas candidatas Q4:", round(float(mascara(Q).mean()), 4))

