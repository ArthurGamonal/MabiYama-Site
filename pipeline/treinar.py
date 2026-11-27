"""Hiperparâmetros fixos, avaliação única no teste, refit com todos os rótulos e previsão de out-dez/2023."""
import json, warnings, joblib
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.metrics import average_precision_score
from mb import *
from mb_extra import aplicar_grade, mascara_grade

warnings.filterwarnings("ignore")

PARAMS_UNIFICADO = dict(
    learning_rate=0.03, max_depth=4, min_child_weight=10, subsample=0.8,
    colsample_bytree=0.6, reg_lambda=5, reg_alpha=0.1, gamma=0.5,
)
PARAMS_POR_PECA = {**PARAMS_UNIFICADO, "max_depth": 3, "min_child_weight": 5}
TIPO_MODELO = "unificado"  # unificado, por_peca, ensemble ou auto
EARLY_STOP = True
N_ARVORES = 600  # usado quando EARLY_STOP é False
HIPERPARAMETROS = {"unificado": PARAMS_UNIFICADO, "por_peca": PARAMS_POR_PECA}

X = pd.read_parquet(OUT + "features_maquina_dia.parquet")
falhas = pd.read_parquet(U + "falhas.parquet")
Lall = montar_longo(X, falhas)
del X
Lall = add_historico(Lall, falhas)
if MODO_ROTULO == "vespera":
    Lall = add_grade(Lall, falhas)
L = Lall[Lall["dia"] <= ULTIMO_DIA_ROTULADO].reset_index(drop=True)
Q = Lall[(Lall["dia"] >= "2023-09-30") & (Lall["dia"] <= "2023-12-30")].reset_index(drop=True)   # 92 dias do sample_submission
del Lall

tr_d, va_d, te_d = split_temporal(L["dia"].unique())
tr, va, te = (L[L["dia"].isin(d)].reset_index(drop=True) for d in (tr_d, va_d, te_d))
sent, alerta = derivar_sentinelas(tr)
tr, va, te = (add_peca_features(d, sent, alerta) for d in (tr, va, te))
cols_u, cols_p = colunas_modelo(tr, True), colunas_modelo(tr, False)
BASE = dict(tree_method="hist", enable_categorical=True, eval_metric="aucpr", n_jobs=1, random_state=7)
USAR_GRADE = MODO_ROTULO == "vespera"


def candidatos(dados):
    if USAR_GRADE:
        candidatos_grade = dados.loc[mascara_grade(dados)]
        if candidatos_grade["y"].nunique() == 2:
            return candidatos_grade
    return dados


def aplicar_fase(dados, prob):
    return aplicar_grade(dados, prob) if USAR_GRADE else prob


def modelo_kw(params):
    arvores = dict(n_estimators=3000, early_stopping_rounds=100) if EARLY_STOP else dict(n_estimators=N_ARVORES)
    return {**BASE, **arvores, **params}


def fit_unificado(params, A, B):
    treino, validacao = candidatos(A), candidatos(B)
    m = xgb.XGBClassifier(**modelo_kw(params)).fit(
        treino[cols_u], treino["y"], eval_set=[(validacao[cols_u], validacao["y"])], verbose=False,
    )
    return m, aplicar_fase(B, m.predict_proba(B[cols_u])[:, 1])

def fit_por_peca(params, A, B):
    ms, p = {}, np.zeros(len(B))
    for pc in PECAS:
        a, k = candidatos(A[A.peca == pc]), (B.peca == pc).to_numpy()
        b = candidatos(B.loc[k])
        m = xgb.XGBClassifier(**modelo_kw(params)).fit(a[cols_p], a["y"], eval_set=[(b[cols_p], b["y"])], verbose=False)
        ms[pc] = m
        p[k] = m.predict_proba(B.loc[k, cols_p])[:, 1]
    return ms, aplicar_fase(B, p)

print("Hiperparâmetros fixos:", HIPERPARAMETROS[TIPO_MODELO] if TIPO_MODELO in HIPERPARAMETROS else HIPERPARAMETROS, flush=True)
tipo = TIPO_MODELO
cand = {}
mu, mp = None, {}
if tipo in ("unificado", "ensemble", "auto"):
    mu, cand["unificado"] = fit_unificado(PARAMS_UNIFICADO, tr, va)
if tipo in ("por_peca", "ensemble", "auto"):
    mp, cand["por_peca"] = fit_por_peca(PARAMS_POR_PECA, tr, va)
if tipo in ("ensemble", "auto"):
    cand["ensemble"] = (cand["unificado"] + cand["por_peca"]) / 2
ap_va = {k: ap_por_peca(va, v)[1] for k, v in cand.items()}
print("AP validação:", {k: round(v, 4) for k, v in ap_va.items()})
if tipo == "auto":
    tipo = max(ap_va, key=ap_va.get)
    if tipo == "ensemble" and ap_va["ensemble"] - max(ap_va["unificado"], ap_va["por_peca"]) < 0.01:
        tipo = max(("unificado", "por_peca"), key=ap_va.get)
print("modelo usado:", tipo)
art = dict(tipo=tipo, modelo_unificado=mu, cols_unificado=cols_u, modelos_peca=mp, cols_peca=cols_p,
           usar_grade=USAR_GRADE)
p_va = prever_prob(art, va)
limiar, f1_va = melhor_limiar(va["y"], p_va)
print(f"limiar (máx F1 na validação) = {limiar:.3f}  F1 val = {f1_va:.3f}")

# ------------------------------------------------------------ TESTE (uma única vez)
p_te = prever_prob(art, te)
R = {"tipo": tipo, "ap_validacao": ap_va, "limiar": limiar, "hiperparametros": HIPERPARAMETROS,
     "split": {"treino": [str(tr_d[0].date()), str(tr_d[-1].date()), len(tr_d)],
               "validacao": [str(va_d[0].date()), str(va_d[-1].date()), len(va_d)],
               "teste": [str(te_d[0].date()), str(te_d[-1].date()), len(te_d)]},
     "sentinelas": {k: list(v) for k, v in sent.items()}, "alerta_codigos": alerta}
R["teste_geral"] = metricas(te["y"], p_te) | pr_rec_f1(te["y"], p_te, limiar)
R["teste_por_peca"] = {}
for pc in PECAS:
    k = (te.peca == pc).to_numpy()
    R["teste_por_peca"][pc] = metricas(te.loc[k, "y"], p_te[k]) | pr_rec_f1(te.loc[k, "y"], p_te[k], limiar)
R["teste_topk_diario"] = topk_diario(te, p_te)
# recall por tipo de falha: iminente (<=1h do corte) x antecipada (>1h)
pos = te["y"] == 1
imin = pos & (te["antecedencia_h"] <= 1.0001); ante = pos & (te["antecedencia_h"] > 1.0001)
alvo = p_te >= limiar
R["recall_iminente"] = dict(n=int(imin.sum()), recall=float(alvo[imin.to_numpy()].mean()) if imin.sum() else None)
R["recall_antecipada"] = dict(n=int(ante.sum()), recall=float(alvo[ante.to_numpy()].mean()) if ante.sum() else None)
R["modo_rotulo"] = MODO_ROTULO
_d, _m = ap_por_peca(te, p_te)
R["ap_por_peca_teste"], R["ap_media_pecas_teste"] = _d, _m          # <- métrica do Kaggle
_d, _m = ap_por_peca(va, p_va)
R["ap_por_peca_validacao"], R["ap_media_pecas_validacao"] = _d, _m
# AP separado: avalia o modelo só contra falhas antecipadas (remove as iminentes dos positivos)
m2 = ~imin.to_numpy()
R["ap_teste_so_antecipadas"] = float(average_precision_score(te.loc[m2, "y"], p_te[m2]))
# referência: regra simples "alerta das 06h de ontem com o código da peça"
R["ap_baseline_sentinela_z24"] = ap_por_peca(te, te["sent_z24"].fillna(0).to_numpy())[0]   # referência: só o sensor-sentinela
# calibração
bins = [0, 0.005, 0.01, 0.03, 0.1, 0.3, 0.6, 1.0001]
cb = pd.DataFrame({"p": p_te, "y": te["y"].to_numpy()}); cb["faixa"] = pd.cut(cb["p"], bins, right=False)
R["calibracao_teste"] = cb.groupby("faixa", observed=True).agg(n=("y", "size"), prob_media=("p", "mean"), taxa_real=("y", "mean")).round(4).reset_index().astype({"faixa": str}).to_dict("records")
json.dump(R, open(OUT + "resultados.json", "w"), indent=2, default=float)
te.assign(prob=p_te)[["dia", "id_maquina", "peca", "y", "antecedencia_h", "prob"]].to_parquet(OUT + "predicoes_teste.parquet", index=False)
va.assign(prob=p_va)[["dia", "id_maquina", "peca", "y", "antecedencia_h", "prob"]].to_parquet(OUT + "predicoes_validacao.parquet", index=False)
print(json.dumps({k: R[k] for k in ["modo_rotulo", "ap_media_pecas_validacao", "ap_media_pecas_teste", "ap_por_peca_teste", "ap_baseline_sentinela_z24", "teste_topk_diario"]}, indent=1, default=float), flush=True)

# ------------------------------------------------------------ REFIT com todos os rótulos + previsão 4T/2023
sent_f, alerta_f = derivar_sentinelas(L)


def iters(m):
    return int(m.best_iteration * 1.2) + 1 if EARLY_STOP else N_ARVORES


arvores_unificado = iters(mu) if mu is not None else None
arvores_peca = {pc: iters(m) for pc, m in mp.items()}
del tr, va, te, p_va, p_te, cand, cb, art, mp, mu
taxa_rotulada = L["y"].mean()
Lf = add_peca_features(L, sent_f, alerta_f)
del L


art_f = dict(tipo=tipo, cols_unificado=cols_u, cols_peca=cols_p, sentinelas=sent_f, alerta_codigos=alerta_f,
             limiar=limiar, usar_grade=USAR_GRADE)
if tipo in ("unificado", "ensemble"):
    treino = candidatos(Lf)
    art_f["modelo_unificado"] = xgb.XGBClassifier(
        **BASE, **PARAMS_UNIFICADO, n_estimators=arvores_unificado,
    ).fit(treino[cols_u], treino["y"])
    del treino
if tipo in ("por_peca", "ensemble"):
    art_f["modelos_peca"] = {}
    for pc in PECAS:
        treino = candidatos(Lf[Lf.peca == pc])
        art_f["modelos_peca"][pc] = xgb.XGBClassifier(
            **BASE, **PARAMS_POR_PECA, n_estimators=arvores_peca[pc],
        ).fit(treino[cols_p], treino["y"])
        del treino
joblib.dump(art_f, OUT + "modelo_final.joblib")
share_alerta_treino = (Lf.alerta_peca > 0).mean()
del Lf

Q = add_peca_features(Q, sent_f, alerta_f)
Q["prob_falha"] = prever_prob(art_f, Q)
Q["janela_ate"] = Q["dia"] + pd.Timedelta(days=1, hours=6)
Q["rank_no_dia"] = Q.groupby("dia")["prob_falha"].rank(ascending=False, method="first").astype(int)
Q["alerta"] = (Q["prob_falha"] >= limiar).astype(int)
out = Q[["dia", "janela_ate", "id_maquina", "peca", "prob_falha", "rank_no_dia", "alerta"]].rename(columns={"dia": "data_previsao"})
out = out.sort_values(["data_previsao", "rank_no_dia"]).reset_index(drop=True)
out.to_csv(OUT + "previsoes_out_dez_2023.csv", index=False)
sub = montar_submissao(out, U + "sample_submission.csv")      # <- arquivo para enviar ao Kaggle
sub.to_csv(OUT + "submission.csv", index=False)
print("submission.csv:", sub.shape)
mq = out.groupby(["data_previsao", "id_maquina"])["prob_falha"].apply(lambda s: 1 - np.prod(1 - s)).rename("prob_alguma_peca").reset_index()
mq.to_csv(OUT + "risco_por_maquina_out_dez_2023.csv", index=False)
print("previsões:", out.shape, "| dias", out.data_previsao.nunique(), "| alertas/dia (média):", round(out.groupby('data_previsao').alerta.sum().mean(), 2))
print("taxa média prevista Q4:", round(out.prob_falha.mean(), 4), "| taxa real média rotulada:", round(taxa_rotulada, 4))
print("share linhas com alerta_peca>0  rotulado:", round(share_alerta_treino, 4), "| Q4:", round((Q.alerta_peca > 0).mean(), 4))
