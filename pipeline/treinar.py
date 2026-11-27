"""Tuning (Optuna), avaliação única no teste, refit com todos os rótulos e previsão de out-dez/2023."""
import json, warnings, joblib
import numpy as np, pandas as pd, xgboost as xgb, optuna
from sklearn.metrics import average_precision_score, precision_recall_curve
from mb import *

warnings.filterwarnings("ignore"); optuna.logging.set_verbosity(optuna.logging.WARNING)
import os
N_TRIALS = int(os.environ.get("MABIYAMA_TRIALS", 25))   # tentativas do Optuna por abordagem
X = pd.read_parquet(OUT + "features_maquina_dia.parquet")
falhas = pd.read_parquet(U + "falhas.parquet")
Lall = montar_longo(X, falhas)
Lall = add_historico(Lall, falhas)
L = Lall[Lall["dia"] <= ULTIMO_DIA_ROTULADO].reset_index(drop=True)
tr_d, va_d, te_d = split_temporal(L["dia"].unique())
tr, va, te = (L[L["dia"].isin(d)].reset_index(drop=True) for d in (tr_d, va_d, te_d))
sent, alerta = derivar_sentinelas(tr)
tr, va, te = (add_peca_features(d, sent, alerta) for d in (tr, va, te))
cols_u, cols_p = colunas_modelo(tr, True), colunas_modelo(tr, False)
BASE = dict(n_estimators=3000, tree_method="hist", enable_categorical=True, eval_metric="aucpr",
            early_stopping_rounds=100, n_jobs=-1, random_state=7)

def sugerir(t):
    return dict(learning_rate=t.suggest_float("learning_rate", 0.01, 0.1, log=True),
                max_depth=t.suggest_int("max_depth", 2, 7),
                min_child_weight=t.suggest_int("min_child_weight", 1, 20),
                subsample=t.suggest_float("subsample", 0.5, 1.0),
                colsample_bytree=t.suggest_float("colsample_bytree", 0.3, 1.0),
                reg_lambda=t.suggest_float("reg_lambda", 1, 30, log=True),
                reg_alpha=t.suggest_float("reg_alpha", 1e-3, 5, log=True),
                gamma=t.suggest_float("gamma", 0, 3))

def fit_unificado(params, A, B):
    m = xgb.XGBClassifier(**BASE, **params).fit(A[cols_u], A["y"], eval_set=[(B[cols_u], B["y"])], verbose=False)
    return m, m.predict_proba(B[cols_u])[:, 1]

def fit_por_peca(params, A, B):
    ms, p = {}, np.zeros(len(B))
    for pc in PECAS:
        a, k = A[A.peca == pc], (B.peca == pc).to_numpy()
        m = xgb.XGBClassifier(**BASE, **params).fit(a[cols_p], a["y"], eval_set=[(B.loc[k, cols_p], B.loc[k, "y"])], verbose=False)
        ms[pc] = m; p[k] = m.predict_proba(B.loc[k, cols_p])[:, 1]
    return ms, p

estudos, melhores = {}, {}
for nome, fn in [("unificado", fit_unificado), ("por_peca", fit_por_peca)]:
    est = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=11))
    est.optimize(lambda t: ap_por_peca(va, fn(sugerir(t), tr, va)[1])[1], n_trials=N_TRIALS)
    estudos[nome] = est; melhores[nome] = est.best_params
    print(f"[{nome}] melhor AP val = {est.best_value:.4f} | {est.best_params}", flush=True)

# modelos finais (treino) com os melhores hiperparâmetros
mu, pu_va = fit_unificado(melhores["unificado"], tr, va)
mp, pp_va = fit_por_peca(melhores["por_peca"], tr, va)
pe_va = (pu_va + pp_va) / 2
cand = {"unificado": pu_va, "por_peca": pp_va, "ensemble": pe_va}
ap_va = {k: ap_por_peca(va, v)[1] for k, v in cand.items()}
print("AP validação:", {k: round(v, 4) for k, v in ap_va.items()})
tipo = max(ap_va, key=ap_va.get)
if tipo == "ensemble" and ap_va["ensemble"] - max(ap_va["unificado"], ap_va["por_peca"]) < 0.01:
    tipo = max(("unificado", "por_peca"), key=ap_va.get)       # só combina se o ganho compensar
print("modelo escolhido:", tipo)
art = dict(tipo=tipo, modelo_unificado=mu, cols_unificado=cols_u, modelos_peca=mp, cols_peca=cols_p)
p_va = prever_prob(art, va)
limiar, f1_va = melhor_limiar(va["y"], p_va)
print(f"limiar (máx F1 na validação) = {limiar:.3f}  F1 val = {f1_va:.3f}")

# ------------------------------------------------------------ TESTE (uma única vez)
p_te = prever_prob(art, te)
R = {"tipo": tipo, "ap_validacao": ap_va, "limiar": limiar, "hiperparametros": melhores,
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
Lf = add_peca_features(L, sent_f, alerta_f)
def iters(m): return int(m.best_iteration * 1.2) + 1       # um pouco mais de árvores: mais dados
P = {k: {x: v for x, v in melhores[k].items()} for k in melhores}
art_f = dict(tipo=tipo, cols_unificado=cols_u, cols_peca=cols_p, sentinelas=sent_f, alerta_codigos=alerta_f, limiar=limiar)
FIT = {k: v for k, v in BASE.items() if k not in ("early_stopping_rounds",)}
if tipo in ("unificado", "ensemble"):
    art_f["modelo_unificado"] = xgb.XGBClassifier(**{**FIT, "n_estimators": iters(mu)}, **P["unificado"]).fit(Lf[cols_u], Lf["y"])
if tipo in ("por_peca", "ensemble"):
    art_f["modelos_peca"] = {pc: xgb.XGBClassifier(**{**FIT, "n_estimators": iters(mp[pc])}, **P["por_peca"]).fit(Lf.loc[Lf.peca == pc, cols_p], Lf.loc[Lf.peca == pc, "y"]) for pc in PECAS}
joblib.dump(art_f, OUT + "modelo_final.joblib")

Q = Lall[(Lall["dia"] >= "2023-09-30") & (Lall["dia"] <= "2023-12-30")].reset_index(drop=True)   # 92 dias do sample_submission
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
print("taxa média prevista Q4:", round(out.prob_falha.mean(), 4), "| taxa real média rotulada:", round(L.y.mean(), 4))
print("share linhas com alerta_peca>0  rotulado:", round((Lf.alerta_peca > 0).mean(), 4), "| Q4:", round((Q.alerta_peca > 0).mean(), 4))
