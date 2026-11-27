"""Features de TROCA (manutencao.parquet) para encaixar no mb.py.

Uso no treinar.py, logo depois de add_grade:
    from mb_troca import add_troca, diagnostico_troca
    manut = pd.read_parquet(U + "manutencao.parquet")
    Lall = add_troca(Lall, manut, falhas)
    diagnostico_troca(Lall[Lall["dia"] <= ULTIMO_DIA_ROTULADO])   # rode 1x e olhe os números

Por que as trocas e não só as falhas:
  * as vidas úteis são 100% múltiplas de 15 dias => a fase da grade de cada máquina sai de ~50 trocas
    (desde mar/2022), e não de ~7 falhas;
  * em out-dez/2023 a tabela de manutenção só tem trocas PREVENTIVAS (as trocas por falha sumiram:
    ~220/mês -> ~165/mês, igual ao nº de trocas sem falha). Por isso a idade da peça é calculada
    só com trocas preventivas, no treino e na previsão (mesma definição = sem deslocamento de distribuição).
  * só entra troca com data_hora <= corte das 05:00 do dia D (regra da competição).
"""
import numpy as np, pandas as pd
from mb import PECAS, HORA_CORTE, add_grade


def trocas_longas(manut: pd.DataFrame) -> pd.DataFrame:
    m = manut.copy()
    m["data_hora"] = pd.to_datetime(m["data_hora"]).astype("datetime64[ns]")
    m["id_maquina"] = m["id_maquina"].astype(str)          # manutencao.parquet pode vir como category
    partes = []
    for p in PECAS:
        x = m.loc[m[p] == 1, ["id_maquina", "data_hora"]].copy()
        x["peca"] = p
        partes.append(x)
    return pd.concat(partes, ignore_index=True).sort_values("data_hora").reset_index(drop=True)


def marcar_preventivas(T: pd.DataFrame, falhas: pd.DataFrame, tol_h: int = 72) -> pd.DataFrame:
    f = falhas[["id_maquina", "peca", "data_hora"]].copy()
    f["data_hora"] = pd.to_datetime(f["data_hora"]).astype("datetime64[ns]")
    f["id_maquina"] = f["id_maquina"].astype(str)
    f["peca"] = f["peca"].astype(str)
    f = f.rename(columns={"data_hora": "t_falha"}).sort_values("t_falha")
    T = pd.merge_asof(T.sort_values("data_hora"), f, left_on="data_hora", right_on="t_falha",
                      by=["id_maquina", "peca"], direction="nearest", tolerance=pd.Timedelta(hours=tol_h))
    T["preventiva"] = T["t_falha"].isna()          # em out-dez não há falhas na tabela => tudo preventiva (é o que existe lá)
    return T


def _ultima(L, P, chave, nome):
    """dias (até D+1 06:00) desde a última troca preventiva visível no corte das 05:00 de D."""
    A = pd.DataFrame({"corte": (L["dia"] + pd.Timedelta(hours=HORA_CORTE)).to_numpy(),
                      "id_maquina": L["id_maquina"].astype(str).to_numpy(),
                      "peca": L["peca"].astype(str).to_numpy(), "_i": np.arange(len(L))}).sort_values("corte")
    Pp = P.drop_duplicates(chave + ["t_prev"]).sort_values("t_prev")
    Pp = Pp.assign(t_prev=Pp["t_prev"].astype("datetime64[ns]"))
    A["corte"] = A["corte"].astype("datetime64[ns]")
    A = pd.merge_asof(A, Pp[chave + ["t_prev"]], left_on="corte", right_on="t_prev", by=chave, direction="backward")
    A = A.sort_values("_i")
    alvo = (L["dia"] + pd.Timedelta(days=1, hours=6)).to_numpy().astype("datetime64[ns]")
    L[nome] = (alvo - A["t_prev"].to_numpy().astype("datetime64[ns]")) / np.timedelta64(1, "D")


def add_troca(L: pd.DataFrame, manut: pd.DataFrame, falhas: pd.DataFrame) -> pd.DataFrame:
    L = L.copy()
    T = marcar_preventivas(trocas_longas(manut), falhas)
    # 1) fase da grade de 15 dias a partir de TODAS as trocas (a fase não depende das trocas ocultas)
    G = add_grade(L[["dia", "id_maquina", "peca"]].copy(), T[["id_maquina", "peca", "data_hora"]], conhecido_ate="2100-01-01")
    for c in G.columns:
        if c.startswith("grade_"):
            L["gt_" + c[len("grade_"):]] = G[c].to_numpy()
    # 2) idade da peça (e da máquina) desde a última troca PREVENTIVA
    P = T.loc[T["preventiva"], ["id_maquina", "peca", "data_hora"]].rename(columns={"data_hora": "t_prev"})
    _ultima(L, P, ["id_maquina", "peca"], "gt_idade_dias")
    Pm = P.assign(peca="x")
    L2 = L.assign(peca="x")
    _ultima(L2, Pm, ["id_maquina", "peca"], "gt_idade_maq_dias")
    L["gt_idade_maq_dias"] = L2["gt_idade_maq_dias"].to_numpy()
    L["gt_ciclos"] = np.round(L["gt_idade_dias"] / 15)
    L["gt_idade_ok15"] = ((L["gt_idade_dias"] % 15) == 0).astype(float).where(L["gt_idade_dias"].notna())
    return L


def diagnostico_troca(L: pd.DataFrame):
    """Rode no conjunto ROTULADO. Responde: (1) a grade das trocas é uma regra dura? (2) qual o risco por idade?"""
    pos = L["y"] == 1
    print("positivos com gt_on_m == 1 :", round(float((L.loc[pos, "gt_on_m"] == 1).mean()), 4), "(~1.0 => grade é regra dura)")
    print("linhas com gt_on_m == 1    :", round(float((L["gt_on_m"] == 1).mean()), 4), "(~0.067 esperado)")
    on = L[L["gt_on_m"] == 1]
    print("taxa de falha | grade ligada:", round(float(on["y"].mean()), 4), "| desligada:", round(float(L.loc[L["gt_on_m"] != 1, "y"].mean()), 5))
    print("\ntaxa de falha por idade (ciclos de 15d) só nas linhas na grade:")
    print(on.groupby(on["gt_ciclos"].clip(upper=10))["y"].agg(["size", "sum", "mean"]).round(4).to_string())
