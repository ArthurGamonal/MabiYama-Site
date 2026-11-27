"""Features de COMPETIÇÃO entre as 4 peças da mesma máquina no mesmo dia + filtro de grade.
Uso (treinar.py), logo depois de add_peca_features em tr/va/te, Lf e Q:
    from mb_extra import add_competicao, mascara_grade, aplicar_grade
"""
import numpy as np, pandas as pd

COMP_COLS = ["sent_z6", "sent_z24", "sent_tend", "sent_sobe_24h", "sent_mf25_vs",
             "err_peca_24h", "err_peca_72h", "alerta_peca_n", "horas_desde_alerta_peca"]


def add_competicao(L: pd.DataFrame, cols=COMP_COLS) -> pd.DataFrame:
    """Na grade, a máquina 'tem um evento' e a dúvida é QUAL peça. Compara cada peça com as outras 3 da mesma máquina-dia."""
    L = L.copy()
    g = L.groupby(["id_maquina", "dia"], observed=True)
    for c in cols:
        if c not in L.columns:
            continue
        v = L[c].astype(float)
        L[f"{c}_rk"] = g[c].rank(ascending=False, method="min")
        L[f"{c}_dmax"] = v - g[c].transform("max")          # 0 = é a peça mais "alterada" da máquina
        L[f"{c}_dmed"] = v - g[c].transform("mean")
    if "grade_on_m" in L.columns:
        L["maq_n_alerta_peca"] = g["alerta_peca"].transform("sum") if "alerta_peca" in L.columns else 0
    return L


def mascara_grade(L: pd.DataFrame) -> np.ndarray:
    """True = linha candidata (grade ligada OU fase desconhecida). Fora da grade a falha é ~impossível (0 de 198 positivos)."""
    return (L["grade_on_m"] != 0).to_numpy()          # NaN (sem fase) != 0 -> True: mantém

def aplicar_grade(L: pd.DataFrame, p, piso=1e-3) -> np.ndarray:
    """Mantém as linhas fora da grade SEMPRE abaixo das candidatas (sem zerar o ranking)."""
    p = np.asarray(p, dtype=float)
    return np.where(mascara_grade(L), p, p * piso)
