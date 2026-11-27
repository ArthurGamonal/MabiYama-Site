"""
Poda de features de baixa importância + duplicação das mais importantes.

Aplicar DEPOIS de todas as colunas derivadas (nb_*, nbp_*, gt_*, grade_*, sent_*...)
e ANTES de criar a DMatrix/dataset, no treino e na previsão (mesma função, mesmo mapa).

    from selecao import SelecaoFeatures
    sel = SelecaoFeatures("importancia_features_v8a.csv", imp_min=5e-4, n_dup=20, copias=1)
    X_tr2 = sel.transform(X_tr)       # usar em treino, validação, teste e submission
    ...
    imp_real = sel.importancia_agregada(modelo_importancias, X_tr2.columns)  # soma cópias de volta

Por que duplicar ajuda: com colsample_bytree = 0.4–0.6, cada árvore enxerga só uma fração das
colunas. Uma coluna com cópia tem ~2x a chance de entrar na árvore, então as features fortes
aparecem mais vezes (efeito parecido com dar peso 1.5x a grade_on_m e gt_idade_ok15).
A cópia é idêntica, então não cria informação nova; só muda a frequência de amostragem.
"""
import numpy as np
import pandas as pd

SUF = "__dup"
# colunas que a análise já mostrou como as mais úteis; sempre mantidas (e duplicadas se estiverem no top)
SEMPRE_MANTER = ["grade_on_m", "gt_idade_ok15", "sent_mf25", "sent_z24", "sent_mf49", "sx_mf_max"]


class SelecaoFeatures:
    def __init__(self, caminho_importancia, imp_min=5e-4, n_dup=20, copias=1,
                 peso_dup=None, sempre_manter=SEMPRE_MANTER):
        imp = pd.read_csv(caminho_importancia)
        imp = imp.rename(columns={imp.columns[0]: "f", imp.columns[1]: "imp"})
        imp = imp.sort_values("imp", ascending=False).reset_index(drop=True)
        self.imp = imp
        manter = set(imp.loc[imp["imp"] >= imp_min, "f"]) | set(sempre_manter)
        self.manter = [f for f in imp["f"] if f in manter]                 # ordem de importância
        self.dup = [f for f in imp["f"].head(n_dup) if f in manter]         # top-n a duplicar
        self.copias = int(copias)

    def colunas(self, cols):
        """Lista final de colunas: poda das fracas (colunas fora do CSV sempre ficam) + nomes das cópias."""
        conhecidas = set(self.imp["f"])
        manter = set(self.manter)
        keep = [c for c in cols if c in manter or c not in conhecidas]
        dups = [f"{c}{SUF}{k}" for k in range(1, self.copias + 1) for c in self.dup if c in keep]
        return keep + dups

    def adicionar_dups(self, d: pd.DataFrame) -> pd.DataFrame:
        """Cria as colunas __dupK no DataFrame (usar em treino, validação, teste, refit e previsão)."""
        extras = {f"{c}{SUF}{k}": d[c] for k in range(1, self.copias + 1) for c in self.dup if c in d.columns}
        return pd.concat([d.drop(columns=[c for c in extras if c in d.columns]), pd.DataFrame(extras, index=d.index)], axis=1)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        manter = [c for c in self.manter if c in X.columns]
        # colunas que não estavam no arquivo de importância (ex.: criadas depois) são mantidas
        novas = [c for c in X.columns if c not in set(self.imp["f"])]
        base = manter + novas
        out = X[base].copy()
        extras = {}
        for k in range(1, self.copias + 1):
            for c in self.dup:
                if c in out.columns:
                    extras[f"{c}{SUF}{k}"] = out[c]
        if extras:
            out = pd.concat([out, pd.DataFrame(extras, index=out.index)], axis=1)
        return out

    @staticmethod
    def importancia_agregada(valores, colunas):
        """Soma a importância das cópias de volta na coluna original (para relatórios)."""
        s = pd.Series(np.asarray(valores, dtype=float), index=list(colunas))
        s.index = [c.split(SUF)[0] for c in s.index]
        return s.groupby(level=0).sum().sort_values(ascending=False)


# varredura sugerida (escolher pela AP de VALIDAÇÃO, nunca pelo teste)
GRADE = [
    dict(imp_min=0,    n_dup=0,  copias=0),   # controle (v8a atual)
    dict(imp_min=5e-4, n_dup=0,  copias=0),   # só poda
    dict(imp_min=5e-4, n_dup=20, copias=1),   # poda + top20 x2
    dict(imp_min=5e-4, n_dup=20, copias=2),   # poda + top20 x3
    dict(imp_min=1e-3, n_dup=30, copias=1),   # poda mais forte
]

if __name__ == "__main__":
    # teste rápido de sanidade com dados sintéticos
    imp = pd.read_csv("importancia_features_v8a.csv")
    nomes = imp.iloc[:, 0].tolist()
    X = pd.DataFrame(np.random.rand(10, len(nomes)), columns=nomes)
    X["coluna_nova"] = 1.0
    for cfg in GRADE:
        sel = SelecaoFeatures("importancia_features_v8a.csv", **cfg)
        print(cfg, sel.transform(X).shape)
