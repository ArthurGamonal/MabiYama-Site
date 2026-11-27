"""mb_nb26.py -- as features do notebook 26 (0,65) portadas para o formato LONGO do treinar.py (máquina x dia x peça).

Blocos (todos com a regra das 05:00: só informação anterior a D 05:00):
  ESTÁTICO  (não depende do histórico de falhas): sensores horários (ciclo, CUSUM, deriva r19/r20), erros por janela,
            erros desde a visita, cadastro, operação (dia anterior), sinal e deriva do SENSOR de cada peça.
  DEPENDENTE (depende do histórico de falhas -> precisa ser "congelado" em out-dez): taxa histórica, visitas desde a
            última falha / troca preventiva, peça instalada (idade, fornecedor, técnico).
            Em out-dez a tabela de falhas acaba em 30/09: as falhas escondidas são inferidas pelos ERROS (pseudo-falhas).
Todas as colunas saem com prefixo nb_ (estáticas) ou nbd_ (dependentes) para não colidir com as do mb.py.
Arquivos lidos de U: base2_completa.parquet (ou .csv), falhas.parquet, manutencao.parquet, errostratado.* (opcional).
"""
import os, warnings
import numpy as np, pandas as pd

PECAS = ["peca_A", "peca_B", "peca_C", "peca_D"]
SENSORES = ["sensor_a", "sensor_b", "sensor_c", "sensor_d"]
H, DIA = np.timedelta64(1, "h"), np.timedelta64(1, "D")
HORA = 5
FIM_FALHAS = pd.Timestamp("2023-09-30 23:59")
CORTE_CAL = pd.Timestamp("2023-09-30 05:00")       # o calendário de visitas só usa o que é anterior ao 1º dia a prever
K_DIAS = 90
K_PRECISAO_MIN = 0.5
MAX_CATEG = 20
MUITO = 24 * 400
warnings.filterwarnings("ignore")


# ------------------------------------------------------------------ utilitários (iguais aos do notebook)
def dia_num(t):
    return (np.asarray(t, dtype="datetime64[ns]").astype("datetime64[D]") - np.datetime64("1970-01-01", "D")).astype(int)

def _por_maquina(ev):
    return {m: np.sort(g["data_hora"].to_numpy("datetime64[ns]")) for m, g in ev.groupby("id_maquina")}

def contar_entre(pts, ev, inicio, fim):
    res = np.zeros(len(pts), dtype=int)
    arrs = _por_maquina(ev)
    for m, idx in pts.groupby("id_maquina").indices.items():
        arr = arrs.get(m)
        if arr is not None:
            res[idx] = np.searchsorted(arr, fim[idx], "left") - np.searchsorted(arr, inicio[idx], "left")
    return res

def contar_janela(pts, ev, ini_h, fim_h):
    t = pts["data_hora"].to_numpy("datetime64[ns]")
    return contar_entre(pts, ev, t + np.timedelta64(int(round(ini_h * 60)), "m"), t + np.timedelta64(int(round(fim_h * 60)), "m"))

def horas_desde_ultimo(pts, ev):
    res = np.full(len(pts), np.nan)
    arrs = _por_maquina(ev)
    t_all = pts["data_hora"].to_numpy("datetime64[ns]")
    for m, idx in pts.groupby("id_maquina").indices.items():
        arr = arrs.get(m)
        if arr is not None:
            t = t_all[idx]
            pos = np.searchsorted(arr, t, "left") - 1
            ok = pos >= 0
            res[idx[ok]] = (t[ok] - arr[pos[ok]]) / H
    return res

def ultimo_valor(pts, ev, coluna, horas):
    res = np.full(len(pts), np.nan, dtype=object)
    t_all = pts["data_hora"].to_numpy("datetime64[ns]")
    idx_pts = pts.groupby("id_maquina").indices
    for m, g in ev.groupby("id_maquina"):
        idx = idx_pts.get(m)
        if idx is None:
            continue
        g = g.sort_values("data_hora")
        arr = g["data_hora"].to_numpy("datetime64[ns]")
        vals = g[coluna].to_numpy(object)
        t = t_all[idx]
        pos = np.searchsorted(arr, t, "left") - 1
        ok = (pos >= 0) & (t - arr[np.maximum(pos, 0)] <= horas * H)
        res[idx[ok]] = vals[pos[ok]]
    return res

def erros_ponderados(pts, ev, meia_vida_h, janela_h):
    res = np.zeros(len(pts))
    arrs = _por_maquina(ev)
    t_all = pts["data_hora"].to_numpy("datetime64[ns]")
    for m, idx in pts.groupby("id_maquina").indices.items():
        arr = arrs.get(m)
        if arr is None:
            continue
        a_all = np.searchsorted(arr, t_all[idx] - janela_h * H, "left")
        b_all = np.searchsorted(arr, t_all[idx], "left")
        for i, a, b in zip(idx, a_all, b_all):
            if b > a:
                res[i] = np.sum(0.5 ** (((t_all[i] - arr[a:b]) / H) / meia_vida_h))
    return res

def cusum(x, ciclo, k):
    up, dn = np.zeros(len(x)), np.zeros(len(x))
    cu = cd = 0.0
    ant = None
    for i, (xi, ci) in enumerate(zip(x.tolist(), np.asarray(ciclo).tolist())):
        if ci != ant:
            cu = cd = 0.0
            ant = ci
        if xi == xi:
            cu = max(0.0, cu + xi - k)
            cd = max(0.0, cd - xi - k)
        up[i], dn[i] = cu, cd
    return up, dn

def _para_data(s):
    if pd.api.types.is_numeric_dtype(s):
        s = pd.to_datetime(s, unit="ms")
    else:
        s = pd.to_datetime(s, errors="coerce", dayfirst=s.astype(str).str.match(r"^\d{2}/").any())
    return s.astype("datetime64[ns]")

def _limpar(df):
    df = df.copy()
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
    df["data_hora"] = _para_data(df["data_hora"])
    df["id_maquina"] = df["id_maquina"].astype(str).str.strip().str.upper()
    return df.dropna(subset=["data_hora"])

def _nome_peca(s):
    s = s.astype(str).str.strip()
    return s.where(s.str.startswith("peca_"), "peca_" + s.str.replace("peca", "", case=False).str.strip(" _"))

def _ler(U, *nomes, obrigatorio=True):
    for n in nomes:
        p = U + n
        if os.path.exists(p):
            df = pd.read_parquet(p) if p.endswith(".parquet") else pd.read_csv(p)
            if df.shape[1] == 1:
                df = pd.read_csv(p, sep=";")
            return df
    if obrigatorio:
        raise FileNotFoundError(f"{U}{' / '.join(nomes)}")
    return None


# ------------------------------------------------------------------ a classe principal
class NB26:
    COLS_OP_NUM = ["pecas_produzidas", "meta_producao", "consumo_energia_kwh", "utilizacao_pct", "tempo_setup_min",
                   "tempo_reconhecimento_medio_min"]
    COLS_CAT_DIA = ["produto_fabricado", "operador", "versao_firmware_ultima"]
    COLS_FIXAS = ["capacidade_nominal_h", "linha_producao"]
    COLS_RISCO = ["ind_quebra", "classe_risco"]

    def __init__(self, U, corte_z, usar_risco=False, verbose=True, usar_taxas=False):
        """corte_z: data a partir da qual NADA entra nas estatísticas de padronização / calibrações (= início da validação)."""
        self.U, self.corte_z, self.usar_risco, self.v = U, pd.Timestamp(corte_z), usar_risco, verbose
        self.usar_taxas = usar_taxas   # taxas por fornecedor/técnico: target-encoding calculado nas PRÓPRIAS linhas do treino -> vaza rótulo
        self._carregar()
        self._calendario()
        self._grade_sensores()
        self._operacao()
        self.mapas_ok = False

    def log(self, *a):
        if self.v:
            print("[nb26]", *a, flush=True)

    # ---------------------------------------------------------- carga
    def _carregar(self):
        U = self.U
        cols_base = (["data_hora", "id_maquina", "modelo", "idade", "potencia_kw", "fabricante"] + SENSORES
                     + [f"{s}_imputado" for s in SENSORES] + ["n_erros_hora", "codigo_erro_ultimo", "severidade_max_hora"]
                     + self.COLS_OP_NUM + self.COLS_CAT_DIA + self.COLS_FIXAS + self.COLS_RISCO)
        if os.path.exists(U + "base2_completa.parquet"):
            import pyarrow.parquet as pq
            ex = pq.read_schema(U + "base2_completa.parquet").names
            raw = pd.read_parquet(U + "base2_completa.parquet", columns=[c for c in cols_base if c in ex])
        else:
            raw = pd.read_csv(U + "base2_completa.csv")
            raw = raw[[c for c in cols_base if c in raw.columns]]
        raw = _limpar(raw).sort_values(["id_maquina", "data_hora"]).reset_index(drop=True)
        self.raw = raw
        self.base = raw[[c for c in cols_base[:6 + 4 + 4 + 3] if c in raw.columns]]
        f = _limpar(_ler(U, "falhas.parquet", "falhas.csv"))
        f["peca"] = _nome_peca(f["peca"])
        self.falhas = f[f["data_hora"] <= FIM_FALHAS].sort_values("data_hora").reset_index(drop=True)

        er = _ler(U, "errostratado.parquet", "errostratado.csv", obrigatorio=False)
        if er is not None:
            er = _limpar(er)
            if "codigo_erro" not in er.columns:
                cc = [c for c in er.columns if c.startswith("codigo_E")]
                er["codigo_erro"] = er[cc].fillna(0).astype(int).idxmax(axis=1).str.replace("codigo_", "")
        else:
            h = raw[raw["n_erros_hora"].fillna(0) > 0]
            er = h.loc[h.index.repeat(h["n_erros_hora"].astype(int)), ["id_maquina", "data_hora", "codigo_erro_ultimo", "severidade_max_hora"]]
            er = er.rename(columns={"codigo_erro_ultimo": "codigo_erro", "severidade_max_hora": "severidade"})
        er["codigo_erro"] = er["codigo_erro"].astype(str).str.strip().str.upper()
        er["codigo_erro"] = er["codigo_erro"].where(er["codigo_erro"].str.startswith("E"), "E" + er["codigo_erro"])
        if "severidade" in er.columns:
            er["severidade"] = er["severidade"].astype(str).str.strip().str.lower().str.replace("média", "media")
        self.erros = er.sort_values("data_hora").reset_index(drop=True)
        self.CODIGOS = sorted(self.erros["codigo_erro"].unique())

        m = _ler(U, "manutencao.parquet", "manutencao.csv", obrigatorio=False)
        self.manut = None
        if m is not None:
            m = _limpar(m)
            if "peca" not in m.columns:
                fl = m[[c for c in PECAS if c in m.columns]].fillna(0).astype(int)
                m["peca"] = fl.idxmax(axis=1).where(fl.sum(axis=1) > 0)
            m = m.dropna(subset=["peca"])
            m["peca"] = _nome_peca(m["peca"])
            self.manut = m.sort_values("data_hora").reset_index(drop=True)

        cad_cols = [c for c in ["modelo", "idade", "potencia_kw", "fabricante"] if c in raw.columns]
        self.cad = raw.groupby("id_maquina")[cad_cols].first()
        self.cad["idade"] = pd.to_numeric(self.cad.get("idade"), errors="coerce")
        self.MAQUINAS = sorted(self.cad.index)
        self.INICIO_OBS = np.datetime64(pd.Timestamp("2022-10-01") + pd.Timedelta(hours=HORA), "ns")
        self.FIM_OBS = np.datetime64(FIM_FALHAS, "ns")
        # trocas: corretiva = coincide (máquina, peça, dia) com uma falha
        self.MANUT_ALL = self.MANUT_PREV = None
        self.TEM_FORN = self.TEM_TEC = False
        if self.manut is not None:
            ch = (self.falhas.assign(dia=self.falhas["data_hora"].dt.normalize())[["id_maquina", "peca", "dia"]]
                  .drop_duplicates().assign(_c=1))
            ma = self.manut.assign(dia=self.manut["data_hora"].dt.normalize()).merge(ch, on=["id_maquina", "peca", "dia"], how="left")
            ma["corretiva"] = ma["_c"].notna()
            self.MANUT_ALL = ma.drop(columns=["dia", "_c"]).sort_values("data_hora").reset_index(drop=True)
            self.MANUT_PREV = self.MANUT_ALL[~self.MANUT_ALL["corretiva"]].reset_index(drop=True)
            self.TEM_FORN = "fornecedor_peca" in self.MANUT_ALL.columns
            self.TEM_TEC = "tecnico" in self.MANUT_ALL.columns
            self.FORNECEDORES = sorted(self.MANUT_ALL["fornecedor_peca"].dropna().astype(str).unique()) if self.TEM_FORN else []
        self.log(f"base {len(raw):,} leituras | falhas {len(self.falhas):,} | erros {len(self.erros):,} | manut {0 if self.manut is None else len(self.manut):,}")

    # ---------------------------------------------------------- calendário de visitas
    def _calendario(self):
        vis = pd.concat([d[["id_maquina", "data_hora"]] for d in (self.manut, self.falhas) if d is not None])
        vis = vis[vis["data_hora"] < CORTE_CAL].copy()
        vis["dia"] = dia_num(vis["data_hora"] - pd.Timedelta(hours=6))
        dv = vis.drop_duplicates(["id_maquina", "dia"]).sort_values(["id_maquina", "dia"])
        difs = dv.groupby("id_maquina")["dia"].diff().dropna().astype(int)
        sel = difs[(difs >= 7) & (difs <= 31)]
        self.PERIODO = int(sel.mode().iloc[0]) if len(sel) else 15
        cal = dv.assign(fase=dv["dia"] % self.PERIODO).groupby("id_maquina")["fase"].agg(lambda s: s.mode().iloc[0])
        self.FASE = cal.to_dict()
        self.log(f"período das visitas: {self.PERIODO} dias | máquinas com fase: {len(self.FASE)}")

    def visitas_ao_redor(self, pts):
        t = pts["data_hora"].to_numpy("datetime64[ns]")
        fase = pts["id_maquina"].map(self.FASE).to_numpy(float)
        e = dia_num(t - np.timedelta64(6, "h") - np.timedelta64(1, "ns"))
        ultimo = e - np.mod(e - fase, self.PERIODO)
        ult_t = (np.datetime64("1970-01-01", "D") + np.nan_to_num(ultimo).astype(int) * DIA).astype("datetime64[ns]") + 6 * H
        return ult_t, np.isnan(fase)

    # ---------------------------------------------------------- sensores horários
    def _grade_sensores(self):
        corte, PERIODO, FASE = self.corte_z, self.PERIODO, self.FASE
        partes = []
        for m, g in self.base.groupby("id_maquina"):
            g = g.drop_duplicates("data_hora").set_index("data_hora").sort_index()
            idx = pd.date_range(g.index.min().floor("h"), g.index.max().floor("h"), freq="h")
            g = g.reindex(idx)
            antes = g.index < corte
            f = pd.DataFrame(index=idx)
            fase = FASE.get(m, np.nan)
            e = dia_num(idx - pd.Timedelta(hours=6))
            ciclo = np.floor((e - fase) / PERIODO) if np.isfinite(fase) else np.zeros(len(idx))
            for s in SENSORES:
                v = g[s].astype(float)
                flag = f"{s}_imputado"
                if flag in g.columns:
                    imp = g[flag].astype(str).str.strip().str.lower().isin(["true", "1", "1.0"]).to_numpy()
                    v = v.where(~imp)
                mu, sd = v[antes].mean(), v[antes].std()
                if not np.isfinite(sd) or sd == 0:
                    mu, sd = v.mean(), v.std()
                z = (v - mu) / sd
                f[f"{s}_z_ult"] = z.ffill(limit=6)
                f[f"{s}_z_24h"] = z.rolling(24, min_periods=6).mean()
                f[f"{s}_z_72h"] = z.rolling(72, min_periods=18).mean()
                f[f"{s}_var_24h"] = f[f"{s}_z_24h"] - z.shift(96).rolling(72, min_periods=18).mean()
                gz = f[f"{s}_z_24h"].groupby(ciclo)
                f[f"{s}_ciclo_max"] = gz.cummax()
                f[f"{s}_ciclo_min"] = gz.cummin()
                f[f"{s}_ciclo_media"] = (z.fillna(0).groupby(ciclo).cumsum()
                                         / z.notna().astype(int).groupby(ciclo).cumsum().replace(0, np.nan))
                ok = z.notna().astype(float)
                hh = pd.Series(np.ones(len(idx)), index=idx).groupby(ciclo).cumsum()
                xx, yy = hh.where(z.notna(), 0.0), z.fillna(0.0)
                acum = lambda x: x.groupby(ciclo).cumsum()
                n_, sx, sy, sxy, sxx = acum(ok), acum(xx), acum(yy), acum(xx * yy), acum(xx * xx)
                den = n_ * sxx - sx ** 2
                f[f"{s}_c2_tend"] = (24 * (n_ * sxy - sx * sy) / den.where(den > 0)).where(n_ >= 24)
                f[f"{s}_c2_fora"] = (z.abs() > 2).astype(float).groupby(ciclo).cumsum()
                normal = v.shift(48).rolling(240, min_periods=48).median()
                pct = 100 * (v / normal - 1)
                f[f"{s}_n16_pct24"] = pct.rolling(24, min_periods=12).mean()
                f[f"{s}_n16_pct48"] = pct.rolling(48, min_periods=24).mean()
                x6 = pct.rolling(6, min_periods=3).mean()
                ref6 = x6[antes].dropna()
                sd6 = 1.4826 * np.median(np.abs(ref6 - np.median(ref6))) if len(ref6) > 48 else np.nan
                if not np.isfinite(sd6) or sd6 == 0:
                    sd6 = np.nanstd(x6.to_numpy()) or 1.0
                cu, cd = cusum((x6 / sd6).to_numpy(), ciclo, 0.25)
                f[f"{s}_n16_cusum_up"], f[f"{s}_n16_cusum_dn"] = cu, cd
                f[f"{s}_r19_pct3"] = pct.rolling(3, min_periods=2).mean()
                f[f"{s}_r19_pct6"] = x6
                f[f"{s}_r19_pct12"] = pct.rolling(12, min_periods=6).mean()
                f[f"{s}_r19_max24"] = x6.rolling(24, min_periods=12).max()
                f[f"{s}_r19_min24"] = x6.rolling(24, min_periods=12).min()
                f[f"{s}_r19_incl12"] = x6 - x6.shift(12)
                f[f"{s}_r19_incl24"] = x6 - x6.shift(24)
                ref1 = pct[antes].dropna()
                sd1 = 1.4826 * np.median(np.abs(ref1 - np.median(ref1))) if len(ref1) > 48 else np.nan
                if not np.isfinite(sd1) or sd1 == 0:
                    sd1 = np.nanstd(pct.to_numpy()) or 1.0
                zr = pct / sd1
                z3 = zr.rolling(3, min_periods=2).mean()
                f[f"{s}_r20_pct1"] = pct
                f[f"{s}_r20_pct2"] = pct.rolling(2, min_periods=1).mean()
                f[f"{s}_r20_zr3"] = z3
                f[f"{s}_r20_zr6"] = zr.rolling(6, min_periods=3).mean()
                f[f"{s}_r20_zr12"] = zr.rolling(12, min_periods=6).mean()
                f[f"{s}_r20_ewm3"] = zr.ewm(halflife=3, ignore_na=True).mean()
                f[f"{s}_r20_ewm8"] = zr.ewm(halflife=8, ignore_na=True).mean()
                f[f"{s}_r20_acima24"] = (z3 > 1).astype(float).rolling(24, min_periods=12).sum()
                f[f"{s}_r20_abaixo24"] = (z3 < -1).astype(float).rolling(24, min_periods=12).sum()
                for k in (6, 12, 18, 24, 30, 36):
                    f[f"{s}_r20_lag{k}"] = z3.shift(k)
            f = f.astype("float32")
            f["id_maquina"] = m
            partes.append(f.rename_axis("data_hora").reset_index())
        self.GRADE = pd.concat(partes, ignore_index=True).sort_values("data_hora")
        self.COLS_SENS = [c for c in self.GRADE.columns if c.startswith("sensor_")]
        self.log(f"grade de sensores: {len(self.GRADE):,} horas, {len(self.COLS_SENS)} colunas")

    # ---------------------------------------------------------- operação (dia anterior)
    def _operacao(self):
        ex = self.raw
        op_num = [c for c in self.COLS_OP_NUM if c in ex]
        cat_dia = [c for c in self.COLS_CAT_DIA if c in ex]
        fixas_c = [c for c in self.COLS_FIXAS if c in ex]
        risco_c = [c for c in self.COLS_RISCO if c in ex] if self.usar_risco else []
        ex = ex[["data_hora", "id_maquina"] + op_num + cat_dia + fixas_c + risco_c].copy()
        ex["dia"] = (ex["data_hora"] - pd.Timedelta(hours=HORA)).dt.normalize() + pd.Timedelta(days=1)
        for c in op_num + (["ind_quebra"] if "ind_quebra" in risco_c else []):
            ex[c] = pd.to_numeric(ex[c], errors="coerce")

        def moda_por_dia(col):
            v = ex[["id_maquina", "dia", col]].dropna()
            v[col] = v[col].astype(str)
            cont = v.groupby(["id_maquina", "dia", col]).size().reset_index(name="n")
            return cont.sort_values("n").drop_duplicates(["id_maquina", "dia"], keep="last").set_index(["id_maquina", "dia"])[col]

        agg = ex.groupby(["id_maquina", "dia"])[op_num].mean() if op_num else pd.DataFrame(index=ex.groupby(["id_maquina", "dia"]).size().index)
        agg.columns = [f"op_{c}" for c in agg.columns]
        if {"pecas_produzidas", "meta_producao"} <= set(op_num):
            agg["op_prod_sobre_meta"] = agg["op_pecas_produzidas"] / agg["op_meta_producao"].where(agg["op_meta_producao"] > 0)
        for c in cat_dia:
            agg[f"op_{c}_n_diferentes"] = ex.groupby(["id_maquina", "dia"])[c].nunique()
            if ex[c].nunique() <= MAX_CATEG:
                moda = moda_por_dia(c)
                for val in sorted(ex[c].dropna().astype(str).unique()):
                    agg[f"op_{c}_{val}"] = (moda.reindex(agg.index) == val).astype(float)
        if "ind_quebra" in risco_c:
            agg["risco_ind_quebra"] = ex.groupby(["id_maquina", "dia"])["ind_quebra"].mean()
        if "classe_risco" in risco_c:
            moda = moda_por_dia("classe_risco")
            for val in sorted(ex["classe_risco"].dropna().astype(str).unique()):
                agg[f"risco_classe_{val}"] = (moda.reindex(agg.index) == val).astype(float)
        agg = agg.sort_index()
        for c in [c for c in ["op_consumo_energia_kwh", "op_utilizacao_pct", "op_pecas_produzidas", "op_tempo_setup_min"] if c in agg]:
            sem = agg.groupby(level=0)[c].transform(lambda s: s.shift(1).rolling(7, min_periods=3).mean())
            agg[f"{c}_var7d"] = agg[c] - sem
        self.op_tab = agg
        fx = pd.DataFrame(index=self.MAQUINAS)
        if "capacidade_nominal_h" in fixas_c:
            fx["cad_capacidade_nominal_h"] = pd.to_numeric(ex.groupby("id_maquina")["capacidade_nominal_h"].first(), errors="coerce").reindex(self.MAQUINAS)
        if "linha_producao" in fixas_c:
            lin = ex.groupby("id_maquina")["linha_producao"].first().astype(str).reindex(self.MAQUINAS)
            for val in sorted(lin.dropna().unique()):
                fx[f"cad_linha_{val}"] = (lin == val).astype(float)
        self.fixas = fx
        self.log(f"operação: {self.op_tab.shape[1]} colunas | cadastro fixo: {fx.shape[1]}")

    # ---------------------------------------------------------- bloco ESTÁTICO (wide, por máquina-dia)
    @staticmethod
    def _pts(keys):
        p = keys[["id_maquina"]].copy().reset_index(drop=True)
        p["id_maquina"] = p["id_maquina"].astype(str)
        p["data_hora"] = (pd.to_datetime(keys["dia"]).reset_index(drop=True) + pd.Timedelta(hours=HORA)).astype("datetime64[ns]")
        return p

    def estatico_wide(self, keys):
        """keys: DataFrame com id_maquina, dia (únicos). Retorna DataFrame (index = ordem de keys) com colunas nb_*."""
        pts = self._pts(keys)
        G = self.GRADE.copy()
        G["id_maquina"] = G["id_maquina"].astype(str)
        s = pd.merge_asof(pts.reset_index().sort_values("data_hora"), G, on="data_hora", by="id_maquina",
                          allow_exact_matches=False, direction="backward", tolerance=pd.Timedelta(hours=6))
        s = s.set_index("index").sort_index()
        X = pd.DataFrame(index=pts.index)
        X[self.COLS_SENS] = s[self.COLS_SENS]
        for w in (6, 24, 48, 72, 168):
            X[f"erros_{w}h"] = contar_janela(pts, self.erros, -w, 0)
        X["horas_desde_erro"] = np.nan_to_num(np.clip(horas_desde_ultimo(pts, self.erros), 0, 720), nan=720)
        for c in self.CODIGOS:
            X[f"erros_{c}_48h"] = contar_janela(pts, self.erros[self.erros["codigo_erro"] == c], -48, 0)
        if "severidade" in self.erros.columns:
            X["erros_alta_24h"] = contar_janela(pts, self.erros[self.erros["severidade"] == "alta"], -24, 0)
        X["c2_erros_recentes"] = erros_ponderados(pts, self.erros, 24, 168)
        ult_t, _ = self.visitas_ao_redor(pts)
        t = pts["data_hora"].to_numpy("datetime64[ns]")
        X["ciclo_erros"] = contar_entre(pts, self.erros, ult_t, t)
        for c in self.CODIGOS:
            X[f"ciclo_erros_{c}"] = contar_entre(pts, self.erros[self.erros["codigo_erro"] == c], ult_t, t)
        if "severidade" in self.erros.columns:
            X["ciclo_erros_alta"] = contar_entre(pts, self.erros[self.erros["severidade"] == "alta"], ult_t, t)
        # operação (dia anterior) + cadastro fixo
        chave = pd.MultiIndex.from_arrays([pts["id_maquina"].to_numpy(), pts["data_hora"].dt.normalize() - pd.Timedelta(days=1)])
        op = pd.DataFrame(self.op_tab.reindex(chave).to_numpy(dtype="float32", na_value=np.nan), index=pts.index, columns=self.op_tab.columns)
        fx = pd.DataFrame(self.fixas.reindex(pts["id_maquina"].to_numpy()).to_numpy(dtype="float32", na_value=np.nan), index=pts.index, columns=self.fixas.columns)
        X = pd.concat([X, op, fx], axis=1)
        X.columns = [f"nb_{c}" for c in X.columns]
        X = X.astype("float32")
        X.index = keys.index
        return X

    # ---------------------------------------------------------- mapas peça->sensor / peça->códigos (aprendidos só no TREINO)
    def aprender(self, keys, W, Ltr):
        """keys/W: entrada e saída de estatico_wide; Ltr: DataFrame (id_maquina, dia, peca, y) das linhas de TREINO candidatas.
        Aprende: sensor de cada peça, códigos de cada peça, taxas por fornecedor/técnico e a calibração das pseudo-falhas."""
        d = Ltr[["id_maquina", "dia", "peca", "y"]].reset_index(drop=True)
        Wk = W.set_axis(pd.MultiIndex.from_frame(keys[["id_maquina", "dia"]].reset_index(drop=True)))
        Wd = Wk.reindex(pd.MultiIndex.from_frame(d[["id_maquina", "dia"]])).reset_index(drop=True)
        self.SENSOR_DA_PECA, self.CODIGOS_DA_PECA = {}, {p: [] for p in PECAS}
        for p in PECAS:
            k = (d["peca"].astype(str) == p).to_numpy()
            y = d.loc[k, "y"].to_numpy() == 1
            ef = {}
            for s in SENSORES:
                v = Wd[f"nb_{s}_ciclo_media"].to_numpy(float)[k]
                dp = np.nanstd(v)
                ef[s] = (np.nanmean(v[y]) - np.nanmean(v[~y])) / dp if dp > 0 and y.any() else 0.0
            melhor = max(ef, key=lambda q: abs(ef[q]))
            self.SENSOR_DA_PECA[p] = (melhor, 1 if ef[melhor] > 0 else -1)
        lifts = {}
        for c in self.CODIGOS:
            tem = Wd[f"nb_ciclo_erros_{c}"].to_numpy() > 0
            lifts[c] = {}
            for p in PECAS:
                k = (d["peca"].astype(str) == p).to_numpy()
                y = d["y"].to_numpy()[k] == 1
                lifts[c][p] = (tem[k][y].mean() + 1e-6) / (tem[k][~y].mean() + 1e-6) if y.any() else 0.0
            mp = max(lifts[c], key=lifts[c].get)
            if lifts[c][mp] >= 1.5:
                self.CODIGOS_DA_PECA[mp].append(c)
        self.log("sensor por peça:", self.SENSOR_DA_PECA, "| códigos por peça:", self.CODIGOS_DA_PECA)
        # taxa geral e fornecedor/técnico
        dias0 = max(1.0, (np.datetime64(self.corte_z, "ns") - self.INICIO_OBS) / DIA)
        self.TAXA_GERAL = {p: len(self.falhas[(self.falhas["peca"] == p) & (self.falhas["data_hora"] < self.corte_z)]) / (len(self.MAQUINAS) * dias0) for p in PECAS}
        self.TAXA_FORN, self.TAXA_TEC = {}, {}
        if self.usar_taxas and self.MANUT_ALL is not None and (self.TEM_FORN or self.TEM_TEC):
            pts = self._pts(d[["id_maquina", "dia"]])
            for p in PECAS:
                k = (d["peca"].astype(str) == p).to_numpy()
                y = pd.Series(d["y"].to_numpy()[k])
                ev = self.MANUT_ALL[self.MANUT_ALL["peca"] == p]
                sub = pts[k].reset_index(drop=True)
                if self.TEM_FORN:
                    self.TAXA_FORN[p] = self._taxa(pd.Series(ultimo_valor(sub, ev, "fornecedor_peca", MUITO)), y)
                if self.TEM_TEC:
                    self.TAXA_TEC[p] = self._taxa(pd.Series(ultimo_valor(sub, ev, "tecnico", MUITO)), y)
        self._calibrar_pseudo()
        self.mapas_ok = True

    @staticmethod
    def _taxa(chave, y, alpha=50):
        base_ = y.mean()
        g = pd.DataFrame({"k": chave.astype(str).to_numpy(), "y": y.to_numpy()}).groupby("k")["y"].agg(["sum", "size"])
        return ((g["sum"] + alpha * base_) / (g["size"] + alpha)).to_dict()

    # ---------------------------------------------------------- bloco ESTÁTICO por peça (long)
    def estatico_long(self, keys, W):
        """Une, por peça, as colunas compartilhadas (W) e as específicas (sinal e deriva do sensor da peça)."""
        assert self.mapas_ok, "rode aprender() antes"
        partes = []
        for p in PECAS:
            s, d = self.SENSOR_DA_PECA[p]
            F = pd.DataFrame(index=W.index)
            g = lambda nome: W[f"nb_{s}_{nome}"]
            F["nbp_sinal_24h"] = d * g("z_24h")
            F["nbp_sinal_var"] = d * g("var_24h")
            F["nbp_sinal_ciclo"] = d * (g("ciclo_max") if d > 0 else g("ciclo_min"))
            F["nbp_sinal_media"] = d * g("ciclo_media")
            F["nbp_sinal_tend"] = d * g("c2_tend")
            cods = self.CODIGOS_DA_PECA[p]
            F["nbp_erros_ciclo"] = W[[f"nb_ciclo_erros_{c}" for c in cods]].sum(axis=1) if cods else 0.0
            F["nbp_erros_48h"] = W[[f"nb_erros_{c}_48h" for c in cods]].sum(axis=1) if cods else 0.0
            for m in ["pct3", "pct6", "pct12", "incl12", "incl24"]:
                F[f"nbp_r19_{m}"] = d * g(f"r19_{m}")
            F["nbp_r19_pico"] = d * (g("r19_max24") if d > 0 else g("r19_min24"))
            for m in ["pct1", "pct2", "zr3", "zr6", "zr12", "ewm3", "ewm8"] + [f"lag{k}" for k in (6, 12, 18, 24, 30, 36)]:
                F[f"nbp_r20_{m}"] = d * g(f"r20_{m}")
            F["nbp_r20_horas"] = g("r20_acima24" if d > 0 else "r20_abaixo24")
            out = pd.concat([keys[["id_maquina", "dia"]], W, F.astype("float32")], axis=1)
            out["peca"] = pd.Categorical([p] * len(out), categories=PECAS)
            partes.append(out)
        return pd.concat(partes, ignore_index=True)

    # ---------------------------------------------------------- bloco DEPENDENTE do histórico de falhas
    def _hist(self, pts, corte):
        t = pts["data_hora"].to_numpy("datetime64[ns]")
        lim = np.minimum(t, self.FIM_OBS if corte is None else min(self.FIM_OBS, np.datetime64(corte, "ns")))
        dias_obs = np.maximum((lim - self.INICIO_OBS) / DIA, 0)
        ini = np.full(len(pts), self.INICIO_OBS)
        out = {}
        for p in PECAS:
            n = contar_entre(pts, self.falhas[self.falhas["peca"] == p], ini, lim)
            out[p] = 30 * (n + self.TAXA_GERAL[p] * K_DIAS) / (dias_obs + K_DIAS)
        return out

    def dep_long(self, keys, corte=None, extra=None, usar_falhas_reais=True):
        """keys: id_maquina, dia (únicos). corte: data em que o histórico de falhas congela (None = usa tudo que existe).
        extra: pseudo-falhas (id_maquina, data_hora, peca) a somar ao histórico (só para a recorrência)."""
        assert self.mapas_ok
        pts = self._pts(keys)
        hist = self._hist(pts, corte)
        fal = self.falhas if corte is None else self.falhas[self.falhas["data_hora"] < corte]
        if not usar_falhas_reais:                      # modo "só pseudo": mesma informação em treino e em out-dez
            fal = self.falhas.iloc[0:0]
        if extra is not None and len(extra):
            fal = pd.concat([fal[["id_maquina", "data_hora", "peca"]], extra[["id_maquina", "data_hora", "peca"]]])
        vis = lambda ev: np.ceil(horas_desde_ultimo(pts, ev) / (24 * self.PERIODO))
        qualquer = np.clip(vis(fal), 0, 12)
        if self.MANUT_ALL is not None:
            ev_all = self.MANUT_ALL if corte is None else self.MANUT_ALL[(~self.MANUT_ALL["corretiva"]) | (self.MANUT_ALL["data_hora"] < corte)]
        partes = []
        for p in PECAS:
            F = pd.DataFrame(index=keys.index)
            F["nbd_hist_taxa"] = hist[p]
            F["nbd_hist_taxa_total"] = sum(hist.values())
            v = vis(fal[fal["peca"] == p])
            F["nbd_rec_falha_vis"] = np.clip(v, 0, 12)
            F["nbd_rec_falha_nunca"] = np.isnan(v).astype(float)
            F["nbd_rec_falha_vis_qualquer"] = qualquer
            if self.MANUT_PREV is not None:
                vp = vis(self.MANUT_PREV[self.MANUT_PREV["peca"] == p])
                F["nbd_rec_prev_vis"] = np.clip(vp, 0, 12)
                F["nbd_rec_repos_vis"] = np.clip(np.fmin(v, vp), 0, 12)
                # ---- IDADE DA PEÇA (degrau: <15 d ~0,3% | 15-30 d ~6% | >=30 d ~17% de falha por dia da grade) ----
                # prev: desde a última troca PREVENTIVA (existe em out-dez, sem ambiguidade)
                # repos: desde a última reposição = max(troca preventiva, falha real/pseudo conhecida)
                h_prev = horas_desde_ultimo(pts, self.MANUT_PREV[self.MANUT_PREV["peca"] == p]) / 24
                h_fal = horas_desde_ultimo(pts, fal[fal["peca"] == p]) / 24
                h_rep = np.fmin(h_prev, h_fal)
                F["nbd_prev_idade_dias"] = np.clip(h_prev, 0, 400)
                F["nbd_repos_idade_dias"] = np.clip(h_rep, 0, 400)
                F["nbd_repos_faixa"] = np.where(np.isnan(h_rep), np.nan, np.where(h_rep < 15, 0.0, np.where(h_rep < 30, 1.0, 2.0)))
                F["nbd_prev_faixa"] = np.where(np.isnan(h_prev), np.nan, np.where(h_prev < 15, 0.0, np.where(h_prev < 30, 1.0, 2.0)))
                ev_p = ev_all[ev_all["peca"] == p]
                F["nbd_inst_idade"] = np.clip(horas_desde_ultimo(pts, ev_p) / 24, 0, 400)
                if self.usar_taxas and self.TEM_FORN:
                    fo = pd.Series(ultimo_valor(pts, ev_p, "fornecedor_peca", MUITO), index=keys.index).astype(object)
                    for f_ in self.FORNECEDORES:
                        F[f"nbd_inst_forn_{f_}"] = (fo == f_).astype(float)
                    F["nbd_inst_taxa_forn"] = fo.map(self.TAXA_FORN.get(p, {})).astype(float)
                if self.usar_taxas and self.TEM_TEC:
                    te = pd.Series(ultimo_valor(pts, ev_p, "tecnico", MUITO), index=keys.index).astype(object)
                    F["nbd_inst_taxa_tec"] = te.map(self.TAXA_TEC.get(p, {})).astype(float)
            out = pd.concat([keys[["id_maquina", "dia"]], F.astype("float32")], axis=1)
            out["peca"] = pd.Categorical([p] * len(out), categories=PECAS)
            partes.append(out)
        return pd.concat(partes, ignore_index=True)

    # ---------------------------------------------------------- pseudo-falhas pelos erros (para descongelar a recorrência)
    def registros_visita(self, ini, fim):
        dias = pd.date_range(pd.Timestamp(ini).normalize(), pd.Timestamp(fim).normalize(), freq="D")
        dn = dia_num(dias.to_numpy())
        regs = []
        for m in self.MAQUINAS:
            fase = self.FASE.get(m, np.nan)
            if np.isfinite(fase):
                regs += [(m, d + pd.Timedelta(hours=6)) for d in dias[(dn % self.PERIODO) == fase]]
        r = pd.DataFrame(regs, columns=["id_maquina", "data_hora"])
        r["data_hora"] = r["data_hora"].astype("datetime64[ns]")
        return r.sort_values("data_hora").reset_index(drop=True)

    def _calibrar_pseudo(self):
        regs0 = self.registros_visita(pd.Timestamp("2022-10-01"), self.corte_z - pd.Timedelta(days=1))
        if not len(regs0):
            self.CODIGOS_REG, self.K_PSEUDO = {p: [] for p in PECAS}, {}
            return
        Yreg = pd.DataFrame({p: contar_janela(regs0, self.falhas[self.falhas["peca"] == p], -0.5, 0.5) > 0 for p in PECAS})
        NCOD = pd.DataFrame({c: contar_janela(regs0, self.erros[self.erros["codigo_erro"] == c], -25, 0) for c in self.CODIGOS})
        self.CODIGOS_REG = {p: [] for p in PECAS}
        for c in self.CODIGOS:
            lift = {p: ((NCOD[c] > 0)[Yreg[p]].mean() + 1e-6) / ((NCOD[c] > 0)[~Yreg[p]].mean() + 1e-6) for p in PECAS}
            mp = max(lift, key=lift.get)
            if lift[mp] >= 2:
                self.CODIGOS_REG[mp].append(c)
        self.K_PSEUDO = {}
        for p in PECAS:
            n = NCOD[self.CODIGOS_REG[p]].sum(axis=1) if self.CODIGOS_REG[p] else pd.Series(0, index=regs0.index)
            ks = [k for k in (1, 2, 3, 4) if (n >= k).any() and Yreg[p][n >= k].mean() >= K_PRECISAO_MIN]
            self.K_PSEUDO[p] = min(ks) if ks else None
        self.log("pseudo-falhas: códigos", self.CODIGOS_REG, "| nº mínimo de erros", self.K_PSEUDO)

    def pseudo_por_erros(self, ini, fim):
        regs = self.registros_visita(ini, pd.Timestamp(fim) + pd.Timedelta(days=1))
        out = []
        for p in PECAS:
            if self.K_PSEUDO.get(p) is None or not self.CODIGOS_REG[p] or not len(regs):
                continue
            n = sum(contar_janela(regs, self.erros[self.erros["codigo_erro"] == c], -25, 0) for c in self.CODIGOS_REG[p])
            out.append(regs[n >= self.K_PSEUDO[p]].assign(peca=p))
        return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["id_maquina", "data_hora", "peca"])

