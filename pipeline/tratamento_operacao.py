
import sys
import pandas as pd




def sem_acento(coluna):
    
    return (coluna.str.strip().str.lower()
            .str.normalize("NFKD").str.encode("ascii", "ignore").str.decode("ascii"))


def limpa_operacao(caminho_csv):
   
    df = pd.read_csv(caminho_csv, dtype=str)

   


    relatorio = {"linhas_entrada": len(df)}

    
    df["data"] = pd.to_datetime(df["data"], errors="coerce")
    relatorio["datas_invalidas"] = int(df["data"].isna().sum())
    df = df.dropna(subset=["data"]).copy()

    
    df["id_maquina"] = df["id_maquina"].str.strip().str.upper()
    df["operador"] = df["operador"].str.strip().str.upper()
    df["produto_fabricado"] = df["produto_fabricado"].str.strip().str.upper()
    df["turno"] = sem_acento(df["turno"])
    df["hora_extra"] = sem_acento(df["hora_extra"])      
    df["classe_risco"] = sem_acento(df["classe_risco"])  

   
    numericas = ["pecas_produzidas", "meta_producao", "pecas_defeituosas", "qtd_operadores",
                 "consumo_energia_kwh", "utilizacao_pct", "tempo_setup_min",
                 "ind_quebra", "score_saude"]
    for c in numericas:
        df[c] = pd.to_numeric(df[c].str.strip().str.replace(",", ".", regex=False), errors="coerce")

   
    impossivel = {
        "pecas_defeituosas": df["pecas_defeituosas"] > df["pecas_produzidas"],
        "utilizacao_pct": (df["utilizacao_pct"] < 0) | (df["utilizacao_pct"] > 100),
        "consumo_energia_kwh": df["consumo_energia_kwh"] < 0,
        "pecas_produzidas": df["pecas_produzidas"] < 0,
        "meta_producao": df["meta_producao"] <= 0,
    }
    for coluna, mascara in impossivel.items():
        relatorio[f"impossiveis_{coluna}"] = int(mascara.sum())
        df.loc[mascara, coluna] = pd.NA

    
    relatorio["meta_producao_nula"] = int(df["meta_producao"].isna().sum())
    df["mes"] = df["data"].dt.to_period("M")
    meta_do_mes = df.groupby(["id_maquina", "mes"])["meta_producao"].transform("first")
    df["meta_producao"] = df["meta_producao"].fillna(meta_do_mes)
    df = df.drop(columns="mes")
    relatorio["meta_producao_nula_restante"] = int(df["meta_producao"].isna().sum())

    
    antes = len(df)
    df = df.drop_duplicates()
    relatorio["duplicatas_exatas_removidas"] = antes - len(df)
    antes = len(df)
    df = df.drop_duplicates(subset=["id_maquina", "data", "turno"], keep="first")
    relatorio["chave_repetida_removida"] = antes - len(df)


    df = df.astype({
        "id_maquina": "category",
        "operador": "category",
        "produto_fabricado": "category",
        "turno": pd.CategoricalDtype(["manha", "tarde", "noite"], ordered=True),
        "hora_extra": "category",
        "classe_risco": pd.CategoricalDtype(["baixo", "medio", "alto"], ordered=True),
        "pecas_produzidas": "Int32",
        "meta_producao": "Int32",
        "pecas_defeituosas": "Int32",
        "qtd_operadores": "Int8",
        "tempo_setup_min": "Int16",
    })
    df["hora_extra"] = df["hora_extra"].map({"sim": True, "nao": False}).astype("boolean")
    df = df.sort_values(["id_maquina", "data", "turno"]).reset_index(drop=True)
    df["id_maquina"] = df["id_maquina"].str.strip().str.upper()
    df = df[df["id_maquina"].notna() & (df["id_maquina"] != "")]
    relatorio["linhas_saida"] = len(df)
    return df, relatorio


if __name__ == "__main__":
    entrada = sys.argv[1] if len(sys.argv) > 1 else "operacao.csv"
    saida = sys.argv[2] if len(sys.argv) > 2 else "operacaoTratado.parquet"
    df, rel = limpa_operacao(entrada)
    df.to_parquet(saida, index=False)
    print(rel)



print("FIM")