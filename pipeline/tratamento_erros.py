import sys
import pandas as pd
import pyarrow as pa


def limpa_erros(caminho_csv):
    df = pd.read_csv(caminho_csv)

    relatorio = {"linhas_entrada": len(df)}

    
    df["data_hora"] = pd.to_datetime(df["data_hora"], errors="coerce")
    relatorio["datas_invalidas"] = int(df["data_hora"].isna().sum())
    df = df.dropna(subset=["data_hora"]).copy()

    df["id_maquina"] = df["id_maquina"].str.strip().str.upper()
    df["versao_firmware"] = df["versao_firmware"].str.strip().str.lower()
   
    # ALTERADO: codigo_erro em várias colunas binárias (1 = é aquele código, 0 = não é)
    codigo = "E" + df["codigo_erro"].str.extract(r"(\d+)\s*$", expand=False)
    codigos_validos = ["E1", "E2", "E3", "E4", "E5"]
    relatorio["codigo_erro_invalido"] = int((~codigo.isin(codigos_validos)).sum())
    for c in codigos_validos:
        df[f"codigo_{c}"] = (codigo == c).astype(int)
    df = df.drop(columns="codigo_erro")
   
    # ALTERADO: severidade em 3 colunas binárias (1 = tem aquela severidade, 0 = não tem)
    sev = (df["severidade"].str.strip().str.lower()
           .str.normalize("NFKD").str.encode("ascii", "ignore").str.decode("ascii"))
    relatorio["severidade_invalida"] = int((~sev.isin(["baixa", "media", "alta"])).sum())
    df["severidade_baixa"] = (sev == "baixa").astype(int)
    df["severidade_media"] = (sev == "media").astype(int)
    df["severidade_alta"] = (sev == "alta").astype(int)
    df = df.drop(columns="severidade")

   
    df["tempo_reconhecimento_min"] = pd.to_numeric(df["tempo_reconhecimento_min"], errors="coerce")
    df.loc[df["tempo_reconhecimento_min"] <= 0, "tempo_reconhecimento_min"] = pd.NA
    relatorio["tempo_reconhecimento_nulo"] = int(df["tempo_reconhecimento_min"].isna().sum())
    df["tempo_reconhecimento_min"] = df["tempo_reconhecimento_min"].fillna(16)   

   
    antes = len(df)
    df = df.drop_duplicates()
    relatorio["duplicatas_removidas"] = antes - len(df)

    
    df = df.astype({
        "id_maquina": "category",
        "versao_firmware": "category",
        "codigo_E1": "Int8",   # ALTERADO: no lugar do dtype categórico de codigo_erro
        "codigo_E2": "Int8",
        "codigo_E3": "Int8",
        "codigo_E4": "Int8",
        "codigo_E5": "Int8",
        "severidade_baixa": "Int8",   # ALTERADO: no lugar do dtype categórico de severidade
        "severidade_media": "Int8",
        "severidade_alta": "Int8",
        "tempo_reconhecimento_min": "Int16",  
    })
    df = df.sort_values(["id_maquina", "data_hora"]).reset_index(drop=True)
    df["id_maquina"] = df["id_maquina"].str.strip().str.upper()
    df = df[df["versao_firmware"] != "v2.0.7"]
    df = df[df["id_maquina"].notna() & (df["id_maquina"] != "")]

    # O NB26 usa o código e a severidade de cada evento separadamente.
    df["codigo_erro"] = (
        df[[f"codigo_{c}" for c in codigos_validos]]
        .dot(range(1, len(codigos_validos) + 1))
        .map(dict(enumerate(codigos_validos, 1)))
    )
    severidades = ["baixa", "media", "alta"]
    df["severidade"] = (
        df[[f"severidade_{s}" for s in severidades]]
        .dot(range(1, len(severidades) + 1))
        .map(dict(enumerate(severidades, 1)))
    )
    relatorio["linhas_saida"] = len(df)
    return df, relatorio


if __name__ == "__main__":
    entrada = sys.argv[1] if len(sys.argv) > 1 else "erros.csv"
    
    saida = sys.argv[2] if len(sys.argv) > 2 else "errostratado.parquet"
    df, rel = limpa_erros(entrada)
    df.to_parquet(saida, index=False)
    print(rel)


print("FIM")