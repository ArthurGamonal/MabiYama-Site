import sys
import pandas as pd

COLUNAS = ["data_hora", "id_maquina", "peca", "tecnico", "duracao_min",
           "fornecedor_peca", "lote_peca", "custo_troca_reais"]


def sem_acento(coluna):
    """Minúsculo, sem espaços nas pontas e sem acento: 'Peça D' -> 'peca d'."""
    return (coluna.str.strip().str.lower()
            .str.normalize("NFKD").str.encode("ascii", "ignore").str.decode("ascii"))


def limpa_manutencao(caminho_csv):
    # dtype=str: lê tudo como texto, para tratar cada coluna com cuidado
    df = pd.read_csv(caminho_csv, dtype=str)

    # 1. valida colunas
    faltando = [c for c in COLUNAS if c not in df.columns]
    if faltando:
        raise ValueError(f"Colunas ausentes: {faltando}")

    relatorio = {"linhas_entrada": len(df)}
    df = df.dropna(how="all")   # remove linhas totalmente vazias

    # 2. datas
    df["data_hora"] = pd.to_datetime(df["data_hora"], errors="coerce")
    relatorio["datas_invalidas"] = int(df["data_hora"].isna().sum())
    df = df.dropna(subset=["data_hora"]).copy()

    # 3. textos
    df["id_maquina"] = df["id_maquina"].str.strip().str.upper()
    df = df[df["id_maquina"].notna() & (df["id_maquina"] != "")]   # remove linhas sem id de máquina
    df["tecnico"] = df["tecnico"].str.strip().str.upper()
    df["lote_peca"] = df["lote_peca"].str.strip().str.upper()

    # 'peca_A', 'PECA_A', 'peca a', 'Peça A' -> 'peca_A'
    letra = sem_acento(df["peca"]).str.extract(r"([a-z])\s*$", expand=False).str.upper()
    df["peca"] = "peca_" + letra

    # 'Metal Parts', 'METALPARTS', 'Rolatec Ltda', 'usimec' -> nome padrão
    forn = sem_acento(df["fornecedor_peca"]).str.replace("ltda", "", regex=False).str.replace(" ", "", regex=False)
    padrao = {"metalparts": "MetalParts", "rolatec": "Rolatec", "usimec": "Usimec"}
    df["fornecedor_peca"] = forn.map(padrao).fillna(forn)   # fornecedor novo é mantido, só contado abaixo
    relatorio["fornecedores_fora_do_padrao"] = int((~forn.isin(padrao)).sum())

    # 4. números (vírgula decimal vira ponto; inválido vira nulo)
    for c in ["duracao_min", "custo_troca_reais"]:
        df[c] = pd.to_numeric(df[c].str.strip().str.replace(",", ".", regex=False), errors="coerce")

    # 5. valores impossíveis viram nulo (nulos são mantidos, só contados)
    for c in ["duracao_min", "custo_troca_reais"]:
        mascara = df[c] <= 0
        relatorio[f"impossiveis_{c}"] = int(mascara.sum())
        df.loc[mascara, c] = pd.NA
        relatorio[f"{c}_nulo"] = int(df[c].isna().sum())

    # 6. duplicatas: linhas iguais e chave repetida (data_hora + máquina + peça)
    antes = len(df)
    df = df.drop_duplicates()
    relatorio["duplicatas_exatas_removidas"] = antes - len(df)
    antes = len(df)
    df = df.drop_duplicates(subset=["data_hora", "id_maquina", "peca"], keep="first")
    relatorio["chave_repetida_removida"] = antes - len(df)

    # 7. técnico e peça em colunas binárias (1 = é aquele valor, 0 = não é)   # ALTERADO
    tecnicos = pd.get_dummies(df["tecnico"], prefix="tecnico", dtype="int8")   # tecnico_T01, tecnico_T02...
    pecas = pd.get_dummies(df["peca"], dtype="int8")                           # peca_A, peca_B... (já têm o prefixo)
    df = pd.concat([df.drop(columns=["tecnico", "peca"]), tecnicos, pecas], axis=1)

    # 8. tipos finais e ordenação
    df = df.astype({
        "id_maquina": "category",
        "fornecedor_peca": "category",   # ALTERADO (peca e tecnico não existem mais como coluna única)
        "duracao_min": "Int16",
        "custo_troca_reais": "float64",
    })
    df = df.sort_values(["id_maquina", "data_hora"]).reset_index(drop=True)
    relatorio["linhas_saida"] = len(df)
    return df, relatorio


if __name__ == "__main__":
    entrada = sys.argv[1] if len(sys.argv) > 1 else "manutencao.csv"
    saida = sys.argv[2] if len(sys.argv) > 2 else "manutencao.parquet"
    df, rel = limpa_manutencao(entrada)
    df.to_parquet(saida, index=False, compression="snappy")   # grava em parquet (precisa do pyarrow)
    print(rel)