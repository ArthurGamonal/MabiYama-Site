#Junta telemetria + máquinas + operação + erros em UM parquet.

#Granularidade final: 1 linha por leitura de telemetria (máquina x hora).

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ARQ_TELE = "telemetria_tratada.parquet"
ARQ_MAQ = "maquinas_tratado.parquet"
ARQ_OPER = "operacaoTratado.parquet"
ARQ_ERROS = "errostratado.parquet"
SAIDA = "base_completa.parquet"

# ASSUMIÇÃO A CONFIRMAR: horário dos turnos (a tabela de operação só diz manha/tarde/noite)
#   manha = 06h-14h, tarde = 14h-22h, noite = 22h-06h (a noite pertence ao dia em que começou)
INI_MANHA, INI_TARDE, INI_NOITE = 6, 14, 22

SEVERIDADES = ["baixa", "media", "alta"]


#auxilia
def ns(s):
    """Garante datetime64[ns] (evita erro de tipos diferentes nos merges)."""
    return s.astype("datetime64[ns]")


def norm_id(s):
    return s.astype(str).str.strip().str.upper()


def atribuir_turno(ts):
    """Devolve (turno, data_do_turno) para cada timestamp."""
    h = ts.dt.hour
    turno = np.select(
        [(h >= INI_MANHA) & (h < INI_TARDE), (h >= INI_TARDE) & (h < INI_NOITE)],
        ["manha", "tarde"],
        default="noite",
    )
    # 00h-05h pertencem à noite do dia anterior
    data_turno = ns((ts - pd.Timedelta(hours=INI_MANHA)).dt.normalize())
    return pd.Series(turno, index=ts.index), data_turno


def carregar_operacao(caminho):
    op = pd.read_parquet(caminho)
    op["id_maquina"] = norm_id(op["id_maquina"])
    op["turno"] = op["turno"].astype(str).str.strip().str.lower()
    op["hora_extra"] = op["hora_extra"].astype("boolean")
    for c in ["operador", "produto_fabricado", "classe_risco"]:
        op[c] = op[c].astype("string")
    op = op.rename(columns={"data": "data_turno"})
    op["data_turno"] = ns(op["data_turno"])
    op["_tem_operacao"] = True
    chave = ["id_maquina", "data_turno", "turno"]
    assert not op.duplicated(chave).any(), "operação tem chave (maquina, data, turno) repetida"
    return op


def carregar_erros(caminho):
    er = pd.read_parquet(caminho)
    codigos = [f"E{i}" for i in range(1, 6)]
    er["codigo_erro"] = er[[f"codigo_{c}" for c in codigos]].dot(np.arange(1, 6)).map(dict(enumerate(codigos, 1)))
    er["severidade"] = (er[[f"severidade_{s}" for s in SEVERIDADES]]
                        .dot(np.arange(1, 4)).map(dict(enumerate(SEVERIDADES, 1))))
    er["id_maquina"] = norm_id(er["id_maquina"])
    er["data_hora"] = ns(er["data_hora"])
    er["severidade"] = er["severidade"].astype(str).str.strip().str.lower()
    er = er.sort_values("data_hora", kind="stable")

    # agregado por máquina x hora (vários erros na mesma hora viram uma linha)
    er["_hora"] = er["data_hora"].dt.floor("h")
    er["_sev"] = er["severidade"].map({s: i + 1 for i, s in enumerate(SEVERIDADES)})
    ag = er.groupby(["id_maquina", "_hora"]).agg(
        n_erros_hora=("codigo_erro", "size"),
        _sev=("_sev", "max"),
        tempo_reconhecimento_medio_min=("tempo_reconhecimento_min", "mean"),
        codigo_erro_ultimo=("codigo_erro", "last"),
    ).reset_index()
    ag["severidade_max_hora"] = pd.Categorical(
        ag["_sev"].map({i + 1: s for i, s in enumerate(SEVERIDADES)}),
        categories=SEVERIDADES, ordered=True,
    )
    ag = ag.drop(columns="_sev")

    # firmware: última versão conhecida de cada máquina (vem do evento de erro)
    fw = (er[["id_maquina", "data_hora", "versao_firmware"]]
          .dropna(subset=["versao_firmware"])
          .rename(columns={"data_hora": "_t_fw", "versao_firmware": "versao_firmware_ultima"})
          .sort_values("_t_fw"))
    return ag, fw


def checar_sobreposicao(base, outra, chaves, nome):
    repetidas = (set(base.columns) & set(outra.columns)) - set(chaves)
    assert not repetidas, f"colunas repetidas ao juntar {nome}: {sorted(repetidas)}"


# ======================= EXECUÇÃO =======================
def main():
    pasta = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent

    tele = pd.read_parquet(pasta / ARQ_TELE)
    maq = pd.read_parquet(pasta / ARQ_MAQ)
    op = carregar_operacao(pasta / ARQ_OPER)
    erros_hora, fw = carregar_erros(pasta / ARQ_ERROS)

    tele = tele.copy()
    tele["id_maquina"] = norm_id(tele["id_maquina"])
    tele["data_hora"] = ns(tele["data_hora"])
    maq = maq.assign(id_maquina=norm_id(maq["id_maquina"]))

    n = len(tele)
    cols_tele = list(tele.columns)
    cols_maq = [c for c in maq.columns if c != "id_maquina"]
    cols_op = [c for c in op.columns if c not in ("id_maquina", "data_turno", "turno", "_tem_operacao")]
    cols_err = ["n_erros_hora", "severidade_max_hora", "tempo_reconhecimento_medio_min",
                "codigo_erro_ultimo", "versao_firmware_ultima"]

    # --- turno de cada leitura ---
    tele["turno"], tele["data_turno"] = atribuir_turno(tele["data_hora"])
    tele["_hora"] = tele["data_hora"].dt.floor("h")

    # --- firmware (última versão conhecida até aquele momento) ---
    tele = tele.sort_values("data_hora")
    tele = pd.merge_asof(tele, fw, left_on="data_hora", right_on="_t_fw",
                         by="id_maquina", direction="backward").drop(columns="_t_fw")

    # --- máquinas ---
    checar_sobreposicao(tele, maq, ["id_maquina"], "máquinas")
    df = tele.merge(maq, on="id_maquina", how="left", validate="many_to_one")

    # --- operação ---
    checar_sobreposicao(df, op, ["id_maquina", "data_turno", "turno", "_tem_operacao"], "operação")
    df = df.merge(op, on=["id_maquina", "data_turno", "turno"], how="left", validate="many_to_one")
    df["_tem_operacao"] = df["_tem_operacao"].eq(True)

    # --- erros por hora ---
    df = df.merge(erros_hora, on=["id_maquina", "_hora"], how="left", validate="many_to_one")
    df["n_erros_hora"] = df["n_erros_hora"].fillna(0).astype("int32")

    assert len(df) == n, "o join mudou o número de linhas"

    # --- checagens ---
    print("Linhas:", len(df))
    print("Leituras sem cadastro de máquina:", int(df["modelo"].isna().sum()), "(modelo nulo)")
    print("\nLeituras com dados de operação, por turno (% casadas):")
    print((df.groupby("turno")["_tem_operacao"].mean() * 100).round(1).to_string())
    print("\nLeituras com pelo menos 1 erro na hora: %.2f%%" % (100 * (df["n_erros_hora"] > 0).mean()))
    print("Leituras sem firmware conhecido: %.2f%%" % (100 * df["versao_firmware_ultima"].isna().mean()))

    # --- ordem das colunas e salvar ---
    ordem = list(dict.fromkeys(
        ["data_hora", "id_maquina", "turno", "data_turno"]
        + [c for c in cols_tele if c not in ("data_hora", "id_maquina")]
        + cols_maq + cols_op + cols_err
    ))
    df = df[ordem].sort_values(["id_maquina", "data_hora"]).reset_index(drop=True)
    df.to_parquet(pasta / SAIDA, index=False)
    print(f"\nSalvo: {pasta / SAIDA}  {df.shape}")


if __name__ == "__main__":
    main()