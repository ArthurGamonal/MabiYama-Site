"""Orquestra os scripts do projeto em uma pasta isolada por envio."""
import csv
import os
import re
import subprocess
import sys
from datetime import date
from pathlib import Path
from shutil import copyfile

ARQUIVOS = ("telemetria", "maquinas", "operacao", "falhas", "manutencao", "erros")
SCRIPTS = Path(__file__).resolve().parent
PECAS = ("peca_A", "peca_B", "peca_C", "peca_D")


def preparar_falhas(pasta):
    import pandas as pd

    falhas = pd.read_csv(pasta / "falhas.csv", dtype=str)
    falhas = falhas[["data_hora", "id_maquina", "peca"]].copy()
    falhas["data_hora"] = pd.to_datetime(falhas["data_hora"], format="mixed", dayfirst=True, errors="coerce")
    falhas["id_maquina"] = falhas["id_maquina"].str.strip().str.upper()
    letra = (falhas["peca"].str.normalize("NFKD").str.encode("ascii", "ignore")
             .str.decode("ascii").str.lower().str.extract(r"^\s*peca[ _-]*([a-d])\s*$", expand=False))
    falhas["peca"] = "peca_" + letra.str.upper()
    falhas = falhas.dropna().loc[lambda df: df["id_maquina"].str.fullmatch(r"M\d{3,}")].drop_duplicates()
    if falhas.empty:
        raise ValueError("falhas.csv não tem falhas válidas para treinamento.")
    falhas.to_parquet(pasta / "falhas.parquet", index=False)


def ler_resultado(pasta, saida):
    import pandas as pd

    caminho = saida / "submission.csv"
    sub = pd.read_csv(caminho)
    if list(sub) != ["id", *PECAS] or sub.empty or sub["id"].duplicated().any():
        raise ValueError("O treinamento não gerou uma submission válida.")
    if not sub[list(PECAS)].apply(lambda p: p.between(0, 1)).all().all():
        raise ValueError("A submission contém probabilidades inválidas.")
    ids = sub["id"].str.extract(r"^(M\d{3,})_(\d{4}-\d{2}-\d{2})$")
    if ids.isna().any().any():
        raise ValueError("A submission contém identificadores de máquinas inválidos.")
    inicio = date.fromisoformat(ids[1].min())
    dias = (date.fromisoformat(ids[1].max()) - inicio).days + 1
    cadastro = pd.read_parquet(pasta / "maquinas_tratado.parquet").set_index("id_maquina")
    maquinas = []
    for nome in sorted(ids[0].unique(), key=lambda nome: int(nome[1:])):
        cad = cadastro.loc[nome] if nome in cadastro.index else {}
        modelo = str(cad.get("modelo", "")).upper()
        maquinas.append({
            "nome": nome,
            "modelo": "modelo " + modelo[-1] if re.fullmatch(r"MODELO[_ ][A-D]", modelo) else "modelo A",
            "linha": cad.get("linha_producao") if pd.notna(cad.get("linha_producao")) else None,
            "serie": cad.get("numero_serie") if pd.notna(cad.get("numero_serie")) else None,
            "observacoes": cad.get("observacoes") if pd.notna(cad.get("observacoes")) else None,
        })
    if len(sub) != len(maquinas) * dias:
        raise ValueError("A submission não cobre todos os dias do período para cada máquina.")
    posicoes = {m["nome"]: i for i, m in enumerate(maquinas, 1)}
    manut = {}
    for troca in pd.read_parquet(pasta / "manutencao.parquet").to_dict("records"):
        posicao = posicoes.get(troca["id_maquina"])
        if posicao is None:
            continue
        tecnico = next((c.removeprefix("tecnico_") for c, v in troca.items() if c.startswith("tecnico_") and v == 1), "—")
        for peca in PECAS:
            if troca.get(peca) != 1:
                continue
            dia = (troca["data_hora"].date() - inicio).days
            ficha = [dia, str(troca["lote_peca"]), str(troca["fornecedor_peca"]), tecnico,
                     int(troca["duracao_min"]) if pd.notna(troca["duracao_min"]) else None,
                     float(troca["custo_troca_reais"]) if pd.notna(troca["custo_troca_reais"]) else None]
            manut.setdefault(posicao, {}).setdefault(peca[-1], []).append(ficha)
    return {"csv": caminho.read_text(encoding="utf-8"), "maquinas": maquinas, "manutencao": manut}


def executar(pasta, informar):
    saida = pasta / "saida"
    saida.mkdir()
    env = dict(os.environ, PYTHONUTF8="1", MABIYAMA_DADOS=str(pasta), MABIYAMA_SAIDA=str(saida))

    def rodar(script, *args):
        comando = [sys.executable, "-u", str(SCRIPTS / script), *map(str, args)]
        processo = subprocess.run(comando, cwd=pasta, env=env, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        print(processo.stdout, end="", flush=True)
        if processo.returncode:
            linhas = processo.stdout.strip().splitlines()
            raise RuntimeError(f"{script}: {linhas[-1] if linhas else 'o script foi interrompido'}")

    informar(1)
    rodar("tratamento_telemetria_maquina.py", pasta, pasta)
    rodar("tratamento_operacao.py", "operacao.csv", "operacaoTratado.parquet")
    rodar("tratamento_erros.py", "erros.csv", "errostratado.parquet")
    rodar("TratamentoManutencao.py", "manutencao.csv", "manutencao.parquet")
    preparar_falhas(pasta)
    # O sample do desafio só ordena a submission quando pertence à frota enviada.
    with (SCRIPTS / "sample_submission.csv").open(encoding="utf-8-sig", newline="") as modelo:
        frota_modelo = {linha["id"].rsplit("_", 1)[0] for linha in csv.DictReader(modelo)}
    import pandas as pd
    frota = set(pd.read_parquet(pasta / "telemetria_tratada.parquet", columns=["id_maquina"])["id_maquina"])
    if frota == frota_modelo:
        copyfile(SCRIPTS / "sample_submission.csv", pasta / "sample_submission.csv")

    informar(2)
    rodar("zinal.py", pasta)
    informar(3)
    rodar("criar_base2.py", pasta)
    informar(4)
    rodar("features.py")
    informar(5)
    rodar("treinar.py")
    informar(6)
    return ler_resultado(pasta, saida)
