"""Servidor local do painel e da fila de treinamento."""
import asyncio
import json
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import uvicorn
from starlette.applications import Starlette
from starlette.datastructures import UploadFile
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response
from starlette.routing import Route

from pipeline.executar import ARQUIVOS, executar

RAIZ = Path(__file__).resolve().parent
RUNTIME = RAIZ / ".mabiyama"
RESULTADO = RUNTIME / "resultado.json"
LIMITE = 128 * 1024 * 1024
trabalho = None
tarefa = None


def erro(mensagem, status):
    return JSONResponse({"erro": mensagem}, status_code=status)


async def status(request):
    return JSONResponse({"trabalho": trabalho}, headers={"Cache-Control": "no-store"})


async def resultado(request):
    try:
        dados = await asyncio.to_thread(RESULTADO.read_text, encoding="utf-8")
    except FileNotFoundError:
        return erro("Ainda não há uma submission treinada.", 404)
    if request.url.path.endswith(".csv"):
        return Response(
            json.loads(dados)["csv"], media_type="text/csv",
            headers={"Content-Disposition": 'attachment; filename="submission.csv"'},
        )
    return Response(dados, media_type="application/json", headers={"Cache-Control": "no-store"})


async def iniciar(request: Request):
    global trabalho, tarefa
    if request.headers.get("x-mabiyama") != "treinamento" or request.headers.get("sec-fetch-site") == "cross-site":
        return erro("Envie os CSVs pela aba Treinamento.", 403)
    if trabalho and trabalho["estado"] in ("recebendo", "executando"):
        return erro("Já existe um treinamento em andamento.", 409)
    tamanho = request.headers.get("content-length", "")
    if not tamanho.isdecimal() or not 0 < int(tamanho) <= LIMITE:
        return erro("O envio deve ter no máximo 128 MB.", 413)

    RUNTIME.mkdir(exist_ok=True)
    pasta = TemporaryDirectory(prefix="treino-", dir=RUNTIME)
    execucao = {"id": uuid4().hex, "estado": "recebendo", "etapa": 0}
    trabalho = execucao
    try:
        async with request.form(max_files=len(ARQUIVOS), max_fields=0) as campos:
            for nome in ARQUIVOS:
                arquivo_csv = campos.get(nome)
                if not isinstance(arquivo_csv, UploadFile) or not arquivo_csv.size:
                    raise ValueError(f"Selecione o arquivo {nome}.csv.")
                if not (arquivo_csv.filename or "").lower().endswith(".csv"):
                    raise ValueError(f"{nome} precisa ser um arquivo CSV.")
                with (Path(pasta.name) / f"{nome}.csv").open("wb") as destino:
                    while bloco := await arquivo_csv.read(1024 * 1024):
                        destino.write(bloco)
    except Exception as exc:
        pasta.cleanup()
        mensagem = str(exc) if isinstance(exc, ValueError) else "Não foi possível receber os seis CSVs."
        execucao.update(estado="erro", erro=mensagem)
        return erro(mensagem, 400)

    execucao["estado"] = "executando"
    tarefa = asyncio.create_task(processar(pasta, execucao))
    return JSONResponse({"trabalho": execucao}, status_code=202)


async def processar(pasta, execucao):
    loop = asyncio.get_running_loop()

    def etapa(numero):
        loop.call_soon_threadsafe(execucao.update, {"etapa": numero})

    def publicar():
        dados = executar(Path(pasta.name), etapa)
        dados["id"] = execucao["id"]
        pendente = Path(pasta.name) / "resultado.json"
        pendente.write_text(json.dumps(dados, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        os.replace(pendente, RESULTADO)

    try:
        await asyncio.to_thread(publicar)
        execucao.update(estado="concluido")
    except Exception as exc:
        print(exc, file=sys.stderr, flush=True)
        execucao.update(estado="erro", erro=str(exc).splitlines()[0])
    finally:
        pasta.cleanup()


app = Starlette(routes=[
    Route("/", lambda request: FileResponse(RAIZ / "index.html")),
    Route("/index.html", lambda request: FileResponse(RAIZ / "index.html")),
    Route("/treinamento.js", lambda request: FileResponse(RAIZ / "treinamento.js")),
    Route("/api/treinamento", iniciar, methods=["POST"]),
    Route("/api/treinamento", status),
    Route("/api/treinamento/resultado", resultado),
    Route("/api/treinamento/submission.csv", resultado),
])


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("MABIYAMA_PORTA", "8000")), timeout_graceful_shutdown=3600)
