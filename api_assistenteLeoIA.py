##instalar dependencias##
# pip install fastapi uvicorn

from fastapi import FastAPI
from pydantic import BaseModel
from AssitentLeoIA import responder, indexar_transcricoes

app = FastAPI()

class Pergunta(BaseModel):
    texto: str

# Indexa transcrições ao iniciar o servidor
@app.on_event("startup")
def iniciar_indexacao():
    print("Indexando transcrições ao iniciar o servidor...")
    indexar_transcricoes()    

@app.post("/perguntar")
def perguntar(p: Pergunta):
    resposta = responder(p.texto)
    return resposta

@app.get("/teste")
def teste():
    return 'ok'

###iniciar sevidor##
# uvicorn api_assistenteLeoIA:app --host 0.0.0.0 --port 49800

###requisição###
#curl -X POST http://localhost:49800/perguntar -H "Content-Type: application/json" -d '{"texto": "Qual é o procedimento para marcar consulta?"}'
