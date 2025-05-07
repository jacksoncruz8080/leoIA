import os
from openai import OpenAI
import chromadb
from typing import List

# === CONFIGURAÇÕES ===
OPENAI_API_KEY = "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"  # Substitua por sua chave da OpenAI
ARQUIVO_TRANSCRICAO = "transcricoes.txt"
N_RESULTADOS = 4

# === CLIENTES E COLLECTION ===
client = OpenAI(api_key=OPENAI_API_KEY)
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="conversas")

# === FUNÇÕES AUXILIARES ===

def dividir_texto(texto: str, tamanho_maximo: int = 500) -> List[str]:
    palavras = texto.split()
    blocos, bloco = [], []
    for palavra in palavras:
        bloco.append(palavra)
        if len(bloco) >= tamanho_maximo:
            blocos.append(" ".join(bloco))
            bloco = []
    if bloco:
        blocos.append(" ".join(bloco))
    return blocos

def gerar_embedding(texto: str) -> List[float]:
    response = client.embeddings.create(
        input=texto,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def indexar_transcricoes():
    if not os.path.exists(ARQUIVO_TRANSCRICAO):
        print(f"Arquivo '{ARQUIVO_TRANSCRICAO}' não encontrado.")
        return
    with open(ARQUIVO_TRANSCRICAO, "r", encoding="utf-8") as f:
        texto = f.read()
    blocos = dividir_texto(texto)
    for i, bloco in enumerate(blocos):
        id_bloco = f"bloco_{i}"
        try:
            collection.add(
                documents=[bloco],
                embeddings=[gerar_embedding(bloco)],
                ids=[id_bloco]
            )
            print(f"Indexado: {id_bloco}")
        except chromadb.errors.IDAlreadyExistsError:
            print(f"Já indexado: {id_bloco}")

def buscar_contexto(pergunta: str, n: int = N_RESULTADOS) -> List[str]:
    emb = gerar_embedding(pergunta)
    resultado = collection.query(query_embeddings=[emb], n_results=n)
    return resultado["documents"][0]

def responder(pergunta: str):
    contexto = buscar_contexto(pergunta)
    prompt = f"""
Você é um assistente virtual que responde exclusivamente com base no conteúdo abaixo, que representa transcrições reais feitas por um especialista.
Você não deve adicionar conhecimento próprio, inventar dados ou sair do contexto. Se não souber a resposta com base nesses textos, apenas diga:

"Desculpe, não tenho informações suficientes para responder com base nas informações disponíveis."

Base de conhecimento:
{chr(10).join(contexto)}

Pergunta: {pergunta}
Resposta:"""
    
    resposta = client.chat.completions.create(
        model="gpt-3.5-turbo",  # gpt-4 ou gpt-3.5-turbo
        messages=[
            {"role": "system", "content": "Responda estritamente com base no conteúdo fornecido. Não invente nem use conhecimento próprio."},
            {"role": "user", "content": prompt}
        ],
        temperature=0  # zero criatividade, máximo foco no conteúdo fornecido
    )
    # print("\nResposta do assistente.:")
    # print(resposta.choices[0].message.content)
    return resposta

# === FLUXO PRINCIPAL ===
if __name__ == "__main__":
    print("Indexando transcrições...")
    indexar_transcricoes()
    print("\nDigite sua pergunta (ou 'sair'):")
    while True:
        pergunta = input("> ").strip()
        if pergunta.lower() in ["sair", "exit", "quit"]:
            break
        responder(pergunta)
