import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from langchain.vectorstores.faiss import FAISS
from langchain.chains import RetrievalQA

# Carrega variáveis de ambiente do arquivo .env (opcional, mas recomendado)
load_dotenv()

# --- 1. CONFIGURAÇÃO INICIAL ---
# Certifique-se que sua GOOGLE_API_KEY está no arquivo .env ou como variável de ambiente
if "GOOGLE_API_KEY" not in os.environ:
    print("Erro: A variável de ambiente GOOGLE_API_KEY não foi encontrada.")
    print("Por favor, crie um arquivo .env e adicione a linha: GOOGLE_API_KEY='sua-chave-aqui'")
    exit()

# ### MUDANÇA ###: Atualizando o caminho para a sua pasta
DIRETORIO_BASE_CONHECIMENTO = "./base_de_conhecimento"

print("Iniciando o assistente de TI com Gemini...")

# --- 2. CARREGAMENTO DOS DOCUMENTOS ---
if not os.path.isdir(DIRETORIO_BASE_CONHECIMENTO):
    print(f"Erro: O diretório '{DIRETORIO_BASE_CONHECIMENTO}' não foi encontrado.")
    print("Por favor, crie a pasta e adicione seus arquivos de conhecimento.")
    exit()

print(f"Carregando documentos de '{DIRETORIO_BASE_CONHECIMENTO}'...")
loader = DirectoryLoader(DIRETORIO_BASE_CONHECIMENTO, glob="**/*.*", show_progress=True)
documentos = loader.load()

if not documentos:
    print("Nenhum documento encontrado na pasta. O assistente não terá conhecimento específico.")
    # Podemos decidir encerrar ou continuar com o conhecimento geral do modelo.
    # Por segurança, vamos encerrar para garantir que ele use apenas a base.
    exit()


print(f"{len(documentos)} documentos carregados com sucesso.")

# --- 3. QUEBRA DO TEXTO EM PEDAÇOS (CHUNKS) ---
print("Dividindo os documentos em trechos (chunks)...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
textos_divididos = text_splitter.split_documents(documentos)
print(f"Documentos divididos em {len(textos_divididos)} trechos.")

# --- 4. CRIAÇÃO DOS EMBEDDINGS E DA VECTOR STORE ---
print("Criando embeddings do Google e a base de vetores (Vector Store)...")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vector_store = FAISS.from_documents(textos_divididos, embeddings)
print("Base de vetores criada com sucesso.")

# --- 5. CRIAÇÃO DA CADEIA DE PERGUNTAS E RESPOSTAS (QA CHAIN) ---
print("Configurando a cadeia de Perguntas e Respostas (QA Chain) com Gemini...")
llm = GoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(),
    return_source_documents=True
)

print("\n✅ Assistente de Suporte TI (Gemini) pronto! Faça sua pergunta ou digite 'sair' para encerrar.")
print("-" * 50)

# --- 6. LOOP INTERATIVO ---
while True:
    pergunta = input("Sua pergunta: ")
    if pergunta.lower() == 'sair':
        print("Até logo!")
        break
    
    if not pergunta.strip():
        continue
    
    print("\nBuscando na base de conhecimento...")
    resultado = qa_chain.invoke({"query": pergunta})
    
    print("\nResposta do Assistente:")
    print(resultado["result"])
    
    print("-" * 50)