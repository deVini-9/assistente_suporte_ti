import streamlit as st
import os
import glob
from dotenv import load_dotenv

# Importações do LangChain
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from langchain.chains import RetrievalQA

# Carrega variáveis de ambiente do arquivo .env
load_dotenv()

# --- CONFIGURAÇÃO DA PÁGINA STREAMLIT ---
st.set_page_config(page_title="Assistente de Suporte TI", page_icon="🤖")
st.title("🤖 Assistente de Suporte de TI")
st.caption("Eu sou um assistente baseado nos documentos da base de conhecimento.")

# --- LÓGICA DE CACHE PARA CARREGAR RECURSOS PESADOS ---


@st.cache_resource
def carregar_recursos():
    DIRETORIO_BASE_CONHECIMENTO = "./base_de_conhecimento"

    # Verificação da API Key do Google
    if "GOOGLE_API_KEY" not in os.environ:
        st.error(
            "A variável de ambiente GOOGLE_API_KEY não foi encontrada. Configure-a no arquivo .env.")
        return None, None

    # Verifica se o diretório existe
    if not os.path.isdir(DIRETORIO_BASE_CONHECIMENTO):
        st.error(
            f"O diretório '{DIRETORIO_BASE_CONHECIMENTO}' não foi encontrado.")
        return None, None

    with st.spinner("Analisando e carregando a base de conhecimento..."):
        # Define o loader para encontrar os arquivos
        directory_loader = DirectoryLoader(
            DIRETORIO_BASE_CONHECIMENTO,
            glob="**/*.{txt,pdf,docx,md,sql,csv,doc,png,pptx,xlsx}",
            show_progress=True,
            use_multithreading=True,
        )

        # === MELHORIA: Carregamento robusto de documentos ===
        # Em vez de carregar tudo de uma vez, carregamos um por um
        # para identificar exatamente qual arquivo pode estar com problema.
        documentos = []
        arquivos_com_erro = []

        # Usamos lazy_load() que prepara os arquivos sem carregar na memória ainda
        for loader in directory_loader.lazy_load():
            try:
                # Tentamos carregar o conteúdo do arquivo
                documentos.extend(loader)
            except Exception as e:
                # Se falhar, guardamos o nome do arquivo e o erro
                nome_arquivo = os.path.basename(loader.file_path)
                arquivos_com_erro.append(
                    f"Arquivo: '{nome_arquivo}' - Erro: {e}")

        # Mostra avisos na tela para cada arquivo que falhou
        if arquivos_com_erro:
            st.warning("Alguns arquivos não puderam ser carregados:")
            for erro in arquivos_com_erro:
                st.write(erro)

        if not documentos:
            st.error(
                "Nenhum documento válido foi lido com sucesso na base de conhecimento.")
            return None, None

        # Quebra dos textos em pedaços
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200)
        textos_divididos = text_splitter.split_documents(documentos)

        # Criação dos embeddings e da vector store
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_documents(textos_divididos, embeddings)

        # Criação da cadeia de Perguntas e Respostas
        llm = GoogleGenerativeAI(
            model="gemini-1.5-flash-latest", temperature=0.3)
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(),
            return_source_documents=False
        )
        return qa_chain, len(documentos)


# --- LÓGICA PRINCIPAL DA APLICAÇÃO ---
qa_chain, num_docs = carregar_recursos()

if qa_chain:
    st.success(
        f"Base de conhecimento carregada! {num_docs} documentos foram processados com sucesso.")

    # Inicializa o histórico da conversa na memória da sessão
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Olá! Como posso te ajudar hoje?"}]

    # Exibe as mensagens do histórico
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Captura a nova mensagem do usuário
    if prompt := st.chat_input("Digite sua pergunta aqui..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Gera e exibe a resposta do assistente
        with st.chat_message("assistant"):
            with st.spinner("Pensando..."):
                try:
                    resultado = qa_chain.invoke({"query": prompt})
                    resposta = resultado["result"]
                except Exception as e:
                    resposta = f"Desculpe, ocorreu um erro ao processar sua pergunta: {e}"
            st.markdown(resposta)

        st.session_state.messages.append(
            {"role": "assistant", "content": resposta})
else:
    st.warning(
        "O assistente não está pronto. Verifique as mensagens de erro ou aviso acima.")
