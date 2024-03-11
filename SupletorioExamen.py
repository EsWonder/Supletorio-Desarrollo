import os
import streamlit as st
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
import time
from PyPDF2 import PdfReader
import openai

# Set OpenAI API key
os.environ['OPENAI_API_KEY'] = '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'

def process_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        text += pdf_reader.pages[page_num].extract_text()
    return text

def generate_response(uploaded_file, query_text, file_type):
    start_time = time.time()
    if file_type == 'pdf':
        processed_text = process_pdf(uploaded_file)
    else:
        st.error(f"Tipo de archivo no admitido: {file_type}")
        return None, None, None

    # Reduce the length of the text if necessary
    max_context_length = 4097
    processed_text = processed_text[:max_context_length]

    # Split documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.create_documents([processed_text])

    # Select embeddings
    embeddings = OpenAIEmbeddings()

    # Create a vectorstore from documents
    db = Chroma.from_documents(texts, embeddings)

    # Create retriever interface
    retriever = db.as_retriever()

    # Create QA chain
    qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type='stuff', retriever=retriever)
    response = qa.run(query_text)

    elapsed_time = time.time() - start_time
    tokens_consumed = len(response.split()) if response else 0

    return response, elapsed_time, tokens_consumed

# Page title
st.set_page_config(page_title='💻🔗 Prueba Desarrollo')
st.title('💻🔗 Uso de Chroma por Faiss')

# File upload PDF
uploaded_pdf_file = st.file_uploader('Sube un archivo PDF', type=['pdf'])
query_pdf_text = st.text_input('Ingresa tu pregunta para PDF:', placeholder='Por favor proporciona un breve resumen.',
                               disabled=not uploaded_pdf_file)

# Form input and query for PDF
result_pdf = []
with st.form('pdf_form', clear_on_submit=True):
    submitted_pdf = st.form_submit_button('Enviar', disabled=not (uploaded_pdf_file and query_pdf_text))
    if submitted_pdf:
        with st.spinner('Calculando...'):
            response_pdf, elapsed_time_pdf, tokens_consumed_pdf = generate_response(uploaded_pdf_file,
                                                                                    query_pdf_text, 'pdf')
            result_pdf.append(response_pdf)

if len(result_pdf):
    st.info(f"Respuesta PDF: {response_pdf}")
    st.info(f"Tiempo transcurrido PDF: {elapsed_time_pdf} segundos")
    st.info(f"Tokens consumidos PDF: {tokens_consumed_pdf}")

# Clasificador----------------------------------------------------------------------------

import streamlit as st
import openai

# Reemplaza 'tu_clave_de_api_aqui' con tu clave de API real
openai.api_key = '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'
def obtener_respuesta(pregunta, temperatura=0.1):
    try:
        # Agrega la pregunta del usuario a las conversaciones
        conversation = [
            {"role": "system",
             "content": "Eres un clasificador de texto que etiqueta preguntas en las categorías: política, religión, deportes, otros"},
            {"role": "user", "content": pregunta}
        ]

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=conversation,
            temperature=temperatura
        )

        # Accede al contenido directamente
        respuesta = response['choices'][0]['message']['content'].strip()
        return respuesta

    except Exception as e:
        return f"Se produjo un error al obtener la respuesta: {e}"


# Función para clasificar la pregunta
def clasificar_pregunta(pregunta):
    # Lista de palabras clave para cada categoría
    palabras_clave_politica = ["política", "gobierno", "elecciones"]
    palabras_clave_religion = ["religión", "creencia", "fe"]
    palabras_clave_deportes = ["deportes", "atleta", "equipo"]

    # Convertir la pregunta a minúsculas para una comparación sin distinción entre mayúsculas y minúsculas
    pregunta = pregunta.lower()

    # Verificar las palabras clave para determinar la categoría
    if any(palabra in pregunta for palabra in palabras_clave_politica):
        return "política"
    elif any(palabra in pregunta for palabra in palabras_clave_religion):
        return "religión"
    elif any(palabra in pregunta for palabra in palabras_clave_deportes):
        return "deportes"
    else:
        return "otros"


# Interfaz de Streamlit
def main():
    st.title("Clasificador de Texto")

    pregunta_usuario = st.text_input("Ingrese su pregunta:")

    if st.button("Obtener Respuesta"):
        # Clasificar la pregunta
        categoria = clasificar_pregunta(pregunta_usuario)

        # Obtener la respuesta del asistente
        respuesta_asistente = obtener_respuesta(pregunta_usuario)

        # Mostrar la respuesta del asistente sin la categoría 'otros'
        if not respuesta_asistente.startswith("Se produjo un error"):
            st.write(f"Respuesta del asistente: {respuesta_asistente}")


if __name__ == "__main__":
    main()