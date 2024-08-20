# Импорт библиотек
import streamlit as st 
from sentence_transformers import SentenceTransformer
import chromadb
from llama_cpp import Llama


# -----------------------------------------------------------------------------
# --                              Секция путей                               --
# -----------------------------------------------------------------------------

# Определяем пути до ИИ моделей
SentenceModelPath = "C:/AI-models/Embeddings/multilingual-e5-large" # локальный путь для модели эмбедера
# model_full_path = "C:/AI-models/models/saiga/model-q4_K.gguf" # локальный путь для NLP модели
# model_full_path = "C:/AI-models/models/T-lite-instruct-0.1/T-lite-instruct-0.1.Q4_1.gguf" # локальный путь для NLP модели
model_full_path = "C:/AI-models/models/vihr/it-5.4-fp16-orpo-v2-Q4_1.gguf" # локальный путь для NLP модели

# Векторная база данных
DB_path = "C:/NPK-2/Bases/knowledge_base/"  # Путь к векторной базе данных
collection_name = "knowledge_base_collection_microsoft"  # Имя коллекции в векторной базе данных


# -----------------------------------------------------------------------------
# --                       Секция констант/переменных                        --
# -----------------------------------------------------------------------------

number_results=8 # Количество передаваемых чанков в промпт
distance=.4 # Ограничение дистанции найденных чанков от заданного вопроса
top_k=30
top_p=0.9
temperature=0.1
repeat_penalty=1.1

# Шаблон для генерации вывода на основе контекста и вопроса
PROMPT_TEMPLATE = """
Ты специалист консультант по ЭОС 'Дело'. Ответь на вопрос, базируясь только на этом контексте:

{context}

---

Ответь на вопрос, используя только контекст: {question}
"""

# -----------------------------------------------------------------------------
# --                Секция инициализации хранилища и модели                  --
# -----------------------------------------------------------------------------
# префикс @st.cache_resource используется для кеширования функции Streamlit ом

@st.cache_resource
def initChroma(DB_path):     # Создание клиента Chroma для работы с постоянным хранилищем (PersistentClient)
    chroma_client = chromadb.PersistentClient(path=DB_path)
    return chroma_client

@st.cache_resource
def initCollection():        # Получение коллекции из векторной базы данных
    collection = chroma_client.get_or_create_collection(name=collection_name)
    return collection

@st.cache_resource
def initSentenceModel():     # Инициализация модели для работфы с word embedding
    SentenceModel = SentenceTransformer(SentenceModelPath) 
    return SentenceModel

@st.cache_resource
def initModel():             # Инициализация ИИ модели для генерации ответа
    model = Llama(
        model_path=model_full_path,
        n_ctx=8092,
        n_gpu_layers=-1, n_threads=32, n_batch=1024, 
        n_parts=1,
        verbose=False,
       )
    return model
    
# -----------------------------------------------------------------------------
# --                     Секция вспомогательных процедур                     --
# -----------------------------------------------------------------------------

# Функция для поиска и генерации промпта на основе контекста
def search_and_generate_prompt(question, collection, SentenceModel, distance_threshold, n_results=number_results):
    # Оцифровка текста запроса и преобразование его в список
    query_embedding = SentenceModel.encode([question])[0].tolist()

    # Запрос к коллекции для поиска документов, которые наиболее близки к запросу
    results = collection.query(
        query_embeddings=[query_embedding],  # Векторное представление текста запроса в виде списка
        n_results=n_results  # Количество возвращаемых результатов
    )
    print (results['distances'][0]) # Отладочная информация
    # Фильтрация результатов по порогу расстояния (distance_threshold)
    filtered_results = [
        (doc, metadata, dist) for doc, metadata, dist in zip(results['documents'][0], results['metadatas'][0], results['distances'][0]) if dist <= distance_threshold
    ]

    # Проверка наличия релевантных результатов
    if not filtered_results:
        return "В базе знаний не нашлось необходимой информации.", ""

    # Формирование контекста из отфильтрованных результатов
    context = " ".join([doc for doc, _, _ in filtered_results])

    # Формирование списка метаданных для вывода
    metadata_info = "  \n  ".join([
        f"File: {metadata['filename']}, Page: {metadata.get('page_number', 'N/A')}" 
        for _, metadata, _ in filtered_results if metadata is not None
    ])
 
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)  # Подстановка контекста и вопроса в шаблон
    return prompt, metadata_info


# -----------------------------------------------------------------------------
# --                        Секция генерации ответов                         --
# -----------------------------------------------------------------------------

def getAnswer(question):


    logging.info(f"Question: {question}.")



    prompt, metadata_info = search_and_generate_prompt(question, collection, SentenceModel, distance)
    logging.info(f"Prompt: {prompt}.")

    if "В базе знаний не нашлось необходимой информации." in prompt:
        return prompt
    else:
    # Отправка промпта модели и получение ответа
        logging.info(f"Prompt:{prompt}")
        completion  =  model.create_chat_completion(
            messages=[ {"role": "user", "content": prompt} ],
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
            stream=False)
        response = completion['choices'][0]['message']['content']
        logging.info(f"{response}Источники информации:{metadata_info}")

        return f"{response}  \n  \n  **Источники информации:**  \n  {metadata_info}"
    



import logging

logging.basicConfig(level=logging.INFO, filename="py_log.log",filemode="w",
                    format="%(asctime)s %(levelname)s %(message)s")
logging.info("Старт сервера")


# -----------------------------------------------------------------------------
# --                     Секция работы со StreamLit                          --
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="ИИ песочница",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
    )


chroma_client = initChroma(DB_path)
collection = initCollection()
SentenceModel = initSentenceModel()
model = initModel()

    
st.title("ИИ песочница")
container = st.container(border=True) 

container.write("Это песочница для НПК.")
if container.button("Очистить чат"):
    if "messages"  in st.session_state:
        st.session_state.messages = []

 


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

    # Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])



        

    # React to user input
if prompt := st.chat_input("Введите Ваш запрос к модели:"):
    chat_prompt = f"**{prompt}**"
    # Display user message in chat message container
    st.chat_message("user").write(chat_prompt)
      

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": chat_prompt})

    response = getAnswer(prompt) # Это главная функция, все остальное - обвязка!

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.write(response)
        
        #st.html(response)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

        
    

