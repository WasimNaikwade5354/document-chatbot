import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, user_template, bot_template
from langchain_google_genai import GoogleGenerativeAI
import google.generativeai as genai
import os
from langchain_community.embeddings import GooglePalmEmbeddings
from io import BytesIO
from gtts import gTTS 
import librosa
import firebase_admin
from firebase_admin import credentials
from firebase_admin import auth
from st_audiorec import st_audiorec
import speech_recognition as sr

GENAI_API_KEY = "AIzaSyCbRM1raXsjteH45M92e49YvmBTkcTN1M0"
os.environ["GOOGLE_API_KEY"] = GENAI_API_KEY

genai.configure(api_key=GENAI_API_KEY)
r = sr.Recognizer()



########################################Login/SignUp########################################################

st.set_page_config(page_title="Chat with multiple PDFs",
                    page_icon=":books:")


if 'username' not in st.session_state:
    st.session_state.username = ''
if 'useremail' not in st.session_state:
    st.session_state.useremail = ''



def f(): 
    try:
        user = auth.get_user_by_email(email)
        print(user.uid)
        st.session_state.username = user.uid
        st.session_state.useremail = user.email
        
        global Usernm
        Usernm=(user.uid)
        
        st.session_state.signedout = True
        st.session_state.signout = True    
        
    except: 
        st.warning('Login Failed')

def t():
    st.session_state.signout = False
    st.session_state.signedout = False   
    st.session_state.username = ''


    

    
if "signedout"  not in st.session_state:
    st.session_state["signedout"] = False
if 'signout' not in st.session_state:
    st.session_state['signout'] = False
# creating a audio checking session state
if 'is_aud' not in st.session_state:
    st.session_state['is_aud'] = False    
    

    

if  not st.session_state["signedout"]: # only show if the state is False, hence the button has never been clicked
    choice = st.selectbox('Login/Signup',['Login','Sign up'])
    email = st.text_input('Email Address')
    password = st.text_input('Password',type='password')
    

    
    if choice == 'Sign up':
        username = st.text_input("Enter  your unique username")
        
        if st.button('Create my account'):
            user = auth.create_user(email = email, password = password,uid=username)
            
            # st.audio_countccess('Account created audio_countccessfully!')
            st.markdown('Please Login using your email and password')
            st.balloons()
    else:
        # st.button('Login', on_click=f)          
        st.button('Login', on_click=f)
        
        
    
    
##############################################PDF ChatBot###################################################

def get_duration_librosa(file_path):
   audio_data, sample_rate = librosa.load(file_path)
   duration = librosa.get_duration(y=audio_data, sr=sample_rate)
   return duration


def text_to_speech(text):
    """
    Converts text to an audio file using gTTS and returns the audio file as binary data
    """
    audio_bytes = BytesIO()
    tts = gTTS(text=text, lang="en")
    tts.write_to_fp(audio_bytes)
    audio_bytes.seek(0)
    return audio_bytes.read()

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = GooglePalmEmbeddings(model="models/embedding-001", google_api_key=GENAI_API_KEY)
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = GoogleGenerativeAI(model="models/text-bison-001", google_api_key=GENAI_API_KEY)

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

if 'doc' not in st.session_state:
    st.session_state['doc'] = False

if 'audio_count' not in st.session_state:
    st.session_state['audio_count'] = 0

if 'audio_history' not in st.session_state:
    st.session_state['audio_history'] = {}

# audio_history = {}
def handle_userinput(user_question):
    # global audio_count
    # audio_count = 0
    if st.session_state.doc == False:
        st.warning("To Chat, Please Load PDF!")
    elif st.session_state.doc == True:
            response = st.session_state.conversation({'question': user_question})
            # print(response)
            
            st.session_state.chat_history = response['chat_history']
            # print(type(text_to_speech(system_response)))
            with st.spinner("Generating..."):
                # try:
                system_response =  str(st.session_state.chat_history[-1]).split("content='")[1]
                # st.session_state['audio_chat_history'] = text_to_speech(system_response)
                st.session_state['audio_count'] += 1
                audio_hist = st.session_state.audio_chat_history = text_to_speech(system_response)
                # audio_count+=1
                # print("EXECUTED")
                print(st.session_state['audio_count'])
                if st.session_state['audio_count'] % 2 != 0:
                    st.session_state['audio_history'][st.session_state['audio_count']] = text_to_speech(system_response)
                elif st.session_state['audio_count'] % 2 == 0:
                    st.session_state['audio_count']+=1
                    st.session_state['audio_history'][st.session_state['audio_count']] = text_to_speech(system_response)
                #     print("EXE")
                # print(st.session_state['audio_history'].keys())
                # print(st.session_state['audio_history'])
                for i, message in enumerate(st.session_state.chat_history):
                    if i % 2 == 0:
                        st.write(user_template.replace(
                            "{{MSG}}", message.content), unsafe_allow_html=True)
                    else:
                        st.write(bot_template.replace(
                            "{{MSG}}", message.content), unsafe_allow_html=True)
                        audio_chat = st.audio(st.session_state['audio_history'][i], format="audio/wav")
                        # print(st.session_state.audio_chat_history)
                        # st.session_state.chat_history = audio_chat

        # except:
        #         st.warning("Please Re-Upload File.")
        # except:
        #     st.warning("Give Clear Input To Get The Answer.")
            # st.audio(text_to_speech(system_response), format="audio/wav")
                
def main():
    st.write(css, unsafe_allow_html=True)
    text = None
    # wav_audio_data = None

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "audio_chat_history" not in st.session_state:
        st.session_state.audio_chat_history = None
    if 'ok_a' not in st.session_state:
        st.session_state.ok_a = False
    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = False


    st.title("Chat with multiple PDFs :books:")
    user_question = st.chat_input(placeholder="Ask a question about your documents:")

    

    with st.sidebar:
        st.header("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True, type="pdf")
        if len(pdf_docs) != 0:
            print(len(pdf_docs))
            
            if not st.session_state.uploaded_file:
                st.session_state.uploaded_file = True
                print("The file exists.")
            if st.button("Process"):
                st.session_state.doc = True
                if not st.session_state.ok_a:
                    with st.spinner("Processing"):
                        # get pdf text
                        raw_text = get_pdf_text(pdf_docs)

                        # get the text chunks
                        text_chunks = get_text_chunks(raw_text)

                        # create vector store
                        vectorstore = get_vectorstore(text_chunks)

                        # create conversation chain
                        st.session_state.conversation = get_conversation_chain(
                            vectorstore)
                    st.success("File uploaded successfully!")
                    st.session_state.ok_a = True
            st.write("ask questions using voice:")
            wav_audio_data = st_audiorec()
            if not user_question:
                if wav_audio_data is not None:
                    st.session_state.is_aud = True
                    print("Not None")
                    if wav_audio_data != None:
                        with open("user_aud.wav", "wb") as source:
                            source.write(wav_audio_data)
                            wav_audio_data = None

                        # Using google to recognize audio
                        with sr.AudioFile("user_aud.wav") as source:
                            # listen for the data (load audio to memory)
                            audio_data = r.record(source)
                            # recognize (convert from speech to text)
                            text = r.recognize_google(audio_data)
                        # handle_userinput(text)
            
        elif len(pdf_docs) == 0:
            st.session_state.conversation = None
            st.session_state.chat_history = None
            st.session_state.audio_chat_history = None
        st.button('Sign out', on_click=t)
    if st.session_state.is_aud:
        handle_userinput(text)
        st.session_state.is_aud = False
        text = None
    elif user_question and text == None:
        print("Got user quest")
        handle_userinput(user_question)  

if __name__=='__main__':
    try:
        if not firebase_admin._apps:
            cred = credentials.Certificate('C:\\Users\\samis\\OneDrive\\Desktop\\DOC Chatbot\\ask-multiple-pdfs-main\\document-chatbot-8b13f-6877f0069a1d.json')
            default_app = firebase_admin.initialize_app(cred)
    except:
        pass
    if st.session_state.signout:
        main()
















