# import streamlit as st
# from bokeh.models.widgets import Button
# from bokeh.models import CustomJS
# from streamlit_bokeh_events import streamlit_bokeh_events

# stt_button = Button(label="Speak", width=100, )

# stt_button.js_on_event("button_click", CustomJS(code="""
#     var recognition = new webkitSpeechRecognition();
#     recognition.continuous = true;
#     recognition.interimResults = true;
 
#     recognition.onresult = function (e) {
#         var value = "";
#         for (var i = e.resultIndex; i < e.results.length; ++i) {
#             if (e.results[i].isFinal) {
#                 value += e.results[i][0].transcript;
#             }
#         }
#         if ( value != "") {
#             document.dispatchEvent(new CustomEvent("GET_TEXT", {detail: value}));
#         }
#     }
#     recognition.start();
#     """))

# result = streamlit_bokeh_events(
#     stt_button,
#     events="GET_TEXT",
#     key="listen",
#     refresh_on_update=False,
#     override_height=75,
#     debounce_time=0)

# if result:
#     if "GET_TEXT" in result:
#         st.write(result.get("GET_TEXT"))

# # import streamlit as st
# # from bokeh.models.widgets import Button
# # from bokeh.models import CustomJS

# # text = st.text_input("Say what ?")

# # tts_button = Button(label="Speak", width=100)

# # tts_button.js_on_event("button_click", CustomJS(code=f"""
# #     var u = new SpeechSynthesisUtterance();
# #     u.text = "{text}";
# #     u.lang = 'en-US';

# #     speechSynthesis.speak(u);
# #     """))

# # st.bokeh_chart(tts_button)
# Python program to translate
# speech to text and text to speech


import speech_recognition as sr
import pyttsx3 

# Initialize the recognizer 
r = sr.Recognizer() 


# Using google to recognize audio
MyText = r.recognize_google("assets\\wav\\ai_resp_audio2.wav")
MyText = MyText.lower()

print("Did you say ",MyText)
# SpeakText(MyText)
