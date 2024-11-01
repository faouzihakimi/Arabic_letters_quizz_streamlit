# import necessary libraries
import pandas as pd
import cv2
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
import numpy as np
import joblib
import random
import datetime
import sys
sys.path.append('../')


# Define global variables
stroke_width = 20
stroke_color = "#000"
bg_color = "#fff"
drawing_mode = "freedraw"
realtime_update = True

arabic_phonetic = {
    'ا': 'alif',
    'ب': 'baa',
    'ت': 'taa',
    'ث': 'thaa',
    'ج': 'jeem',
    'ح': 'ḥaa',      # Using ḥ to denote the emphatic sound
    'خ': 'khā',      # Using khā for the emphatic sound
    'د': 'dāl',
    'ذ': 'dhāl',     # Using dhāl for the emphatic sound
    'ر': 'rā',
    'ز': 'zayn',
    'س': 'sīn',
    'ش': 'shīn',
    'ص': 'ṣād',      # Using ṣ to denote the emphatic sound
    'ض': 'ḍād',      # Using ḍ to denote the emphatic sound
    'ط': 'ṭā',       # Using ṭ to denote the emphatic sound
    'ظ': 'ẓā',       # Using ẓ to denote the emphatic sound
    'ع': 'ʿayn',     # Using ʿ for the voiced pharyngeal fricative
    'غ': 'ghayn',
    'ف': 'fā',
    'ق': 'qāf',
    'ك': 'kāf',
    'ل': 'lām',
    'م': 'mīm',
    'ن': 'nūn',
    'ه': 'hā',       # Keeping this as is since it is distinct
    'و': 'wāw',
    'ي': 'yā'
}

# Create a DataFrame from the dictionary
df_arabic = pd.DataFrame(list(arabic_phonetic.items()), columns=['Arabic Letter', 'Phonetic'])



def rgb2gray(rgb):
    #convert rgb to grayscale
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def initialize_session_state():

    # Initialize session state for the streamlit app

    if 'current_question' not in st.session_state:
        st.session_state.current_question = 0
    if 'player_score' not in st.session_state:
        st.session_state.player_score = 0
    if 'seed' not in st.session_state:
        st.session_state.seed = random.randint(0, 1000)
    if 'answer_check' not in st.session_state:
        st.session_state.answer = 0 #binary
    if 'question_number' not in st.session_state:
        st.session_state.question_number = 0

def reset_session_state():
    #reset the session state when starting a new game
    st.session_state.current_question = 0
    st.session_state.player_score = 0
    st.session_state.seed = random.randint(0, 1000)
    st.session_state.answer = 0 #binary
    st.session_state.question_number = 0



def generate_question():

    random_number = random.randint(0, len(df_arabic['Phonetic']) - 1)
    question = "How do you write the arabic letter '" + df_arabic['Phonetic'][random_number] + "' ? :writing_hand:"
    

    return {'question': question, 'correct_answer': random_number}


def page1():
    #Title page to present the game 
    
    st.title("Learn the arabic letters	:writing_hand:")
    
    
    if st.session_state.question_number ==0:
        st.write("Use the whole drawing screen for optimal results. :spiral_note_pad:")
        if st.button("Play ! :pencil2:"):
            st.session_state.page = 2
            st.experimental_rerun() #to avoid the double click issue
    else:
        st.subheader("Your score: " + str(st.session_state.player_score) + " / " + str(st.session_state.question_number))

        if st.button("Play again !"):
            reset_session_state()
            st.session_state.page = 2
            st.experimental_rerun() #to avoid the double click issue

        

def page2():
    #Game page
    
    random.seed(st.session_state.seed)
    st.session_state.current_question = generate_question()
    st.subheader(st.session_state.current_question['question'])
       

    col1, col2 = st.columns(2, gap="small")
    #col1, col2 = st.columns(2
    with col1:
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            background_color=bg_color,
            background_image=None,
            update_streamlit=realtime_update,
            height=320,
            width= 320,
            drawing_mode=drawing_mode,
            point_display_radius=0,
            display_toolbar= True,
            key="full_app",
        )

        #preprocessing of the image
        img_width = 32
        try:
            gray_array = rgb2gray(canvas_result.image_data)
            image_resized = cv2.resize(gray_array, (img_width,img_width), interpolation = cv2.INTER_AREA)
            image_resized =  1 - image_resized.astype('float32') / 255
            image_resized = image_resized.T #train images are transposed
        except:
            image_resized = np.zeros((img_width,img_width), dtype = np.uint8)
    with col2:
        if st.button("Check :white_check_mark:"):
            if sum(sum(image_resized)) > img_width*0.1:
                #st.write(str(sum(sum(image_resized))))
                model = tf.keras.models.load_model('app/cnn_model.h5')
                image_processed = image_resized.reshape(1, img_width, img_width, 1)
                probas = model.predict(image_processed)
                max_proba = np.max(probas)
                answer = np.argmax(probas, axis=1)[0]
                
                if max_proba <2/3 : 
                    st.write("Unrecognized answer. Please try again.")
                else:
                    #st.write("Reponse: " + str(df_arabic['Phonetic'][answer]))
                    #st.write("Answer: " + str(df_arabic['Phonetic'][st.session_state.current_question['correct_answer']]))
                    
                        st.session_state.answer_check = (answer == st.session_state.current_question['correct_answer'])
                        st.session_state.question_number += 1
                        st.session_state.player_score += st.session_state.answer_check
                        st.session_state.page = 3
                        st.experimental_rerun() #to avoid the double click issue
            else:
                st.write("Please write something.")

        if st.button("Quit", type='primary'):
            st.session_state.page = 1
            st.experimental_rerun()
    if st.button("Help 	:question:"):
        #Display df_arabic
        st.dataframe(df_arabic)                

def page3():
    #Result page

    st.session_state.seed = st.session_state.seed + 1  
    if  st.session_state.answer_check: 
        st.subheader("Well done !  	:+1: " )
        
    else:
        st.subheader("That's not it. :x:") 
        #add an encouragement
        if st.button("It was the right answer", ):
            st.write("Sorry, my mistake. :sweat_smile:")
            st.write("For optimal results, use the whole drawing screen. Also try to draw the letter as clear as possible :spiral_note_pad:")


    
    st.subheader("The letter " + str(df_arabic['Phonetic'][st.session_state.current_question['correct_answer']]) + " is written " + str(df_arabic['Arabic Letter'][st.session_state.current_question['correct_answer']]) )      
    
    st.write("Your score: " + str(st.session_state.player_score) + " / " + str(st.session_state.question_number))
    
    if st.button("Continue"):
        
        st.session_state.page = 2
        st.experimental_rerun()
    elif st.button("Quit", type='primary'):
        st.session_state.page = 1
        st.experimental_rerun()
            




    



def main():

    # Initialize session state
    if "page" not in st.session_state:
        st.session_state.page = 1
    
    initialize_session_state()
    if st.session_state.page == 1:
        page1()
    elif st.session_state.page == 2:
        page2()

    elif st.session_state.page == 3:
        page3()


if __name__ == "__main__":
    main()

    




