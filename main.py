import pandas as pd
import cv2
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import joblib
import random
import datetime
import sys
sys.path.append('../')


stroke_width = 30
stroke_color = "#000"
bg_color = "#fff"
drawing_mode = "freedraw"
realtime_update = True

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

# Define the getSum function
def getSum(n):
    while n > 9:
        n = sum(int(digit) for digit in str(n))
    return n

def initialize_session_state():
    if 'current_question' not in st.session_state:
        st.session_state.current_question = 0
    if 'player_score' not in st.session_state:
        st.session_state.player_score = 0
    if 'seed' not in st.session_state:
        current_date = datetime.date.today()
        date_number = int(current_date.strftime("%Y%m%d"))
        st.session_state.seed = date_number

# Function to generate questions
def generate_questions(number_list):
    questions = []
    for number in number_list:
        question = "Quelle est la somme des chiffres de " + str(number) + "? "
        correct_answer = getSum(number)
        questions.append({'question': question, 'correct_answer': correct_answer})
    return questions

def update_score(player_answer, correct_answer, question_number):
    if int(player_answer) == correct_answer:
        st.subheader("Bonne réponse !") 
        st.write(":white_check_mark:")
        st.session_state.player_score += 1
    else:
        
        st.subheader("Mauvaise réponse. La bonne réponse pour "  + str(question_number) + " est " +    str(correct_answer) +'.')

# Generate questions based on a random list of numbers

def page_one():
    st.title("Maths et dessins 	:writing_hand:")
    st.header("Additionne les chiffres du nombre affiché jusqu'à obtenir un chiffre entre 1 et 9 !")
    st.header("Le chiffre déssiné doit être le plus grand possible !")

    if st.button("Jouer !"):
        st.session_state.page = "page_two"
        st.experimental_rerun()





def page_two():
    st.title("Maths et dessins 	:writing_hand:")
    st.header("Additionne les chiffres du nombre affiché jusqu'à obtenir un chiffre entre 1 et 9 !")
    # Create two side-by-side columns
    col1, col2 = st.columns(2)
    with col1:
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            background_color=bg_color,
            background_image=None,
            update_streamlit=realtime_update,
            height=200,
            width= 200,
            drawing_mode=drawing_mode,
            point_display_radius=0,
            display_toolbar= True,
            key="full_app",
        )

        #preprocessing of the image
        img_width = 8
        try:
            gray_array = rgb2gray(canvas_result.image_data)
            image_resized = cv2.resize(gray_array, (img_width,img_width), interpolation = cv2.INTER_AREA)
            image_resized = abs(image_resized -np.max(image_resized))/max(np.max(image_resized),1)
        except:
            image_resized = np.zeros((img_width,img_width), dtype = np.uint8)
        #st.dataframe(image_resized)
        if sum(sum(image_resized)) > img_width*0.1:
            #model = joblib.load('xgb.pkl')
            model = joblib.load('app/model_svm.pkl')
            processed_im = image_resized.reshape((1, -1))
            prediction = model.predict_proba(processed_im)
            answer = prediction[0]
            st.header('Ta réponse : ' + str(answer))
        else:
            answer = None
    

    initialize_session_state()
    random.seed(st.session_state.seed)
    number_list = random.sample(range(1, 1000), 10)
    quiz_questions = generate_questions(number_list)
    current_question = quiz_questions[st.session_state.current_question]
    player_answer = answer
    
    with col2:
        if st.session_state.current_question < len(quiz_questions)-1:
            if st.button("Répondre"):
                update_score( player_answer, current_question['correct_answer'], number_list[st.session_state.current_question])
                st.session_state.current_question += 1
            st.subheader(quiz_questions[st.session_state.current_question]['question'])

            st.write(f"Ton score: {st.session_state.player_score} / {len(quiz_questions)}")


            
        else:
            st.write("Merci d'avoir joué !")
            st.write("Score final: " + str(st.session_state.player_score) + " / " + str(len(quiz_questions)))
            
            if st.button("Nouvelle partie"):
                st.session_state.player_score = 0
                st.session_state.current_question = 0
                st.session_state.seed = st.session_state.seed + 1000

def main():
    # Initialize session state
    if "page" not in st.session_state:
        st.session_state.page = "page_one"

    # Display the appropriate page
    if st.session_state.page == "page_one":
        page_one()
    elif st.session_state.page == "page_two":
        page_two()



if __name__ == "__main__":
    main()

    




