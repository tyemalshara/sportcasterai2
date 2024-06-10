import cv2
import os
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import streamlit as st
# Essential custom fuctions for the app to run
from main_functions import *

# Adding upper-left logo
LOGO_URL_SMALL = 'https://raw.githubusercontent.com/tyemalshara/sportcasterai2/main/SportCasterAI_Logo_SMALL.png'
st.logo(LOGO_URL_SMALL, link="https://sportcasterai2.streamlit.app/~/+/", icon_image=LOGO_URL_SMALL)
# Adding logo icon to web app
# logo = Image.open('SportCasterAI_Logo.png')
st.set_page_config(page_title="SportCasterAI - Goal Predictor", 
                   page_icon = 'SportCasterAI_Logo.png', 
                   layout="wide", 
                   initial_sidebar_state="auto", 
                   menu_items={
                          'Get help': None,
                          'Report a bug': None,
                          'About': "mailto:s190234@th-ab.de"
                              }
                  )
# # Hiding menu and footer
# hide_default_format = """
#        <style>
#        #MainMenu {visibility: hidden; }
#        footer {visibility: hidden;}
#        </style>
#        """
# st.markdown(hide_default_format, unsafe_allow_html=True)

# hide_streamlit_style = """
#             <style>
#             [data-testid="stToolbar"] {visibility: hidden !important;}
#             footer {visibility: hidden !important;}
#             </style>
#             """
# st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Streamlit UI elements
st.header(':grey[Welcome to] :orange[SportCasterAI] - :blue[Goal Predictor]', divider='rainbow')
YT_URL = st.text_input("Enter YouTube Live URL", placeholder="https://www.youtube.com/watch?v=LTdT9BkW77k")

if YT_URL != "":
  st.video(YT_URL, format="video/mp4", autoplay=True, start_time=0)
  stream_url = stream_to_url(YT_URL, "best")
  if stream_url:  # check if stream_url is not None
    cap = cv2.VideoCapture(stream_url)
    success, frame = cap.read()
    if not success:
      print("Error: Can't receive frame (stream end?). Exiting ...")
    col1, col2 = st.columns(2)
    col1.image(frame, caption='This image is taken from the YouTube live stream. ', channels="BGR")
    cap.release()
    # cv2.destroyAllWindows()
    cv2.imwrite("UserImage.jpg", frame)
    st.session_state.input_file_path = "UserImage.jpg"
    # Read image
    frame = cv2.imread('UserImage.jpg')
    # display the image
    # st.image(frame, caption='Again! This is your image. It has been successfully uploaded.', channels="BGR")
    # add a button to prompt a prediction
    if col2.button("Predict"):
      # Run the prediction
      # add a spinner to the UI
      with st.spinner('Prediction is being processed, results could take a few seconds to 10 minutes...'):
        # Run the workflow on current frame -> wf.run_on(array=frame)
        # Fetch the results from Ikomia's API
        response_uuid, JWT = call_IkomiaAPI()
        results = fetch_workflow_results(response_uuid, JWT)
        if results:
            try:
              print("Workflow results:", results)
              instance_segmentation_objects = results[1]['INSTANCE_SEGMENTATION']['detections']  # instance_segmentation.get_results().get_objects()
              object_detection_objects = results[0]['OBJECT_DETECTION']['detections'] # object_detection.get_results().get_objects()
              player_id_in_crossing_zone, player_id_in_recipient_zone, player_id_in_pitch = players_crossing_zones(instance_segmentation_objects, object_detection_objects, frame)
              player_with_ball = player_near_ball(object_detection_objects)
              player_team_1, player_team_2, player_team_1_in_crossing_zone, player_team_1_in_recipient_zone, player_team_1_in_pitch, player_team_2_in_crossing_zone, player_team_2_in_recipient_zone, player_team_2_in_pitch = team_detection(frame, instance_segmentation_objects)
              df = InitFrameDataDataFrame(player_id_in_crossing_zone, player_id_in_recipient_zone, player_id_in_pitch, player_with_ball, len(player_team_1), len(player_team_2), player_team_1_in_crossing_zone, player_team_1_in_recipient_zone, player_team_1_in_pitch, player_team_2_in_crossing_zone, player_team_2_in_recipient_zone, player_team_2_in_pitch, frame)

              GoalPredResults = PredictGoal(df)
              col2.write(GoalPredResults)
            except Exception as e:
                print(e)
                st.error("Something went wrong. Please try again!")
        else:
            print("Failed to retrieve results within the given timeout.")
