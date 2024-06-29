import cv2
import os
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
# import libs for user authentication
import smtplib
from email.mime.text import MIMEText
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
                   initial_sidebar_state="collapsed", 
                   menu_items={
                          'Get help': None,
                          'Report a bug': None,
                          'About': "mailto:s190234@th-ab.de"
                              }
                  )
# Session State also supports attribute based syntax
if 'already_registered' not in st.session_state:
    st.session_state.already_registered = False
  
# Creating a login widget
with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['pre-authorized']
)
# Authenticating users
authenticator.login()
if st.session_state["authentication_status"]:
    authenticator.logout(location='sidebar')
    # Creating an update user details widget
    if st.session_state["authentication_status"]:
        try:
            if authenticator.update_user_details(st.session_state["username"], 'sidebar'):
                st.success('Entries updated successfully')
                with open('config.yaml', 'w') as file:
                  yaml.dump(config, file, default_flow_style=False)
        except Exception as e:
            st.error(e) 
    # Creating a reset password widget
    if st.session_state["authentication_status"]:
        try:
            if authenticator.reset_password(st.session_state["username"], 'sidebar'):
                st.success('Password modified successfully')
                with open('config.yaml', 'w') as file:
                  yaml.dump(config, file, default_flow_style=False)
        except Exception as e:
            st.error(e)
    # st.write(f'Welcome {st.session_state["name"]}')
    # st.title('Some content')
    st.header(f'{st.session_state["name"]}!, :grey[Welcome to] :orange[SportCasterAI] - :blue[Goal Predictor]', divider='rainbow')
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
                  
                  DisplayMatchDataFrame(player_id_in_crossing_zone, player_id_in_recipient_zone, player_id_in_pitch, player_with_ball, len(player_team_1), len(player_team_2), player_team_1_in_crossing_zone, player_team_1_in_recipient_zone, player_team_1_in_pitch, player_team_2_in_crossing_zone, player_team_2_in_recipient_zone, player_team_2_in_pitch)
                  GoalPredResults = PredictGoal(df)
                  col2.write(f"The predicted outcome is likely a no goal with a probability of {GoalPredResults:.1%}.")
                  BookermakerSuggestedOdds = CalcBookmakerProfitMargin(goal_probs)
                  st.write(f"If you are a bookermaker this is our calculation for your odds offering (decimal formated) suggested by SportCasterAI {BookermakerSuggestedOdds}")
                except Exception as e:
                    print(e)
                    st.error("Something went wrong. Please try again!")
            else:
                print("Failed to retrieve results within the given timeout.")
elif st.session_state["authentication_status"] is False:
    st.error('Username/password is incorrect')
    col_forgotpassword, col_forgotusername = st.columns(2)
    with col_forgotpassword:
      # Creating a forgot password widget
      try:
          username_of_forgotten_password, email_of_forgotten_password, new_random_password = authenticator.forgot_password()
          if username_of_forgotten_password:
              # The developer should securely transfer the new password to the user.
              email_sender = st.secrets["EMAIL_SENDER"]
              email_receiver = email_of_forgotten_password
              subject = 'SportCasterAI - Forgot Password Request'
              body = f'''Dear {username_of_forgotten_password},
              You recently requested a password reset for your account. To proceed, Here's your fresh new password: {new_random_password}

              Reset Password: {new_random_password}

              If you didn't initiate this request, please ignore this email.

              Best regards,
              SportCasterAI Support Team ''' 
              password_sender = st.secrets["PASSWORD_SENDER"]
              try:
                msg = MIMEText(body)
                msg['From'] = email_sender
                msg['To'] = email_receiver
                msg['Subject'] = subject

                server = smtplib.SMTP('smtp.gmail.com', 587)
                server.starttls()
                server.login(email_sender, password_sender)
                server.sendmail(email_sender, email_receiver, msg.as_string())
                server.quit()
                st.success('New password to be sent securely')
              except Exception as e:
                st.error(f"Error sending email: {e}")
              with open('config.yaml', 'w') as file:
                yaml.dump(config, file, default_flow_style=False)
          elif username_of_forgotten_password == False:
              st.error('Username not found')
      except Exception as e:
          st.error(e)
    with col_forgotusername:
      # Creating a forgot username widget
      try:
          username_of_forgotten_username, email_of_forgotten_username = authenticator.forgot_username()
          if username_of_forgotten_username:
              # The developer should securely transfer the username to the user.
              email_sender = st.secrets["EMAIL_SENDER"]
              email_receiver = email_of_forgotten_username
              subject = 'SportCasterAI - Forgot Username Request'
              body = f'''Dear {username_of_forgotten_username},
              You recently requested to retrieve your username. We're here to help! Your username is: {username_of_forgotten_username}

              If you didn't initiate this request, please ignore this email.

              Best regards,
              SportCasterAI Support Team ''' 
              password_sender = st.secrets["PASSWORD_SENDER"]
              try:
                msg = MIMEText(body)
                msg['From'] = email_sender
                msg['To'] = email_receiver
                msg['Subject'] = subject

                server = smtplib.SMTP('smtp.gmail.com', 587)
                server.starttls()
                server.login(email_sender, password_sender)
                server.sendmail(email_sender, email_receiver, msg.as_string())
                server.quit()
                st.success('Username to be sent securely')
              except Exception as e:
                st.error(f"Error sending email: {e}")
              with open('config.yaml', 'w') as file:
                yaml.dump(config, file, default_flow_style=False)
          elif username_of_forgotten_username == False:
              st.error('Email not found')
      except Exception as e:
          st.error(e)
elif st.session_state["authentication_status"] is None and st.session_state.already_registered is False:
    st.warning('Please enter your username and password. New member? Register now!')
    # Creating a new user registration widget
    try:
        email_of_registered_user, username_of_registered_user, name_of_registered_user = authenticator.register_user(pre_authorization=False)
        if email_of_registered_user:
            st.session_state.already_registered = True
            st.success('User registered successfully')
            with open('config.yaml', 'w') as file:
              yaml.dump(config, file, default_flow_style=False)
    except Exception as e:
        st.error(e)





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

