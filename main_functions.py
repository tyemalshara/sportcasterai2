import streamlit as st
# Other tools
# from itertools import zip_longest
import torch
import numpy as np
# import textwrap
import random
from numpy import asarray
import math
import pandas as pd
import cv2
from PIL import Image
from sklearn.cluster import KMeans
from streamlink import Streamlink

def stream_to_url(url, quality='best'):
    """ Get URL, and return streamlink URL """
    session = Streamlink()
    streams = session.streams(url)
    session.set_option("stream-timeout", 30)
    session.set_option("twitch-disable-ads", True)

    if streams:
        return streams[quality].to_url()
    else:
        st.error('Could not locate your stream. \
         (make sure the URL is from YoutTube and the stream is Live and public)', icon="🚨")
        return None
        # raise ValueError('Could not locate your stream.')

def convert_polygon_mask2contours(seg_obj, frame):
  # Extract x and y coordinates
  x_coords = [point['x'] for point in seg_obj]
  y_coords = [point['y'] for point in seg_obj]

  # # Create the polygon coordinates array
  polygon_coords = np.array(list(zip(x_coords, y_coords)))

  width, height, _ = frame.shape
  # Create a black image with the same dimensions
  empty_image = np.zeros((width, height), dtype=np.uint8)

  # Polygon coordinates in the correct format
  # polygon_coords = np.flip(np.argwhere(seg_obj.mask == 1))
  cv2.polylines(empty_image, [polygon_coords], isClosed=True, color=(255, 255, 255))
  # Display the result (you can adapt this part based on your environment)
  # Fill the polygon with white
  cv2.fillPoly(empty_image, pts=[polygon_coords], color=(255, 255, 255))
  # Find the contours of the polygon
  ret, thresh = cv2.threshold(empty_image, 127, 255, 0)
  contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
  contours = np.array(contours, np.int32)
  contours = contours.reshape(-1, 2)

  return contours, empty_image

def players_crossing_zones(instance_segmentation_objects, object_detection_objects_tracked, frame):
  player_id_in_crossing_zone = []
  player_id_in_recipient_zone = []
  player_id_in_pitch = []
  for seg_obj in instance_segmentation_objects:
    for obj in object_detection_objects_tracked:
      H,W,x,y = list(obj['box'].values())
      bottom_left_corner_x = int(x)
      bottom_left_corner_y = int(y+H)
      bottom_right_corner_x = int(x+W)
      bottom_right_corner_y = int(y+H)
      # polygon_coords = np.flip(np.argwhere(seg_obj.mask == 1))  # Assuming 1 represents the polygon
      polygon_countours, empty_image = convert_polygon_mask2contours(seg_obj['polygons'][0]['polygon'], frame)
      result_left = cv2.pointPolygonTest(polygon_countours, (bottom_left_corner_x, bottom_left_corner_y), False)
      result_right = cv2.pointPolygonTest(polygon_countours, (bottom_right_corner_x, bottom_right_corner_y), False)
      if result_left == 1 or result_right == 1:
          # print(f"The player {obj.label} with id {obj.id} is inside the polygon with label: {seg_obj.label}")
          if seg_obj['label'] == "player_crossing_zone":
            player_id_in_crossing_zone.append(obj['id'])
          if seg_obj['label'] == "cross_recipient_zone":
            player_id_in_recipient_zone.append(obj['id'])
          if seg_obj['label'] == "pitch":
            player_id_in_pitch.append(obj['id'])
      # elif result_left == -1 or result_right == -1:
          # print(f"The point {obj.label} with id {obj.id} is outside the polygon with label: {seg_obj.label}")
      # elif result_left == 0 or result_right == 0:
          # print(f"The point {obj.label} with id {obj.id} is on the edge of the polygon with label: ", seg_obj.label)
  return player_id_in_crossing_zone, player_id_in_recipient_zone, player_id_in_pitch

def doOverlap(rect1, rect2, proximity_threshold):
    h1, w1, x1, y1  = rect1  # list(obj['box'].values())
    h2, w2, x2, y2 = rect2

    # Calculate intersection area
    intersection_area = max(0, min(x1 + w1, x2 + w2) - max(x1, x2)) * max(0, min(y1 + h1, y2 + h2) - max(y1, y2))

    # Check if the intersection area is non-zero
    return intersection_area > 0 or (abs(x1 - x2) < proximity_threshold and abs(y1 - y2) < proximity_threshold) or (abs(x1+w1 - x2+w2) < proximity_threshold and abs(y1+h1 - y2+h2) < proximity_threshold)

def player_near_ball(object_detection_objects_tracked):
  player_with_ball = []
  for obj_ball in object_detection_objects_tracked:
    if obj_ball['label'] == "ball":
      ball = list(obj_ball['box'].values())
      for obj in object_detection_objects_tracked:
        if obj['label'] == "player":
          if doOverlap(list(obj['box'].values()), ball, proximity_threshold=50):
              # print(f"{obj.label} with id {obj.id} is near the ball!")
              player_with_ball.append(obj['id'])
          else:
              # print("Rectangles do not overlap.")
              continue
    else:
      # print("ball is missing!")
      continue
  return player_with_ball


def get_AB_value(image, x0, y0, W, H):
  # Convert to LAB color space
  lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

  # Extract A and B channels
  a_channel, b_channel, _ = cv2.split(lab_image)

  # Choose a specific pixel (e.g., at row=100, column=200)
  # row, col = 400, 150
  # row, col = 530, 500
  col = int(x0 + W/2) # get the center pixel of the bounding box of the player x
  row = int(y0 + H/2) # get the center pixel of the bounding box of the player y
  a_value = a_channel[row, col]
  b_value = b_channel[row, col]

  return a_value, b_value

def player_team_detection(image, a_value, b_value):

  # Convert to LAB color space
  lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

  # Extract A and B channels
  a_channel, b_channel, _ = cv2.split(lab_image)

  # Reshape channels for K-means clustering
  ab_channels = np.column_stack((a_channel.flatten(), b_channel.flatten()))

  # Perform K-means clustering (adjust 'n_clusters' as needed)
  n_clusters = 2  # Assuming 2 teams
  kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(ab_channels)

  # Get cluster centroids
  team_colors = kmeans.cluster_centers_

  # Example player color (replace with actual player color)
  player_color = np.array([a_value, b_value])

  # Calculate distances to centroids
  distances = np.linalg.norm(team_colors - player_color, axis=1)

  # Assign player to the closest team
  closest_team = np.argmin(distances)

  # print(f"Player belongs to Team {closest_team + 1}")
  return closest_team + 1

def team_detection(image, instance_segmentation_objects):
  # Run player mask detection on your image -> player_mask_wf.run_on(array=image)
  # Fetch the results
  response_uuid, JWT = call_IkomiaAPI_playermask()
  results = fetch_workflow_results_playermask(response_uuid, JWT)
  if results:
    # print("Workflow results:", results)
    # cv2_imshow(cv2.cvtColor(player_mask_detection.get_image_with_mask_and_graphics(), cv2.COLOR_RGB2BGR))
    # cv2_imshow(player_mask_detection.get_image_with_mask_and_graphics())
    # objects = player_mask_detection.get_results().get_objects()
    objects = results[0]['INSTANCE_SEGMENTATION']['detections']
    # filter out players id's in different zones and store them in the lists
    player_id_in_crossing_zone, player_id_in_recipient_zone, player_id_in_pitch = players_crossing_zones(instance_segmentation_objects, objects, image)

    boxes, masks, ids = list(), list(), list()
    for obj in objects:
      if obj['label'] == "person":
        # Do stuff here on your objects
        boxes.append(obj['box'])
        masks.append(obj['polygons'][0]['polygon'])
        ids.append(obj['id'])
    # print(boxes, '\n', classID, '\n',confidence, "\n", masks)
    instances, visMasks, player_team_1, player_team_2 = list(), list(), list(), list()
    for i in range(0,len(boxes)):
      polygon_contours, empty_image = convert_polygon_mask2contours(masks[i], image)
      # Extract x and y coordinates
      instance = cv2.bitwise_and(image, image, mask=empty_image)
      instance = cv2.cvtColor(instance, cv2.COLOR_RGB2BGR)
      img = Image.fromarray(instance)
      H,W,x0,y0 = list(boxes[i].values())
      x0,y0,W,H = int(x0),int(y0),int(W),int(H)
      x1 = x0 + W
      y1 = y0 + H
      cropped = img.crop((x0,y0,x1,y1)) # left, top, right, bottom
      # visMasks.append(visMask)
      instances.append(instance)
      # cv2.imwrite(f"/content/basketballVideoAnalysis/imagesvisMask{i}.png", visMask)
      # cv2.imwrite(f"/content/basketballVideoAnalysis/instance{i}.png", instance)
      # cropped.save(f"/content/basketballVideoAnalysis/cropped{i}.png")
      a_value, b_value = get_AB_value(instance, x0, y0, W, H)
      player_team = player_team_detection(instance, a_value, b_value)
      # print(f"Player {ids[i]} belongs to Team {player_team}")
      if player_team == 1:
        player_team_1.append(ids[i])  # due to different model used for getting the player mask we don't care about the id since it's impossible to reference or find the link between them
      elif player_team == 2:
        player_team_2.append(ids[i])

    # check if the player's id in player_team_1 is in player_id_in_crossing_zone and player_id_in_recipient_zone and player_id_in_pitch if yes then store it in a new list
    player_team_1_in_crossing_zone = [x for x in player_id_in_crossing_zone if x in player_team_1]
    player_team_1_in_recipient_zone = [x for x in player_id_in_recipient_zone if x in player_team_1]
    player_team_1_in_pitch = [x for x in player_id_in_pitch if x in player_team_1]

    player_team_2_in_crossing_zone = [x for x in player_id_in_crossing_zone if x in player_team_2]
    player_team_2_in_recipient_zone = [x for x in player_id_in_recipient_zone if x in player_team_2]
    player_team_2_in_pitch = [x for x in player_id_in_pitch if x in player_team_2]

    return player_team_1, player_team_2, player_team_1_in_crossing_zone, player_team_1_in_recipient_zone, player_team_1_in_pitch, player_team_2_in_crossing_zone, player_team_2_in_recipient_zone, player_team_2_in_pitch
  else:
    print("Failed to retrieve results within the given timeout.")

import math
import pandas as pd

# create a function that stores data in a dataframe for every frame in the video, and returns the dataframe. the data is stored in a list 'players_in_zone' and 'players_with_ball'.
def InitFrameDataDataFrame(player_id_in_crossing_zone, player_id_in_recipient_zone, player_id_in_pitch, player_with_ball, player_team_1, player_team_2, player_team_1_in_crossing_zone, player_team_1_in_recipient_zone, player_team_1_in_pitch, player_team_2_in_crossing_zone, player_team_2_in_recipient_zone, player_team_2_in_pitch, frame):
  df = pd.DataFrame.from_dict({"player_id_in_crossing_zone": len(player_id_in_crossing_zone), 'player_id_in_recipient_zone:': len(player_id_in_recipient_zone), 'player_id_in_pitch': len(player_id_in_pitch), 'player_with_ball': len(player_with_ball), 'player_team_1': player_team_1, 'player_team_2': player_team_2, 'player_team_1_in_pitch': len(player_team_1_in_pitch), 'player_team_2_in_pitch': len(player_team_2_in_pitch), 'player_team_1_in_crossing_zone': len(player_team_1_in_crossing_zone), 'player_team_2_in_crossing_zone': len(player_team_2_in_crossing_zone), 'player_team_1_in_recipient_zone': len(player_team_1_in_recipient_zone), 'player_team_2_in_recipient_zone': len(player_team_2_in_recipient_zone) }, orient='index')
  df = df.transpose()
  return df

def UpdateFrameDataDataFrame(df, player_id_in_crossing_zone, player_id_in_recipient_zone, player_id_in_pitch, player_with_ball, player_team_1, player_team_2, player_team_1_in_crossing_zone, player_team_1_in_recipient_zone, player_team_1_in_pitch, player_team_2_in_crossing_zone, player_team_2_in_recipient_zone, player_team_2_in_pitch, frame):
  df = df._append({"player_id_in_crossing_zone": len(player_id_in_crossing_zone), 'player_id_in_recipient_zone:': len(player_id_in_recipient_zone), 'player_id_in_pitch': len(player_id_in_pitch), 'player_with_ball': len(player_with_ball), 'player_team_1': player_team_1, 'player_team_2': player_team_2, 'player_team_1_in_pitch': len(player_team_1_in_pitch), 'player_team_2_in_pitch': len(player_team_2_in_pitch), 'player_team_1_in_crossing_zone': len(player_team_1_in_crossing_zone), 'player_team_2_in_crossing_zone': len(player_team_2_in_crossing_zone), 'player_team_1_in_recipient_zone': len(player_team_1_in_recipient_zone), 'player_team_2_in_recipient_zone': len(player_team_2_in_recipient_zone) }, ignore_index=True)
  return df

import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple neural network
class SoccerGoalPredictor(nn.Module):
    def __init__(self):
        super(SoccerGoalPredictor, self).__init__()
        self.fc1 = nn.Linear(12, 64)  # Input layer
        self.fc2 = nn.Linear(64, 2)   # Output layer (2 classes: goal, no_goal)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

loaded_model = SoccerGoalPredictor()
loaded_model.load_state_dict(torch.load('./model.pth'))
loaded_model.eval()  # Set the model to evaluation mode

def PredictGoal(df):

    selected_features = [
        'player_id_in_crossing_zone', 'player_id_in_recipient_zone:',
           'player_id_in_pitch', 'player_with_ball', 'player_team_1',
           'player_team_2', 'player_team_1_in_pitch', 'player_team_2_in_pitch',
           'player_team_1_in_crossing_zone', 'player_team_2_in_crossing_zone',
           'player_team_1_in_recipient_zone', 'player_team_2_in_recipient_zone',  # ... (other features)
        "goal", "no_goal"  # Outcome labels (goal prediction)
    ]
    X = df[selected_features[:-2]].values[0].astype('int64')  # Input features (12 arrays)
    print(X)
    # Now you can predict goals for new match data
    new_match_data = torch.Tensor(X)  # Your 12 input arrays
    predicted_outcome = loaded_model(new_match_data)
    # Apply softmax to the model's output
    softmax_probs = torch.softmax(predicted_outcome, dim=0)
    if predicted_outcome.argmax().item() == 1:
        return f"The predicted outcome is likely a no goal with a probability of {softmax_probs[1].item():.1%}."
    else:
        return f"The predicted outcome is likely a goal with a probability of {softmax_probs[0].item():.1%}."  # {predicted_outcome[0].item():.1%}

import requests
import base64
import json
import time

def call_IkomiaAPI_playermask():

  ############################# Authentication ################################
  IKOMIA_API = st.secrets["IKOMIA_API"]
  fetch_workflow_results_playermask_endpoint = st.secrets["fetch_workflow_results_playermask_endpoint"]
  url = f"https://scale.ikomia.ai/v1/projects/jwt/?endpoint={fetch_workflow_results_playermask_endpoint}/"
  payload = {}
  headers = {
    'Accept': 'application/json',
    'Authorization': f'Token {IKOMIA_API}'
  }
  response = requests.request("GET", url, headers=headers, data=payload)
  JWT = response.json()['id_token']
  # JWT = JWT['id_token']
  with open(r'UserImage.jpg', 'rb') as image_file:
      base64_bytes = base64.b64encode(image_file.read())
      base64_string = base64_bytes.decode()
  ######################## Execute deployment  ##############################
  url = f"{fetch_workflow_results_playermask_endpoint}/api/run"
  payload = json.dumps({
    "inputs": [
      {"image": f"{base64_string}"}
    ],
    "outputs": [
      {
        "task_name": "infer_detectron2_instance_segmentation",
        "task_index": 0,
        "output_index": 1}
    ],
    "parameters": []})
  headers = {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
    'Authorization': f'Bearer {JWT}'
  }
  response_uuid = requests.request("PUT", url, headers=headers, data=payload)

  return response_uuid, JWT

def fetch_workflow_results_playermask(response, JWT):
    # An appropriate time interval (in seconds)
    polling_interval = 10  # Adjust as needed
    # A suitable timeout (in seconds)
    timeout = 90  # Adjust as needed
    while timeout > 0:
        ####################### Retrieve execution results ########################
        # uuid = response.text
        uuid_json = response.json()
        Endpoint_URL = st.secrets["fetch_workflow_results_playermask_endpoint"]
        url = f"{Endpoint_URL}/api/results/{uuid_json}"
        payload = {}
        headers = {
            'Accept': 'application/json',
            'Authorization': f'Bearer {JWT}'
        }
        response_results = requests.request("GET", url, headers=headers, data=payload)
        print(response_results.status_code, type(response_results.status_code))
        if len(response_results.text) > 0 and response_results.text != 'null':
            return json.loads(response_results.text)  # Results retrieved successfully
        time.sleep(polling_interval)  # Wait before the next poll
        timeout -= polling_interval
    return None  # Timeout reached without getting results

# second API call

def call_IkomiaAPI():

  ############################# Authentication ################################
  IKOMIA_API = st.secrets["IKOMIA_API"]
  fetch_workflow_results_endpoint = st.secrets["fetch_workflow_results_endpoint"] 
  url = f"https://scale.ikomia.ai/v1/projects/jwt/?endpoint={fetch_workflow_results_endpoint}/"
  payload = {}
  headers = {
    'Accept': 'application/json',
    'Authorization': f'Token {IKOMIA_API}'
  }
  response = requests.request("GET", url, headers=headers, data=payload)
  JWT = response.json()['id_token']
  # JWT = JWT['id_token']
  with open(r'UserImage.jpg', 'rb') as image_file:
      base64_bytes = base64.b64encode(image_file.read())
      base64_string = base64_bytes.decode()
  ######################## Execute deployment  ##############################
  url = f"{fetch_workflow_results_endpoint}/api/run"
  payload = json.dumps({
    "inputs": [
      {"image": f"{base64_string}"}
    ],
    "outputs": [
      {
        "task_name": "infer_detectron2_detection",
        "task_index": 0,
        "output_index": 1},
      {
        "task_name": "infer_detectron2_instance_segmentation",
        "task_index": 0,
        "output_index": 1}
    ],
    "parameters": [
      # {
      #   "task_name": "infer_detectron2_detection",
      #   "task_index": 0,
      #   "parameters": {}
      # }
    ]})
  headers = {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
    'Authorization': f'Bearer {JWT}'
  }
  response_uuid = requests.request("PUT", url, headers=headers, data=payload)

  return response_uuid, JWT

def fetch_workflow_results(response, JWT):
    # An appropriate time interval (in seconds)
    polling_interval = 10  # Adjust as needed
    # A suitable timeout (in seconds)
    timeout = 60  # Adjust as needed
    while timeout > 0:
        ####################### Retrieve execution results ########################
        # uuid = response.text
        uuid_json = response.json()
        Endpoint_URL = st.secrets["fetch_workflow_results_endpoint"]
        url = f"{Endpoint_URL}/api/results/{uuid_json}"
        payload = {}
        headers = {
            'Accept': 'application/json',
            'Authorization': f'Bearer {JWT}'
        }
        response_results = requests.request("GET", url, headers=headers, data=payload)
        if len(response_results.text) > 0 and response_results.text != 'null':
            return json.loads(response_results.text)  # Results retrieved successfully
        time.sleep(polling_interval)  # Wait before the next poll
        timeout -= polling_interval
    return None  # Timeout reached without getting results
