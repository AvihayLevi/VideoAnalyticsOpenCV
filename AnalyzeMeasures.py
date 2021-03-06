from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import TextOperationStatusCodes
from azure.cognitiveservices.vision.computervision.models import TextRecognitionMode
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials

from array import array
import os
from PIL import Image
import sys
import time
import io
import json
import requests
import re
import cv2
import math
import numpy as np
import base64
from ExceptionsModule import *


def order_points(pts):
    rect = np.zeros((4, 2), dtype = "float32")
    num_pts = np.array(pts)
    s = num_pts.sum(axis = 1)
    rect[0] = num_pts[np.argmin(s)]
    rect[2] = num_pts[np.argmax(s)]
    diff = np.diff(num_pts, axis = 1)
    rect[1] = num_pts[np.argmin(diff)]
    rect[3] = num_pts[np.argmax(diff)]
    return rect


def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
    [0, 0],
    [maxWidth - 1, 0],
    [maxWidth - 1, maxHeight - 1],
    [0, maxHeight - 1]], dtype = "float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    # copy = image.copy()
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    return warped


def detect_markers(frame):
    '''cv2.imshow('frame',frame)
    if cv2.waitKey(100) & 0xFF == ord('q'):
       pass
    time.sleep(10)'''
    
    '''frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(64,64))
    frame = clahe.apply(frame)'''
    
    dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_1000)
    parameters =  cv2.aruco.DetectorParameters_create()
    # print('detencting')
    fixed_corners = []
    markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(frame, dictionary, parameters=parameters)
    clone = frame.copy()
    for mc in markerCorners: # top left, top right, bottom right and bottom left.
        # cv2.rectangle(clone, (mc[0][3][0], mc[0][3][1]), (mc[0][1][0], mc[0][1][1]), (0, 255, 0), 2)
        fixed_corners.append((np.mean([mc[0][0][0], mc[0][1][0], mc[0][2][0], mc[0][3][0]]),np.mean([mc[0][0][1], mc[0][1][1], mc[0][2][1], mc[0][3][1]])))
        
    # cv2.imshow("Window", clone)
    # cv2.waitKey(0)
    #time.sleep(3)
    return fixed_corners



def sliding_window(image, step_size, window_size):
	# slide a window across the image
  for y in range(0, image.shape[0], step_size):
    for x in range(0, image.shape[1], step_size):
			# yield the current window
      if (y + window_size[1] <= image.shape[0]) and (x + window_size[0] <= image.shape[1]):
        yield (x, y, image[y:y + window_size[1], x:x + window_size[0]]) #we want only full windows
      else:
        continue


def sliding_up_window(image, step_size, window_size):
	# slide a window across the image
  for y in reversed(range(0, image.shape[0], step_size)):
    for x in range(0, image.shape[1], step_size):
			# yield the current window
      if (y >= (image.shape[0] / 2)) and (y + window_size[1] <= image.shape[0]) and (x + window_size[0] <= image.shape[1]): 
        print('total is:',image.shape[0], 'y is:', y)
        yield (x, y, image[y:y + window_size[1], x:x + window_size[0]]) #we want only full windows
      else:
        continue


def find_best_windows(computervision_client, warped_frame, mode="simple", num_of_windows = 2): #windos in format [y_down, y_up, x_left, x_right]
    #simple mode, fixed size
    device_type = os.getenv("DEVICE_TYPE")
    print("device type: ", device_type)
    if mode == 'simple': #we know the windows size
        if device_type == "respiration":
            bottom_window = [0.6, 1, 0, 1]
            left_window = [0, 1, 0, 0.35]
            return [bottom_window, left_window]
        elif device_type == "monitor":
            bottom_window = [0.6, 1, 0, 1]
            right_window = [0, 1, 0.65, 1]
            return [bottom_window, right_window]
        else:
            print("UNKNOWN DEVICE TYPE!")
            raise Exception("UNKNOWN DEVICE TYPE!")
        # bottom_window = [0.6, 1, 0, 1]
        # left_window = [0, 1, 0, 0.35]
        # right_window = [0, 1, 0.65, 1]
        # return [bottom_window, right_window, left_window]
    
    """ !!! IF YOU WISH TO CONTINUE - NEED TO DEAL WITH INTEGRATING INTEL OCR TO get_digits_FBW !!! """
    # Pre Process:
    gray = cv2.cvtColor(warped_frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    warped_frame = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, gray)
    
    #variables deff
    best_score_v, best_score_h = 0, 0
    best_window_v, best_window_h= [], []
    s = warped_frame.shape
    min_x, min_y = s[1]/40, s[0]/40
    min_size = min_x * min_y  # min_size is set to be 3% y times 2.5%x
    
    #horizontal w?ndows
    # Avihay changed from 1/10 to 1/8:
    step_size = math.ceil(s[1] / 8)
    winH, winW =  math.ceil(0.3*s[0]), s[1] #define horizontal sliding window size to be 0.3Y by X
    for (x, y, window) in sliding_up_window(warped_frame, step_size, window_size=(winW, winH)):
      temp_results = get_digits_FBW(window, computervision_client)
      # print("HERE!", temp_results)
      temp_results = [x for x in temp_results if ((x[1][4] - x[1][0]) * (x[1][5] - x[1][1])) > min_size and abs(x[1][4] - x[1][0]) > min_x and  abs(x[1][5] - x[1][1]) > min_y] # filter all results smaller than min size
      temp_score = len(temp_results)
      if temp_score > best_score_h:
        best_score_h = temp_score
        best_window_h = [x, x + winW, y, y + winH]
    #vertical w?ndows
    if num_of_windows == 2:
      winH, winW = s[0], math.ceil(0.3*s[1])  # define vertical sliding window size to be 0.3X by Y
      step_size = math.ceil(s[0] / 10)
      if best_window_h:
        processed_frame_v = cv2.rectangle(warped_frame, (best_window_h[0], best_window_h[2]), (best_window_h[1], best_window_h[3]), (0, 255, 0), -1)# block area
      else:
        processed_frame_v = warped_frame
      for (x, y, window) in sliding_window(processed_frame_v, step_size, window_size=(winW, winH)):
        temp_results = get_digits_FBW(window, computervision_client)
        temp_results = [x for x in temp_results if ((x[1][4] - x[1][0]) * (x[1][5] - x[1][1])) > min_size and abs(x[1][4] - x[1][0]) > min_x and  abs(x[1][5] - x[1][1]) > min_y] # filter all results smaller than min size
        temp_score = len(temp_results)
        if temp_score >= best_score_v:
          best_score_v = temp_score
          best_window_v = [x, x + winW, y, y + winH]
    # areas_dict value format is: [y_down, y_up, x_left, x_right]
    final_result = []
    if best_window_v:
        print('best V : ', [best_window_v[2] / s[0], best_window_v[3] / s[0], best_window_v[0] / s[1], best_window_v[1] / s[1]])
        final_result.append([best_window_v[2] / s[0], best_window_v[3] / s[0], best_window_v[0] / s[1], best_window_v[1] / s[1]])
    if best_window_h:
        print('best H : ', [best_window_h[2] / s[0], best_window_h[3] / s[0], best_window_h[0] / s[1], best_window_h[1] / s[1]])
        final_result.append([best_window_h[2] / s[0], best_window_h[3] / s[0], best_window_h[0] / s[1], best_window_h[1] / s[1]])
    return final_result


def create_areas(area_dict, img):
    s = img.shape
    height, width = s[0], s[1]
    areas = []
    for key, value in area_dict.items():
        hmin, hmax, wmin, wmax = value
        # print(hmin, hmax, wmin, wmax)
        hmin *= height
        hmax *= height
        wmin *= width
        wmax *= width
        # print(hmin, hmax, wmin, wmax)
        new_area = [img[math.ceil(hmin):math.ceil(hmax), math.ceil(wmin):math.ceil(wmax)], hmin, wmin]
        areas.append(new_area)
    return areas

def transform_coords(coords, area, mode='avihay'):
    # print(coords[0][0])
    topleft = (coords[0][0]+area[2], coords[0][1]+area[1])
    bottomright = (coords[1][0]+area[2], coords[1][1]+area[1])
    return (topleft, bottomright)


def fix_string(s):
    json_string_fin = ""
    last_c=""
    for c in s:
        if c!="\'":
            json_string_fin += c
            if c=="{":
                if last_c=="\'":
                    json_string_fin = last_string + "\'{"
        else:
            last_string = json_string_fin
            if last_c!="}":
                json_string_fin += "\""
            else:
                json_string_fin += "\'"
        last_c = c
    return json_string_fin


def bounding_boxes_output_former(bbox_dict, mon_id, encoded_image):
    string_json = json.dumps(bbox_dict)
    json_dict = {}
    json_dict["JsonData"] = string_json
    json_dict["MonitorID"] = mon_id
    json_dict["MonitorImage"] = encoded_image
    json_dict_string = str(json_dict)
    # print(json_dict_string)
    output = fix_string(json_dict_string)
    return output


def setup_output_former(fixed_coords, areas, encoded_image, id, corners):
    jdict = {}
    coord_dicts = [{"point": v, "id": k, "type":None} for k, v in fixed_coords.items()]
    areas_dicts = [{"point": v, "id": k} for k, v in areas.items()]
    jdict["boundries"] = coord_dicts
    jdict["areas"] = areas_dicts
    jdict["mapping_image"] = encoded_image
    jdict["id"] = id
    jdict["corners"] = [{"id": str(i), "point": [str(x[0]), str(x[1])]} for i,x in enumerate(corners)]
    jdict["status"] = "AfterCV"
    output = json.dumps(jdict)
    # print(output)
    return output


def get_digits_FBW(image, computervision_client):
    encodedFrame = cv2.imencode(".jpg", image)[1].tostring()
    recognize_printed_results = computervision_client.batch_read_file_in_stream(io.BytesIO(encodedFrame), raw = True)
    operation_location_remote = recognize_printed_results.headers["Operation-Location"]
    operation_id = operation_location_remote.split("/")[-1]
    while True:
        get_printed_text_results = computervision_client.get_read_operation_result(operation_id)
        if get_printed_text_results.status not in ['NotStarted', 'Running']:
            break
    results = []
    if get_printed_text_results.status == TextOperationStatusCodes.succeeded:
        for text_result in get_printed_text_results.recognition_results:
            for line in text_result.lines:
              s = re.sub('[^0123456789./:]', '', line.text)
              if s != "":
                  if s[0] == ".":
                      s = s[1:]
                  s = s.rstrip(".")
                  results.append((s, line.bounding_box))
              else:
                  continue
    return results


def get_digits(img, computervision_client):
    encodedFrame = cv2.imencode(".jpg", img)[1].tostring()
    recognize_printed_results = computervision_client.batch_read_file_in_stream(io.BytesIO(encodedFrame), raw = True)
    # Reading OCR results
    operation_location_remote = recognize_printed_results.headers["Operation-Location"]
    operation_id = operation_location_remote.split("/")[-1]
    while True:
        get_printed_text_results = computervision_client.get_read_operation_result(operation_id)
        if get_printed_text_results.status not in ['NotStarted', 'Running']:
                break
        time.sleep(0.1)
    
    tmp_frame = cv2.imdecode(np.frombuffer(encodedFrame, np.uint8), -1)
    results = []
    text_flag = True
    show_frame_flag = False
    if get_printed_text_results.status == TextOperationStatusCodes.succeeded:
        for text_result in get_printed_text_results.recognition_results:
            for line in text_result.lines:
                print(line.text, line.bounding_box)
                s = re.sub('[^0123456789./:]', '', line.text)
                if s != "":
                    if s[0] == ".":
                        s = s[1:]
                    s = s.rstrip(".")
                    text_flag = True
                    top_left_coords = (int(line.bounding_box[0]), int(line.bounding_box[1]))
                    bottom_right_coords = (int(line.bounding_box[4]), int(line.bounding_box[5]))
                    cv2.rectangle(tmp_frame, top_left_coords, bottom_right_coords, (255,0,0), 2)
                    results.append((top_left_coords, bottom_right_coords))
                else:
                    continue
        if text_flag and show_frame_flag:
            print(results)
            cv2.imshow("image", tmp_frame)
            cv2.waitKey(0)
    return(results)


def get_intel_digits(img, mode):
    # tmp_frame = cv2.imdecode(np.frombuffer(img, np.uint8), -1)
    # cv2.imshow("image", tmp_frame)
    # cv2.waitKey(0)
    enc_img = base64.b64encode(img)
    data = {"image": str(enc_img, 'utf-8')}
    INTEL_OCR_ENDPOINT = os.getenv("INTEL_OCR_ENDPOINT")
    # res = requests.post("http://127.0.0.1:8088/run_ocr", json=data)
    res = requests.post(INTEL_OCR_ENDPOINT, json=data)
    res_str = json.loads(res.text)
    results_dicts = [{"text": x["text"], "coords": [x["coords"]["left"], x["coords"]["top"], x["coords"]["right"], x["coords"]["top"], x["coords"]["right"], x["coords"]["bottom"], x["coords"]["left"], x["coords"]["bottom"]]} for x in res_str]
    # print(results_dicts)
    filtered_results = []
    for item in results_dicts:
        if mode == "digits":
            s = re.sub('[^0123456789./:]', '', item['text'])
            if s != "":
                if s[0] == ".":
                    s = s[1:]
                s = s.rstrip(".")
                # filtered_results.append((s, item['coords']))
                top_left_coords = (int(item["coords"][0]), int(item["coords"][1]))
                bottom_right_coords = (int(item["coords"][4]), int(item["coords"][5]))
                filtered_results.append((top_left_coords, bottom_right_coords))
            else:
                continue
        elif mode == "text":
            s = item['text']
            if s != "":
                filtered_results.append((s, item['coords']))
            else:
                continue
    # print("--------------------")
    # print(filtered_results)
    return(filtered_results)


def is_same_bounding(boundings_1, boundings_2): #bounding in form of (top_left_coords, bottom_right_coords)
    output = abs(boundings_1[0][0] - boundings_2[0][0]) < 17
    output = output and abs(boundings_1[0][1] - boundings_2[0][1]) < 17
    output = output and abs(boundings_1[1][0] - boundings_2[1][0]) < 17
    output = output and abs(boundings_1[1][1] - boundings_2[1][1]) < 17
    return output 


def AnalyzeMeasures(frame, computervision_client):
    frame = cv2.imdecode(np.frombuffer(frame, np.uint8), -1)
    # cv2.imwrite("image.jpg", frame)
    
    corners = detect_markers(frame)
    print(corners)
    if len(corners) != 4:
        # TODO: throw exception if 4 corners wern't found for a long time - for now, there's a one minuet timer at the back-end
        print(len(corners))
        print("NOT DETECTED 4 CORNERS!")
        time.sleep(2)
        return False
    frame = four_point_transform(frame, corners)

    areas_of_intrest = find_best_windows(computervision_client, frame, "simple", 2) #find best windows
    areas_dict = {i:area for i,area in enumerate(areas_of_intrest)} #transform into dictionary of bounderies
    areas = create_areas(areas_dict, frame)

    # our output
    transformed_coords = {}
    i = 0
    CV_MODEL = os.getenv("CV_MODEL")
    for area in areas:
        try:
            if CV_MODEL == "MSOCR":
                result = get_digits(area[0], computervision_client)
            elif CV_MODEL == "INTEL":
                result = get_intel_digits(cv2.imencode(".jpg", area[0])[1], "digits")
            else:
                raise Exception("UNRECOGNIZED MODEL")
        except Exception as e:
            print("Exception in get_digits: \n", e)
            raise e
        print(result)
        for item in result:
            transformed_boundries = transform_coords(item, area)
            if True in [is_same_bounding(x,transformed_boundries) for x in transformed_coords.values()]: #at least one of current boundings is the same
                print('duplicate detected!')
                continue #duplicate detected, do not add this value!
            transformed_coords[i] = transformed_boundries
            i = i + 1
    
    print("fixed coords are:", transformed_coords)
    print("Areas on interest (in percentage) are: ", areas_dict)
    
    #TODO: add argument to choose whether or not to send response (send and/or print)
    b64img = base64.b64encode(cv2.imencode(".jpg", frame)[1])
    b64_encoded_frame = b64img.decode('utf-8')
        
    monitor_id = os.getenv("DEVICE_ID")
    json_string = setup_output_former(transformed_coords, areas_dict, b64_encoded_frame, monitor_id, corners)
    url = os.getenv("API_URL")
    headers = {'Content-type':'application/json', 'Accept':'application/json'}
    response = None
    for trail in range(4):
        try:
            print("SENDING")
            response = requests.post(url, data=json_string, headers=headers)
            if response.status_code != 200:
                raise(Exception("Bad API Status Code: " + str(response.status_code)))
        except Exception as e:
            print("Exception while posting: ", e)
            if trail == 3:
                if response is not None:
                    if response.status_code != 200:
                        raise APIMESSetupStatusCodeVAOCVError("Bad API response to MES Setup! Device ID: " + str(monitor_id) + " \n Status Code: " + str(response.status_code))
                else:
                    raise APIMESSetupVAOCVError("Can't Send MES Setup Results via API! Device ID: " + str(monitor_id) + " \n Original Exception: \n" + str(e))
            else:
                time.sleep(1)
                continue
        break
    print("Sent.")
    return True


