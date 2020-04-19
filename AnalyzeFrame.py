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
import ImagePreprocess
import base64
from Levenshtein import distance as Ldistance
import operator
# from termcolor import colored
from ExceptionsModule import *


def distance(p1, p2):
    xdis = (p1[0]-p2[0])**2
    ydis = (p1[1]-p2[1])**2
    return math.sqrt(xdis + ydis)


def fix_corners(current_corners, last_corners):
    fixed_corners = [[-1,-1], [-1,-1], [-1,-1], [-1,-1]]
    for cp in current_corners:
        min_dis = 500000000
        curr_min = 0
        for i in range(4):
            dis = distance(last_corners[i], cp)
            if dis < min_dis:
                min_dis = dis
                curr_min = i
        fixed_corners[curr_min] = cp
    for i in range(4):
        if fixed_corners[i]==[-1,-1]:
            fixed_corners[i] = last_corners[i]
    return fixed_corners


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
        
    #cv2.imshow("Window", clone)
    #cv2.waitKey(1)
    #time.sleep(3)
    return fixed_corners


def create_areas(area_dict, img):
    s = img.shape
    height, width = s[0], s[1]
    areas = []
    for key, value in area_dict.items():
        hmin, hmax, wmin, wmax = value
        hmin *= height
        hmax *= height
        wmin *= width
        wmax *= width
        new_area = [img[math.ceil(hmin):math.ceil(hmax), math.ceil(wmin):math.ceil(wmax)], hmin, wmin]
        areas.append(new_area)
    return areas

def transform_coords(coords, area):
    fixed_coords = []
    for j in range(8):
        if j%2==0:
            fixed_coords.append(coords[j] + area[2])
        else:
            fixed_coords.append(coords[j] + area[1])
    return fixed_coords
    
def transform_boundries(boundry_dict):
    fixed_dict = {}
    for key, value in boundry_dict.items():
        fixed_value = [value[0][0]-5, value[1][0]+5, value[0][1]-5, value[1][1]+5]
        fixed_dict[key] = fixed_value
    return fixed_dict
    

def create_bounded_output(readings, boundings, boundries, method = 3):
    output_dict = {}
    for key in boundries.keys():
        for i in range(len(readings)):
            if method == 1 : # area contain
                if check_boundry(boundings[i], boundries[key]): #c heck if temp rect in bigger rect
                    output_dict[key] = readings[i]
            elif method == 2: # area intersection
                if check_overlap(boundings[i], boundries[key]):  # using precentage of interseection, greater than 0.7 is true!
                    output_dict[key] = readings[i]
            elif method == 3: # dot and contain
                if check_dot(boundings[i], boundries[key]):  # rectangle containing center point
                    output_dict[key] = readings[i]
        if key not in output_dict.keys():
            output_dict[key] = "N/A"
            # output_dict[key] = None
    return output_dict

"""
def create_bounded_output(readings, boundings, boundries):
    output_dict = {}
    for key in boundries.keys():
        for i in range(len(readings)):
            if check_boundry(boundings[i], boundries[key]):
                output_dict[key] = readings[i]
        if key not in output_dict.keys():
            output_dict[key] = "N/A"
    return output_dict
"""


def check_overlap(temp_bounding, hard_bounding):
    a = [hard_bounding[0][0],hard_bounding[0][1],hard_bounding[1][0],hard_bounding[1][1]]
    b = [temp_bounding[0],temp_bounding[1],temp_bounding[4],temp_bounding[5]]
    total_area = (a[2] - a[0]) * (a[3] - a[1])
    dx = min(a[2], b[2]) - max(a[0], b[0])
    dy = min(a[3], b[3]) - max(a[1], b[1])
    if (dx>=0) and (dy>=0):
        if float((dx * dy) / total_area) > 0.7:
            return True
    return False

    
def check_dot(temp_bounding, hard_bounding):
    # center_dot = (hard_bounding[0][0] + (hard_bounding[1][0] - hard_bounding[0][0])/ 2 , hard_bounding[0][1] + (hard_bounding[1][1] - hard_bounding[0][1])/ 2)
    center_dot = (hard_bounding[0] + (hard_bounding[1] - hard_bounding[0])/ 2 , hard_bounding[2] + (hard_bounding[3] - hard_bounding[2])/ 2)
    if center_dot[0] >= temp_bounding[0] and center_dot[0] <= temp_bounding[4] and center_dot[1] >= temp_bounding[1] and center_dot[1] <= temp_bounding[5]:
        return True
    return False


def check_boundry(bounding, boundry):
    output = bounding[0]>=boundry[0]
    output = output and (bounding[6]>=boundry[0])
    output = output and (bounding[2]<=boundry[1])
    output = output and (bounding[4]<=boundry[1])
    output = output and (bounding[1]>=boundry[2])
    output = output and (bounding[3]>=boundry[2])
    output = output and (bounding[5]<=boundry[3])
    output = output and (bounding[7]<=boundry[3])
    return output 


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


def sockets_output_former(ocr_res, mon_id):
    json_dict = {}
    json_dict["JsonData"] = ocr_res
    json_dict["DeviceID"] = mon_id
    json_dict["deviceType"] = os.getenv("DEVICE_TYPE")
    json_dict["gilayon_num"] = os.getenv("GILAYON_NUM")
    output = json.dumps(json_dict)
    print(output)
    return output


def get_digits(img, computervision_client, mode="digits"):
    # encodedFrame = cv2.imencode(".jpg", img)[1].tostring()
    # tmp_frame = cv2.imdecode(np.frombuffer(img, np.uint8), -1)
    # cv2.imshow("image", tmp_frame)
    # cv2.waitKey(0)
    recognize_printed_results = computervision_client.batch_read_file_in_stream(io.BytesIO(img), raw = True)
    # Reading OCR results
    operation_location_remote = recognize_printed_results.headers["Operation-Location"]
    operation_id = operation_location_remote.split("/")[-1]
    while True:
        get_printed_text_results = computervision_client.get_read_operation_result(operation_id)
        if get_printed_text_results.status not in ['NotStarted', 'Running']:
                break
        time.sleep(0.1)
    
    tmp_frame = cv2.imdecode(np.frombuffer(img, np.uint8), -1)
    results = []
    text_flag = False
    show_frame_flag = False
    if get_printed_text_results.status == TextOperationStatusCodes.succeeded:
        for text_result in get_printed_text_results.recognition_results:
            for line in text_result.lines:
                # print(line.text, line.bounding_box)
                if mode == "digits":
                    s = re.sub('[^0123456789./:]', '', line.text)
                    if s != "":
                        if s[0] == ".":
                            s = s[1:]
                        s = s.rstrip(".")
                        text_flag = True
                        cv2.rectangle(tmp_frame, (int(line.bounding_box[0]), int(line.bounding_box[1])), (int(line.bounding_box[4]), int(line.bounding_box[5])), (255,0,0), 2)
                        cv2.putText(tmp_frame,s,(int(line.bounding_box[0])-5, int(line.bounding_box[1])-5),cv2.FONT_HERSHEY_COMPLEX,0.3,(0,0,0),1)
                        results.append((s, line.bounding_box))
                    else:
                        continue
                if mode == "modes":
                    # s = re.sub('[^0123456789./:]', '', line.text)
                    s = line.text
                    if s != "":
                        if s[0] == ".":
                            s = s[1:]
                        s = s.rstrip(".")
                        text_flag = True
                        cv2.rectangle(tmp_frame, (int(line.bounding_box[0]), int(line.bounding_box[1])), (int(line.bounding_box[4]), int(line.bounding_box[5])), (255,0,0), 2)
                        cv2.putText(tmp_frame,s,(int(line.bounding_box[0])-5, int(line.bounding_box[1])-5),cv2.FONT_HERSHEY_COMPLEX,0.3,(0,0,0),1)
                        results.append((s, line.bounding_box))
                    else:
                        continue
        if text_flag and show_frame_flag:
            cv2.imshow("image", tmp_frame)
            cv2.waitKey(0)
    return(results)


def get_ala_digits(img):
    # tmp_frame = cv2.imdecode(np.frombuffer(img, np.uint8), -1)
    # cv2.imshow("image", tmp_frame)
    # cv2.waitKey(0)
    enc_img = base64.b64encode(img)
    data = {"image": str(enc_img, 'utf-8')}
    # data['screenCorners'] = {"left-top": [1, 1], "right-top": [600, 1], "bottom-left": [1, 550], "bottom-right": [600, 550]}
    res = requests.post("http://127.0.0.1:8088/v1/run_ocr", json=data)
    res_str = json.loads(res.text)
    results_dicts = [{"text": x["value"], "coords": [x["left"], x["top"], x["right"], x["top"], x["right"], x["bottom"], x["left"], x["bottom"]]} for x in res_str]
    # print(results_dicts)
    filtered_results = []
    for item in results_dicts:
        s = re.sub('[^0123456789./:]', '', item['text'])
        if s != "":
            if s[0] == ".":
                s = s[1:]
            s = s.rstrip(".")
            filtered_results.append((s, item['coords']))
        else:
            continue
    # print("--------------------")
    # print(filtered_results)
    return(filtered_results)


def boundries_to_areas(boundries, hight, width):
    areas = {}
    i = 0
    for k,v in boundries.items():
        x1, y1, x2, y2 = v[0][0], v[0][1], v[1][0], v[1][1]
        top, bottom, left, right = y1/hight, y2/hight, x1/width, x2/width
        areas[i] = [top, bottom, left, right]
        i = i + 1
    return areas


def getVelaModeAndWarning(img, marker_corners, computervision_client):  
    s = img.shape
    height, width = s[0], s[1]
    two_top_markers = sorted(marker_corners, key=lambda tup: tup[1])[:2] # get 2 top markers
    left_corner, right_corner = sorted(two_top_markers, key=lambda tup: tup[0])[0] , sorted(two_top_markers, key=lambda tup: tup[0])[1] #get left and right
    
    x_range = [left_corner[0] / width , right_corner[0] / width ] # x range in form of [left_x, right_x]
    y_range = [0, (max(left_corner[1],right_corner[1]) / height) + 0.1] # y range in form of [0, y_top_corners + 10% of frame height
    experiment_area = {"mode_warning": [y_range[0], y_range[1], x_range[0], x_range[1]]}
    # print('experiment_area:', experiment_area)
    
    #top_area = {"mode": [0, 0.35, 0, 0.4], "warning": [0, 0.35, 0.55, 1]}
    top_area = experiment_area    
    areas = create_areas(top_area, img)
    readings = {}
    boundings = {}
    i = 0
    for area in areas:
        results = get_digits(cv2.imencode(".jpg", area[0])[1], computervision_client, "modes")
        # results = get_ala_digits(cv2.imencode(".jpg", area[0])[1])
        for item in results:
            readings[i] = item[0]
            boundings[i] = transform_coords(item[1], area)
            i = i + 1
    strings_found = [v.lower().replace(" ", "") for v in readings.values()]
    # print(strings_found)

    # check mode:
    known_modes = ["volumesimv","prvca/c"]
    min_mode_distance = 99
    found_mode = "UNKNOWN MODE"
    for item in strings_found:
        for mode in known_modes:
            tmp_mode_dis = Ldistance(item, mode)
            # print(item, mode, tmp_mode_dis)
            if tmp_mode_dis < min_mode_distance and tmp_mode_dis <= len(mode)/2:
                min_mode_distance = tmp_mode_dis
                found_mode = mode
                # print("update")
    # print(found_mode, min_mode_distance)

    # check warning:
    known_warnings = ["highpip", "lowpip", "circuitdisconnect", "lowve", "apneainterval", "o2inletlow", "checkfilter"]
    min_warning_distance = 99
    found_warning = "no warning"
    for item in strings_found:
        for warning in known_warnings:
            tmp_warning_dis = Ldistance(item, warning)
            # print(item, warning, tmp_warning_dis)
            if tmp_warning_dis < min_warning_distance and tmp_warning_dis <= len(warning)/2:
                min_warning_distance = tmp_warning_dis
                found_warning = warning
                # print("update")
    # print(found_warning, min_warning_distance)
    return found_mode, found_warning


def fix_readings(readings_dic):
    for name, read in readings_dic.items(): 
        read, rlen = str(read), len(read) 
        if name == 'IBP':
            if rlen >= 6: # XXX/XXX or XXX/XX
                if read[3] in [7,1] : # mistake: 120780 -> 120/80, 1407110 -> 140/110 
                    readings_dic[name] = read[:2] + '/' + read[4:]
            elif rlen == 5: # XX/XX
                if read[2] in [7,1] : # mistake: 90760 -> 90/60 
                    readings_dic[name] = read[:1] + '/' + read[3:]
        elif name == 'HR': 
            continue 
        elif name == 'temp': 
            if rlen >= 3:
                if read[2] not in [',','.']: # XXX 
                    readings_dic[name] = read[:2] + '.' + read[2:] # mistake: 307 -> 30.7
        elif name == 'IE':     
            if rlen >= 2:
                if read[0] == 1 and read[-1] == 1:
                    continue #we dont know what to do
                if read[0] == '1': # numbers in format 1:##
                    if read[1] != ':':
                        read = read[:1] + ':' + read[1:] # mistake: 133 -> 1:33
                    if read[-2] != '.':
                        read = read[:-1] + '.' + read[-1:] # mistake: 1:33 -> 1:3.3
                    readings_dic[name] = read
                elif read[-1] == '1': # numbers in format ##:1
                    if read[-2] != ':':
                        read = read[:-1] + ':' + read[-1:] # mistake: 331 -> 33:1
                    if read[1] != '.':
                        read = read[:1] + '.' + read[1:] # mistake: 33:1 -> 3.3:1
                    readings_dic[name] = read
    return readings_dic
        
def isNumber(x):
    try:
        return bool(0 == float(x)*0)
    except:
        return False

def create_hist_dict(result_list):
    hist_dict = {}
    for res in result_list:
        for (key, val) in res.items():
            if key in hist_dict.keys():
                if val in hist_dict[key].keys():
                    hist_dict[key][val] += 1
                else:
                    hist_dict[key][val] = 1
            else:
                new_dict = {}
                new_dict[val] = 1
                hist_dict[key] = new_dict
    return(hist_dict)

def histogram_out(result_list, key, k):
    hist_dict = create_hist_dict(result_list)
    best = max(hist_dict[key].items(), key=operator.itemgetter(1))
    if (best[1] >= k and best[0] != "N/A"): 
        return best[0]
    else:
        return None

def fix_output(output, results_list, k):
    new_output = output.copy()
    for key, val in output.items():
        new_val = val
        # print(isNumber(val))
        for i in range(len(results_list)):
            last_val = "N/A"
            if results_list[-1-i][key] != "N/A":
                last_val = results_list[-1-i][key]
                last_val_key = i
                break
        if val == "N/A":
            new_val = last_val
        else:
            hist_out = histogram_out(results_list, key, k)
            if hist_out:
                new_val = hist_out
            elif isNumber(val) and isNumber(last_val): 
                if abs(float(val)-float(last_val)) > 0.5 * float(val):
                    new_val = last_val
                    older_val = None
                    for i in range(len(results_list)-last_val_key-1):
                        if results_list[-2-last_val_key-i][key] != "N/A":
                            older_val = results_list[-2-last_val_key-i][key]
                            break
                        older_val = "N/A"
                    if isNumber(older_val):
                        if abs(float(last_val)-float(older_val)) > 0.5 * float(last_val):
                            new_val = val
        new_output[key] = new_val
    return new_output
            

def generic_errors(output, last_results):
    miss_count = 0
    for key, value in output.items():
        if value == "N/A":
            miss_count += 1
    if miss_count>0:
        if miss_count>=0.5*len(output):
            if miss_count>=0.75*len(output):
                print("Fatal error, almost no data in format")
            else:
                print("Error, most data not in format","red")
        else:
            print("Mild error, some missing fields","yellow")
    missing_dict = {}
    amount_of_results = len(last_results)
    for res in last_results:
        for key, value in res.items():
            if value is None:
                if key in missing_dict.keys():
                    missing_dict[key] += 1
                else:
                    missing_dict[key] = 1
    completely_missing_vals = []
    for key, value in missing_dict.items():
        if value == amount_of_results:
            completely_missing_vals.append(key)
    if len(completely_missing_vals) > 0:
        print("Error, ", completely_missing_vals, "are missing from the last ", amount_of_results, "frames!")
    return


def AnalyzeFrame(orig_frame, computervision_client, boundries, areas_of_interes, ocrsocket, last_four_corners, old_results_list):
    frame = cv2.imdecode(np.frombuffer(orig_frame, np.uint8), -1)
    orig_frame = cv2.imdecode(np.frombuffer(orig_frame, np.uint8), -1)
    
    # Find ARuco corners:
    new_corners = detect_markers(frame)
    old_corners = last_four_corners

    if len(new_corners) < 4:
        fixed_corners = fix_corners(new_corners, old_corners)
    elif len(new_corners) > 4:
        # print("too much markers - get old ones")
        fixed_corners = old_corners
    else:
        old_corners = new_corners
        fixed_corners = new_corners
    frame = four_point_transform(frame, fixed_corners)
    
    
    device_type = os.getenv("DEVICE_TYPE")
    if device_type == "respiration":
        found_mode, found_warning = getVelaModeAndWarning(orig_frame, fixed_corners, computervision_client)
        if found_mode != "volumesimv":
            print("UNKNOWN MODE DETECTED!!")
            # TODO: try again and raise exception
        if found_warning != "no warning":
            print("RESPIRATION WARNING:  ", found_warning)

    # Pre-Process: TODO: Integrate Gidi's module
    # frame = ImagePreprocess.unsharp(frame)
    frame = ImagePreprocess.filter2d(frame)
    
    areas_dict = areas_of_interes
    # areas_dict = boundries_to_areas(boundries, frame.shape[0], frame.shape[1])
    areas = create_areas(areas_dict, frame)
    
    # our output
    readings = {}
    boundings = {}
    i = 0
    CV_MODEL = os.getenv("CV_MODEL")
    for area in areas:
        try:
            if CV_MODEL == "MSOCR":
                results = get_digits(cv2.imencode(".jpg", area[0])[1], computervision_client, "digits")
            # elif CV_MODEL == "ALA":
                # results = get_ala_digits(cv2.imencode(".jpg", area[0])[1])
            else:
                raise Exception("UNRECOGNIZED MODEL")
        except Exception as e:
            print(e)
            continue
        for item in results:
            readings[i] = item[0]
            boundings[i] = transform_coords(item[1], area)
            i = i + 1
    
    output = create_bounded_output(readings, boundings, transform_boundries(boundries), 3)
    # IMPORTANT: when needed - comment-out next line and change get_boundries accordingly
    # Fix Readings based on known measures format:
    # output = fix_readings(output)
    
    """
    k_frames_to_save = 5
    if len(old_results_list) < k_frames_to_save:
        old_results_list.append(output) #build "window" of 15 frames
        fixed_result = False
    #return last_results #just append, less than 15 frames seen
    else:
        old_results_list.pop(0) #remove oldest frame from list
        fixed_result = fix_output(output, old_results_list, k_frames_to_save-1) 
        old_results_list.append(output) #add our current result
        output = fixed_result
    """
    generic_errors(output, old_results_list)

    # print(output)

    # TODO: sanity check results (charecters etc.) and send them to somewhere
    # TODO: get as input, when Shany's team is ready
    monitor_id = os.getenv("DEVICE_ID")
    json_to_socket = sockets_output_former(output, monitor_id)
    ocrsocket.emit('data', json_to_socket)

    FRAME_DELAY = os.getenv("FRAME_DELAY")
    time.sleep(float(FRAME_DELAY))
    return old_corners, old_results_list
