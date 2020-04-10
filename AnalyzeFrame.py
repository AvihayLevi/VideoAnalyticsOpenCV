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

def output_former(ocr_res, room, pat_id, mon_id):
    string_json = json.dumps(ocr_res)
    json_dict = {}
    json_dict["JsonData"] = string_json
    json_dict["MonitorID"] = mon_id
    json_dict["PatientID"] = pat_id
    json_dict["Room"] = room
    json_dict_string = str(json_dict)
    print(json_dict_string)
    output = fix_string(json_dict_string)
    return output


def sockets_output_former(ocr_res, room, pat_id, mon_id):
    json_dict = {}
    json_dict["JsonData"] = ocr_res
    json_dict["MonitorID"] = mon_id
    json_dict["PatientID"] = pat_id
    json_dict["Room"] = room
    output = json.dumps(json_dict)
    print(output)

    return output



def get_digits(img, computervision_client):
    # encodedFrame = cv2.imencode(".jpg", img)[1].tostring()
    
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
                s = re.sub('[^0123456789./]', '', line.text)
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



def AnalyzeFrame(frame, computervision_client, boundries, areas_of_interes, ocrsocket):
    frame = cv2.imdecode(np.frombuffer(frame, np.uint8), -1)
    
    # Find ARuco corners:
    new_corners = detect_markers(frame)
    corners = [(508.0, 403.0), (113.5, 400.0), (572.0, 182.25), (53.75, 172.25)] #mon3
    # corners = [(120.0, 404.0), (532.25, 386.0), (573.0, 124.0), (80.75, 113.5)] #mon4

    # TODO: raise exception if more than one corner wasn't detected:
    fixed_corners = fix_corners(new_corners, corners)
    # pts = order_points(fixed_corners)
    frame = four_point_transform(frame, fixed_corners)

    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,3))
    # frame = cv2.morphologyEx(gray,cv2.MORPH_TOPHAT, gray)

    # TODO: get area_dicts from MOB/DB
    #areas_dict = {'side': [0, 1, 0.7, 0.9], 'bottom': [0.6, 0.9, 0.3, 0.7]} #will be an input later! #monitor 1
    #areas_dict = {'side': [0.1, 0.9, 0.67, 0.92]} #will be an input later! #monitor 3
    #areas_dict = {'low': [0.6, 0.85, 0, 0.5], 'side': [0.1, 0.9, 0.6, 0.9]}
    #areas_dict = {'0': [0.0, 1.0, 0.6, 0.8508771929824561], '1': [0.0, 0.30153846153846153, 0.0, 1.0]}

    areas_dict = areas_of_interes
    areas = create_areas(areas_dict, frame)

    # our output
    readings = {}
    boundings = {}
    i = 0
    for area in areas:
        results = get_digits(cv2.imencode(".jpg", area[0])[1], computervision_client)
        for item in results:
            readings[i] = item[0]
            boundings[i] = transform_coords(item[1], area)
            i = i + 1
    # boundry_dict = {i:[min(x[0],x[6]) -15,max(x[2],x[4]) +15 ,min(x[3],x[1]) -15,max(x[5],x[7]) + 15] for i,x in enumerate(boundings.values())}
    # boundry_temp_mon32 = {0: ((471.0, 129), (516.0, 165)), 1: ((464.0, 170), (533.0, 213)), 2: ((469.0, 222), (541.0, 244)), 3: ((469.0, 272), (508.0, 306)), 4: ((471.0, 315), (505.0, 351))}
    
    # MES-Setup inpurt by hand. TODO: get from API in the begining
    # boundry_temp_mon32 = {0: ((132, 316.0), (246, 345.0)), 1: ((449.0, 172.0), (509.0, 221.0)), 2: ((439.0, 230.0), (485.0, 269.0)), 3: ((435.0, 271.0), (483.0, 312.0))}
    # helka_dictionary = {0: [374.0, 18.0, 429.0, 18.0, 429.0, 51.0, 374.0, 52.0], 1: [370.0, 59.0, 419.0, 59.0, 417.0, 93.0, 369.0, 92.0], 2: [358.0, 96.0, 419.0, 92.0, 420.0, 128.0, 358.0, 122.0], 3: [34.0, 132.0, 197.0, 134.0, 196.0, 163.0, 33.0, 160.0]} #mon3
    # boundry_temp_mon32 = {0: ((365.0, 26.0), (379.0, 39.0)), 1: ((380.0, 17.0), (437.0, 53.0)), 2: ((375.0, 61.0), (425.0, 95.0)), 3: ((377.0, 96.0), (429.0, 131.0)), 4: ((58.0, 140.0), (160.0, 166.0)), 5: ((93.0, 164.0), (140.0, 177.0))}

    # boundry_temp_mon32 = {0: ((342.0, 9.0), (359.0, 16.0)), 1: ((398.0, 7.0), (470.0, 16.0)), 2: ((395.0, 47.0), (420.0, 55.0)), 3: ((407.0, 54.0), (419.0, 65.0)), 4: ((421.0, 38.0), (484.0, 84.0)), 5: ((400.0, 101.0), (420.0, 111.0)), 6: ((404.0, 110.0), (418.0, 122.0)), 7: ((422.0, 99.0), (472.0, 148.0)), 8: ((404.0, 159.0), (420.0, 170.0)), 9: ((416.0, 149.0), (477.0, 198.0)), 10: ((80.0, 9.0), (104.0, 18.0)), 11: ((331.0, 7.0), (358.0, 16.0)), 12: ((399.0, 7.0), (469.0, 16.0)), 13: ((40.0, 49.0), (64.0, 58.0)), 14: ((394.0, 46.0), (420.0, 57.0)), 15: ((86.0, 70.0), (108.0, 77.0)), 16: ((424.0, 36.0), (488.0, 87.0))}
    # temp_mon = {k:[[v[0],v[1]],[v[4],v[5]]] for k,v in helka_dictionary.items()} #translate dic to normal version

    output = create_bounded_output(readings, boundings, transform_boundries(boundries), 3)
    # output = create_bounded_output(readings, boundings, temp_mon, 3)
    # print(output)

    
    # TODO: get as input, when Shany's team is ready
    pat_id = "200465524"
    room = "13"
    mon_id = "90210"
    mon_id = "3333"
    json_to_socket = sockets_output_former(output, room, pat_id, mon_id)
    ocrsocket.emit('data', json_to_socket)
    return


    json_string_fin = output_former(output, room, pat_id, mon_id)
    print(json_string_fin)
    url = "http://rstreamapp.azurewebsites.net/api/InsertMonitorData"
    headers={'Content-type':'application/json', 'Accept':'application/json'}
    response = requests.post(url, data=json_string_fin, headers=headers)

    # TODO: sanity check results (charecters etc.) and send them to somewhere
    return


