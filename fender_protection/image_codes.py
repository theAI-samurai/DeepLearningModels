"""
--------------------------------------------------
filename        : data_validate.py
Name            : Ankit Mishra
Email           : ankitmishra723@gmail.com
Date created    : Feb 28, 2021
Date Modified   : APR 13, 2021
---------------------------------------------------
"""

import cv2
import vlc
import time
import rtsp

stream1 = 'rtsp://admin:DS-2CD206@192.168.29.32'
stream2 = 'rtsp://192.168.29.73:8554/vlcc'


def vlc_stream_object_init(act_cam_ids, conf_cam_data):
    """----------------------------------
    This function takes as input the LIST of Active Cameras
    identified by the 'ACTIVE CAMERA INITIALIZE VLC in DataValidate.py'
    and returns VLC player objects for the active streams.

    Args:
        act_cam_ids     : LIST of Active Camera IDs
        conf_cam_data   : camera id and urls data from config file

    Return:
        vlc_player_object : DICT containing {cam_id : player_object}
    ----------------------------------"""
    vlc_player_object ={}
    for cam in act_cam_ids:
        vlc_player_object.update({cam: vlc.MediaPlayer(conf_cam_data[cam])})
    return vlc_player_object


def vlc_stream_object_init_2(cam_id, conf_cam_data):
    """-------------------------------------------------------
    This function takes as input Active Camera ID
    identified by the 'ACTIVE CAMERA INITIALIZE VLC in DataValidate.py'
    and returns VLC player objects for the Active ID passed as parameter.

    (SAME AS ABOVE FUNC, but instead of LIST of Cam IDs, this fucntion
     take 1 ID at a Time )
    Args:
        act_cam_ids     : Active Camera ID
        conf_cam_data   : camera id and urls data from config file

    Return:
        vlc_player_object : VLC object for ID provided
    -------------------------------------------------------- """
    vlc_obj = vlc.MediaPlayer(conf_cam_data[cam_id])
    return vlc_obj


def vlc_stream_object_init_3(url):
    """-------------------------------------------------------
    This function takes URL as argument and returns VLC OBJECT.

    (SAME AS ABOVE FUNC, but instead of LIST of Active cams and DIct data , this fucntion
     take just the URL )
    Args:
        URL :  URL of camera to create VLC_object

    Return:
        vlc_player_object : VLC object for ID provided
    -------------------------------------------------------- """
    vlc_obj = vlc.MediaPlayer(url)
    return vlc_obj


def read_frames_using_vlc(player, delay_time, cam_id, base_path):
    
    player.play()  # --> play
    if player.is_playing():
        time.sleep(delay_time)
        player.video_take_snapshot(0, base_path + str(cam_id) + '.jpg', 0, 0)
        return True
    else:
        return False


def fender_coordi(path_):
    """-----------------------------------------------------
    this function calculates the (X_min,Y_min) and (X_max, Y_max)
    coordinates of the freehand MARKUP image
    ARGUMENTS:
        path : IMAGE file Path to the MArkup image

    RETURN : MAT format Image, of markup, x1,y1,x2,y2
    ------------------------------------------------------"""
    fender_ref = cv2.imread(path_)  # ---> H=400, W=500
    lst_x = []
    lst_y = []
    for i in range(fender_ref.shape[0]):  # ----> Height
        for j in range(fender_ref.shape[1]):  # ----> Width
            if fender_ref[i, j, 0] > 20:
                lst_x.append(j)
                lst_y.append(i)
    min_x = min(lst_x)
    max_x = max(lst_x)
    min_y = min(lst_y)
    max_y = max(lst_y)

    return fender_ref, min_x,min_y, max_x, max_y


def check_for_overlap(rec1_x1, rec1_y1, rec1_x2, rec1_y2, rec2_x1, rec2_y1, rec2_x2, rec2_y2):
    rectangle_a = {"x1":rec1_x1, "y1":rec1_y1, "x2":rec1_x2,"y2":rec1_y2}   # ----> det
    rectangle_b = {"x1": rec2_x1, "y1":rec2_y1, "x2":rec2_x2,"y2":rec2_y2}  # ----> fender

    # from Left
    if rectangle_a["y1"] < rectangle_b["y2"] and rectangle_a["x2"] > rectangle_b["x1"]:
        print("OVERLAP from LEFT ! ")
        return True
    else:
        print("there is no overlap")
        return False


def overlap_from_left_red_markup(cord_lst, top_right):
    for iter, tup in enumerate(cord_lst):
        w = tup[0]
        h = tup[1]
        W = top_right[0]
        H = top_right[1]
        if w in range(W-30, W+20) and h in range(H-15, H+15):
            print('OVERLAP FROM LEFT')
            return True


def list_marking_coord(mark_img):
    """-----------------------------------------

    Args:
        mark_img: markup image to get list of coordinates for hand markup

    Returns: A list containg tuples in format (W, H) cordinates

    -----------------------------------------"""
    lst = []
    for i in range(mark_img.shape[0]):       # H
        for j in range(mark_img.shape[1]):   # W
            if mark_img[i, j, 2] > 150:
                lst.append((j,i))       # (W,H)
    return lst


def read_frames_rtsp_lib(url, obj_client):
    #with rtsp.Client(rtsp_server_uri = url) as client:
    time.sleep(2)
    #while True:
    _image = obj_client.read(raw=True)
    return _image



