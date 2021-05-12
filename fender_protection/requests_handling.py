"""
--------------------------------------------------
filename        : request_handling.py
Name            : Ankit Mishra
Email           : ankitmishra723@gmail.com
Date created    : Feb 28, 2021
Date Modified   : Mar 21, 2021
---------------------------------------------------
"""

from read_config import *
from PIL import Image
import numpy as np
import requests
import os


def save_image(pil_image, name_for_file, camera_id):
    """ ----------------------------------------------
        this function saves the PNG format PIL image obtained from GET link
        for each cam_ID.

        ARGUMENT    :
                camera_id : cam for which Image obtained
                name_for_file : file name (markup/snapshot)
                pil_image: Image in PIL_PNG format

        RETURN      : None
        ------------------------------------------- """
    ref_dir = dir_of_file + '/ship/reference_files/'
    if not os.path.isdir(ref_dir):
        oldmask = os.umask(000)
        os.makedirs(ref_dir, 0o0777)
        os.umask(oldmask)
    pil_image.save(ref_dir+name_for_file+'_'+str(camera_id)+'.jpg')


def get_markup_image(cameraID):
    """ -------------------------------------------------------------
    this function fetches the markup image for the camera ID provided.
    (Uses above 'save image' function to save image at location)
    ARGUMENT    : camera id
    RETURN      : if FETCHED/EXISTS : True , else: False/0
    -------------------------------------------------------------- """
    markup_cam_1 = markup_base_url+str(cameraID)
    file_name = dir_of_file + '/ship/reference_files/markup_'+str(cameraID)+'.jpg'
    if os.path.isfile(file_name):
        # if File Exists no need to fetch
        return True
    else:
        try:
            r = requests.get(markup_cam_1, stream=True).raw
            if r.status == 200:
                img = Image.open(r).convert('RGB')
                save_image(img, name_for_file='markup', camera_id=cameraID)
                return True
            else:
                return False
        except:
            print('ERROR  GET Request FAILED for markup image for Camera ID : %(id)s' % {'id': cameraID})
            return 0


def get_reference_snapshot_image(cameraID):
    """ ----------------------------------------------------------------
    this function fetches the Snapshot image for the camera ID provided.
    (Uses above 'save image' function to save image at location)
    ARGUMENT    : camera id
    RETURN      : if FETCHED/EXISTS : True , else: False/0
    ---------------------------------------------------------------- """
    ref_cam_1 = ref_base_url+str(cameraID)
    file_name = dir_of_file + '/ship/reference_files/fender_' + str(cameraID) + '.jpg'
    if os.path.isfile(file_name):
        # if File Exists no need to fetch
        return True
    else:
        try:
            r = requests.get(ref_cam_1, stream=True).raw
            if r.status == 200:
                img = Image.open(r).convert('RGB')
                save_image(img, name_for_file='fender', camera_id=cameraID)
                return True
            else:
                return False
        except:
            print('ERROR GET Request FAILED for reference image for Camera ID : %(id)s' % {'id': cameraID})
            return 0


def notification_trigger(cameraID, object, status, object_known, image_path):
    """ -----------------------------------------------------------
    this function sends a POST notification to front_end
    ARGUMENTS :
                cameraID    : id of camera for which notification generated --> 1,2,3,4
                object      : 'ship'/ 'fender'/ 'others'
                status      : threat/ no-threat
                object_known: known/ unknown
                image_path  : path to jpg file
    RETURN : NONE
    ----------------------------------------------------------- """

    headers = {'x-access-token': 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJfaWQiOiI2MDQyMzA3ZDZhM2IyMjI2YjgyZmE0MmMiLCJyb2xlIjoiQURNSU4iLCJpYXQiOjE2MTUyODMyMTR9.FPv1C_AELB9UStxx-Jwj5Vax7OgRJr6Rr3sFPRxMa4M'}
    payload = {'title': object, 'status': status,
               'type': object_known, 'siteNumber': cameraID}
    files = {'imageUrl': open(image_path, 'rb')}
    session = requests.Session()
    try:
        session.post(notification_url, headers=headers, data=payload, files=files)
    except:
        print('ERROR : Post notification url failed for cameraID : ' + str(cameraID))


def camera_status_notification_all(all_list, data):
    """ ----------------------------------------------
        this function POSTs notification to frontend for camera status

        ARGUMENT    :
                    all_list : LIST of All camera IDs
                    data     : LIST of active cameras IDs
        RETURN      : None
        ------------------------------------------- """
    false_cameras = np.setdiff1d(all_list, data)
    for cam_id in data:
        cam_post = camera_status_url + str(cam_id)
        try:
            requests.post(cam_post, data={'active': 1})
        except:
            print('ERROR Post Notification camera status for Camera ID : %(id)s' % {'id': cam_id})
            print('NOTE: Potential Error on UI, Reload Application in Browser')
    for cam_id in false_cameras:
        cam_post = camera_status_url + str(cam_id)
        try:
            requests.post(cam_post, data={'active': 0})
        except:
            print('ERROR Post Notification for Camera ID : %(id)s' % {'id': cam_id})
            print('NOTE: Potential Error on UI, Reload Application in Browser')


def camera_status_notification(cam_id, status_code):
    """ ---------------------------------------------------
    this function POSTs notification to frontend for camera status
    (same as above, but takes individual camera_ids instead of list of cam id's)
    cam_id      :   Camera ID
    status_code :   0 if camera is Inactive;
                    1 if camera is Active
    --------------------------------------------------- """
    cam_post = camera_status_url + str(cam_id)
    try:
        requests.post(cam_post, data={'active': status_code})
    except:
        print('ERROR POST camStatus Notification FAILED for Camera ID : %(id)s' % {'id': cam_id})

