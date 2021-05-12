from read_config import *
from data_validate import *
from darknet import *
from image_codes import *
import time
import math


active_cams = []
VLC_PLAYER_OBJECT = {}
start_timer = 0
ctr = 0

fgbg1 = cv2.bgsegm.createBackgroundSubtractorMOG()

# loading Deep learning model
net = model_load(cfgPath=dir_of_file + '/ship/cfg/yolov3_full_ship.cfg',
                 wgtPath=dir_of_file + '/ship/cfg/yolov3_full_ship_2000.weights')

obj_detect = ObjectDetection(dataPath=dir_of_file + '/ship/cfg/ship.data', netwrk=net, camID=0)

if os.path.exists(dir_of_file + '/ship/unknown_objects'):
    pass
else:
    os.mkdir(dir_of_file + '/ship/unknown_objects', mode=0o777)


def main_program(cam_, cam_url):
    restart_status = False
    while True:

        global active_cams
        global VLC_PLAYER_OBJECT
        global start_timer
        global fgbg1
        global ctr

        if cam_ not in active_cams:                                     # if camID is inactive initialize the camera
            is_active = active_stream_initialize_vlc(cam_, cam_url)     # checks if frame available
            if is_active:                                               # if True
                active_cams.append(cam_)
                vlc_obj = vlc_stream_object_init_3(cam_url)
                VLC_PLAYER_OBJECT.update({cam_: vlc_obj})
                del vlc_obj
        else:                                                           # if camID is active
            fender_markup_image, minx, miny, maxx, maxy = fender_coordi(path_=dir_of_file + '/ship/reference_files/markup_' + (cam_) + '.jpg')

            while not restart_status:

                read = read_frames_using_vlc(player=VLC_PLAYER_OBJECT[cam_], delay_time=1,
                                             cam_id=cam_, base_path=dir_of_file + '/ship/reference_files/')
                if read:
                    start_timer = 0                                     # RESET timer = 0 as frame was received
                    frame_path = dir_of_file + '/ship/reference_files/' + str(cam_) + '.jpg'  # frame path
                    markup_img_pa = dir_of_file + '/ship/reference_files/markup_' + str(cam_) + '.jpg'  # markup_path
                    new_dim = height_width_validate(frame_path, markup_img_pa)
                    fender_markup_image = cv2.resize(cv2.imread(markup_img_pa), new_dim)            # Read markup image and resize
                    lst_markup_coord = list_marking_coord(fender_markup_image)                      # List of Markup Coordinates ie RED line coordinates in markup image
                    res = obj_detect.detect(frame_path.encode('ascii'))  # calling detection on the saved snapshot

                    for i in range(len(res)):
                        cls, confi, coordi = res[i]
                        if cls == b'boat':
                            xmi, ymi, xma, yma = bbox2points(coordi)                                # coordinate of object in image
                            # overlap = check_for_overlap(xmi, ymi, xma, yma, minx, miny, maxx, maxy)               # overlap using rectangular box
                            overlap = overlap_from_left_red_markup(cord_lst=lst_markup_coord, top_right=(xma, ymi))
                            if overlap:
                                image = cv2.resize(cv2.imread(frame_path), new_dim)                                                                     # snapshot image file read
                                fgmask = fgbg1.apply(image)
                                fgmask = cv2.rectangle(fgmask, (math.floor(xmi), math.floor(ymi)), (math.floor(xma), math.floor(yma)), (0, 0, 0), -1)  # masking the detected coordinate
                                image = cv2.rectangle(image, (math.floor(xmi), math.floor(ymi)), (math.floor(xma), math.floor(yma)), (255, 255, 0), 2)  # Rectange marking for boat coordinates
                                image = cv2.rectangle(image, (minx, miny), (maxx, maxy), (0, 0, 0), 2)                                                  # Rectangle marking for fender coordinates
                                image_ = cv2.addWeighted(image, 1, fender_markup_image, 1, 0)                                                           # overlaps Fender Markups on Snapshot image
                                contours, _ = cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                                for contour in contours:
                                    area = cv2.contourArea(contour)
                                    if area > 100:
                                        cv2.drawContours(image_, contours, -1, (0,0,0), 0)
                                        curr_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
                                        save_unknown_path = dir_of_file + '/ship/unknown_objects/unk_overlap_'+str(cam_)+'_'+ curr_time + str(ctr) + '.jpg'
                                        ctr += 1
                                        cv2.imwrite(save_unknown_path, image_)
                                        #time.sleep(1)
                                        notification_trigger(cameraID=cam_, object='unknown_countours', status='Unknown'
                                                             , object_known='UnKnown', image_path=save_unknown_path)
                                save_detect_path = dir_of_file + '/ship/reference_files/detect_' + str(cam_) + '.jpg'
                                cv2.imwrite(save_detect_path, image_)
                                #time.sleep(1)
                                notification_trigger(cameraID=cam_, object=cls, status='Threat',
                                                     object_known='Known', image_path=save_detect_path)
                            else:
                                image = cv2.resize(cv2.imread(frame_path), new_dim)  # snapshot image file read
                                fgmask = fgbg1.apply(image)
                                fgmask = cv2.rectangle(fgmask, (math.floor(xmi), math.floor(ymi)),(math.floor(xma), math.floor(yma)), (0, 0, 0),-1)  # masking the detected coordinate
                                cv2.imwrite('D:/darknet_fender_protection/ship/out/'+str(ctr)+'.jpg',fgmask)
                                ctr=ctr+1
                                #time.sleep(1)
                                contours, _ = cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                                for contour in contours:
                                    area = cv2.contourArea(contour)
                                    if area > 100:
                                        cv2.drawContours(image, contours, -1, (0,0,0), 0)
                                        imge_ = cv2.addWeighted(image, 1, fender_markup_image, 1, 0)
                                        curr_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
                                        save_unknown_path = dir_of_file + '/ship/unknown_objects/unk_'+str(cam_)+'_'+ curr_time + str(ctr) + '.jpg'
                                        ctr += 1
                                        cv2.imwrite(save_unknown_path, imge_)
                                        #time.sleep(1)
                                        notification_trigger(cameraID=cam_, object='unknown_countours', status='Unknown'
                                                             , object_known='UnKnown', image_path=save_unknown_path)

            else:
                    start_timer = time.time()                   # start timer if Frame not Recieved
                    if start_timer != 0:
                        if time.time() - start_timer > 1200:    # check if timer is active for 1200 sec or 20 mins
                            active_cams.remove(cam_)            # delete cameraID from Active cam list
                            del VLC_PLAYER_OBJECT[cam_]         # del VLC object of cameraID
                            restart_status = True               # Status : True to reacquire objects for camID
                            camera_status_notification(cam_id=cam_, status_code=0)      # notification trigger to add
                            #"""






