
import cv2
import os
from os.path import dirname, realpath
dir_of_file = dirname(realpath(__file__))
os.chdir(dir_of_file)

imges = []
path_ =dir_of_file + '/ship/test_data/'
path_ ='D:/1/ships/'

for f in os.listdir(path_) :
    if f.endswith('.jpg'):
        temp = os.path.join(path_, f)
        if os.path.exists(temp):
            img = cv2.imread(temp)
            imges.append(img)
            #res = ship_det.detect(temp.encode('ascii'))
            #print(res, type(res))
            #for e in range(len(res)):
            #    print('**************')
        #else:
        #    print("Image path does not exsist")

video=cv2.VideoWriter(os.path.join(path_, 'video.mp4'),-1,1,(imges[1].shape[1], imges[1].shape[0]))

for j in range(len(imges)):
    video.write(imges[j])

cv2.destroyAllWindows()
video.release()

import cv2
path  = r'D:\my_repos\darknet_fender_protection\ship\test_data\wassup.mp4'
cap = cv2.VideoCapture(path)
ctr = 0
while cap.isOpened():
    ret, frame = cap.read()
    print(ret)
    if ret:
        cv2.imwrite("temp_frame"+str(ctr)+".jpg", frame)
        ctr = ctr+1
    else:
        break

# overlay of markup image on frame captured TEST
import cv2
main_img = cv2.resize(cv2.imread(r'D:\darknet_fender_protection\ship\reference_files\5.jpg'),(960,540))
markup_image = cv2.resize(cv2.imread(r'D:\darknet_fender_protection\ship\reference_files\markup_5.jpg'),(960,540))
print(main_img.shape, markup_image.shape)
main_img = cv2.addWeighted(main_img, 1, markup_image, 1, 0)
save_detect_path = r'D:\darknet_fender_protection/ship/reference_files/detect_5'+'.jpg'
cv2.imwrite(save_detect_path, main_img)

cv2.waitKey(0)
cv2.destroyAllWindows()



# height width validate

from data_validate import *
import cv2

main_img = r'D:\darknet_fender_protection\ship\reference_files\5.jpg'
markup_image = r'D:\darknet_fender_protection\ship\reference_files\markup_5.jpg'

height_width_validate(main_img, markup_image)

# ----------baground subtraction
import cv2

backSub_1 = cv2.createBackgroundSubtractorMOG2()
backSub_2 = cv2.createBackgroundSubtractorKNN()


ref_frame_path = r'D:\darknet_fender_protection\ship\reference_files\fender_8.jpg'
new_frame = r'D:\darknet_fender_protection\ship\reference_files\5.jpg'



# ------------------------------------- RED MARKUP TEST -----------------

import cv2
from data_validate import *
from image_codes import *

# STEP 1 : Read makrup image and get  top-Left and bottom right coorinates  to draw Rectangle of markup
# STEP 2: read boat image
# STEP 3:  Find right resolution size and Resize image
# STEP 4: Overlap markup on Boat image
# Step 5: Draw rectangle of MArkup image
# STEP 6: Rectangle of Boat coordinats
# -------- checking overlap ---------------
# STEP 7:
markup = r'D:\darknet_fender_protection\ship\reference_files\markup_9.jpg'
fender_markup_image, minx, miny, maxx, maxy = fender_coordi(markup)
boat = r'D:\darknet_fender_protection\ship\reference_files\4.jpg'

newdim = height_width_validate(boat, markup)

bat = cv2.resize(cv2.imread(boat), newdim)          # boat
mrk = cv2.resize(fender_markup_image, newdim)       # markup

image_ = cv2.addWeighted(bat, 1, mrk, 1, 0)
image_ = cv2.rectangle(image_, (minx, miny), (maxx, maxy), (0, 0, 0), 2)
c1 = (189, 269)  # --> 189:W, 269:H in a image where W = 960, H = 540------- to access img[H,W]
c2 = (504, 500)     # (W,H)
c3 = (504, 269)
c4 = (189, 500)

image_ = cv2.rectangle(image_, c1, c2, (0, 255, 0), 2)

image_ = cv2.circle(image_, c1, radius=1, color=(255, 0, 255), thickness=4)  # (w,h)
image_ = cv2.circle(image_, c2, radius=1, color=(255, 0, 255), thickness=4)
image_ = cv2.circle(image_, c3, radius=1, color=(255, 0, 255), thickness=4)
image_ = cv2.circle(image_, c4, radius=1, color=(255, 0, 255), thickness=4)

#cv2.imshow('jfbf', image_)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

#cv2.imwrite(r'D:\darknet_fender_protection\ship\reference_files\overlaped.jpg',image_)

lst = list_marking_coord(mrk)

for iter, tup in enumerate(lst):
    w = tup[0]
    h = tup[1]
    W = c3[0]
    H = c3[1]
    if w in range(W - 130, W + 20) and h in range(H - 15, H + 15):
        image_ = cv2.circle(image_, (w,h), radius=1, color=(255, 0, 255), thickness=4)  # (w,h)
        cv2.imshow('jfbf', image_)
        cv2.waitKey(0)
        cv2.destroyAllWindows()



# ---------------------------------------------------------------------
import requests
from read_config import *
from PIL import Image
import cv2

markup_cam_1 = markup_base_url+str(5)
for i in range(6):
    try:
        r = requests.get(markup_cam_1, stream=True).raw
    except:
        r=0
    print(r.status)
    i+=1
    if r.status ==200:
        img = Image.open(r).convert('RGB')
        img.save('D:/darknet_fender_protection/ship/reference_files/test'+str(i)+'.jpg')




import cv2
import os
import pathlib

path = r'D:\darknet_fender_protection\ship\test_data'
markup = r'D:\darknet_fender_protection\ship\reference_files\markup_9.jpg'
mark = cv2.imread(markup)
#"""
if os.path.exists(path + '/out'):
    pass
else:
    os.mkdir(path + '/out', mode=0o777)
#"""
out_path = path +'/out/'
ctr = 0

#"""
cap = cv2.VideoCapture(path+'/video2.mp4')
while cap.isOpened():
    ret, img = cap.read()
    try:
        img = cv2.resize(img,(960,540))
        cv2.imwrite(out_path+str(ctr)+'.jpg', img)
    except:
        break
    ctr +=1
#"""

# ------------------------------------------ BACKGROUND SUBTRACTION ------------------------

import cv2
import numpy as np
path = r'D:\darknet_fender_protection\ship\test_data'
detect_coor_1 = (125, 295)
detect_coor_4 = (400, 435)

cap = cv2.VideoCapture(path+'/test3.mp4')
fgbg1 = cv2.bgsegm.createBackgroundSubtractorMOG()
#fgbg2 = cv2.createBackgroundSubtractorMOG2()
#fgbg3 = cv2.bgsegm.createBackgroundSubtractorGMG()
lst = []
ctr = 0
while(cap.isOpened()):# and ctr<2:
#if ctr < 2:
    # read frames
    ret, img = cap.read()
    fgmask1 = fgbg1.apply(img)
    fgmask1 = cv2.rectangle(fgmask1, detect_coor_1, detect_coor_4, (0, 0, 0), -1)  # masking the detected coordinate
    #edge = cv2.Canny(fgmask1, 10, 150)
    #print(edge.shape, fgmask1.shape)
    cont, _ = cv2.findContours(fgmask1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    ctr+=1
    #print(len(cont))
    for e in cont:
        area = cv2.contourArea(e)
        if area != 0:
            lst.append(area)
            print(area)
        if area > 7000:
            print(area)
            cv2.drawContours(img, cont, -1, (0, 255, 0), 0)
    cv2.imshow('Original', img)
    cv2.imshow('MOG', fgmask1)
    #cv2.imshow('edge', edge)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()

# ------------------------------------------- SUBTRACTION ---------------------------------
import cv2
import numpy as np
import os

detect_coor_1 = (125, 295)
detect_coor_2 = (125, 435)
detect_coor_3 = (400, 295)
detect_coor_4 = (400, 435)

path = r'D:\darknet_fender_protection\ship\test_data'
markup = r'D:\darknet_fender_protection\ship\reference_files\markup_9.jpg'
makrup_n = r'D:\darknet_fender_protection\ship\reference_files\markup_9_up.jpg'
out_path = path +'/out/'
src1 = None
ctr = 0                 # to set reference file

mark = cv2.imread(markup)
test = cv2.imread(makrup_n)


def mask_fender(img_m, markup_img):
    new_lst = []
    mark_2d_ = markup_img[:, :, 0]              # 3D to 2D image conversion
    for j in range(mark_2d_.shape[0]):           # H but rows of image
        i_start_ = 0
        i_end_ = 0
        if np.all((mark_2d_[j, :] == 0)):       # check if all row values are zero
            pass
        else:
            f_s = False
            f_e = False
            for i in range(mark_2d_.shape[1]):       # W but cols of image
                if mark_2d_[j, i] != 0 and mark_2d_[j, i+5] == 0 and f_s==False:
                    i_start_ = i
                    f_s = True

            for k in range(mark_2d_.shape[1]-1, i_start_, -1):
                if mark_2d_[j, k] > 0 and mark_2d_[j, k - 5] == 0 and f_e == False:
                    i_end_ = k
                    f_e = True
        if i_end_ != 0 and i_start_ != 0:
            new_lst.append((j, i_start_, i_end_))
    for i in new_lst:
        j = i[0]
        for k in range(i[1], i[2]):
            img_m[j, k, :] = [0, 0, 0]                 # mask as black
    return img_m


for file in os.listdir(out_path):
    if ctr == 0:
        src1 = cv2.imread(out_path+file)
        ctr = ctr+1
    else:
        src2 = cv2.imread(out_path+file)
        dst = src2 - src1
        dtype = -1
        sub = cv2.subtract(src2, src1, dst)
        sub = cv2.rectangle(sub, detect_coor_1, detect_coor_4, (0, 0, 0), -1)   # masking the detected coordinate
        gray = cv2.cvtColor(sub,cv2.COLOR_BGR2GRAY)
        edge = cv2.Canny(gray, 10,150)
        #cont, _ = cv2.findContours(cv2.cvtColor(sub,cv2.COLOR_BGR2GRAY), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #cv2.drawContours(sub, cont, 0, (0,255,0), -1)
        #sub = mask_fender(sub, mark)
        cv2.imwrite(out_path +str(ctr)+'new.jpg', edge)
        cv2.imwrite(out_path + str(ctr)+'_new.jpg', sub)
        ctr=ctr+1


path = "D:/darknet_fender_protection/ship/test_data/2_dst_2_1_sub_2_1/"

##
#makr_n = cv2.cvtColor(cv2.imread(makrup_n), cv2.COLOR_BGR2GRAY)
#cont, _ = cv2.findContours(makr_n, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#cv2.drawContours(test, cont, 0, (0,255,0), -1)
#cv2.fillPoly(test,cont,(255,255,255))
#cv2.imwrite(path +'/contours.jpg', test)



# *********************************** STREAMMING TEST USING VARIOUS LIBRARIES *******************

from read_config import *
from image_codes import *

stream0 = 'D:/darknet_fender_protection/ship/test_data/video2.mp4'  # path1
stream1 = 'rtsp://192.168.29.73:8554/vlc'
stream2 = 'rtsp://admin:DS-2CD206@192.168.29.32'
stream3 = 'rtsp://localhost:8554/stream'
stream4 = 'rtsp://192.168.51.37:8554/streams'
stream5 = 'rtsp://192.168.6.62:8554/stream'

# Method 1 ---- using OPENCV
cap = cv2.VideoCapture(stream3, cv2.CAP_FFMPEG)
ctr = 0
print(cap)
while True and ctr < 20 :
    print(cap.isOpened())
    time.sleep(4)
    ret, frame = cap.read()
    print(ret, type(frame))
    ctr = ctr+1


# Method 2  --------> using vlc
vlc_player_object = vlc.MediaPlayer(stream3)
time.sleep(1)
vlc_player_object.play()
ctr = 0
while (True and ctr<10) or (vlc_player_object.is_playing() and ctr<100):
#while True:
    print(vlc_player_object.is_playing())
    print(ctr)
    time.sleep(2)
    vlc_player_object.video_take_snapshot(0, 'D:/darknet_fender_protection/vlc_stream.jpg', 0, 0)
    ctr = ctr +1
vlc_player_object.pause()
#vlc_player_object.release()
del vlc_player_object


# Method 3 ----------> using RTSP
with rtsp.Client(rtsp_server_uri = stream3) as client:
    ctr = 0
    time.sleep(2)
    while True and ctr < 20:
        time.sleep(4)
        _image = client.read(raw=True)
        ctr+=1
        print(type(_image))
    #return _image

import requests
image_path = 'D:/darknet_fender_protection/vlc_stream.jpg'
notification_url='http://localhost:3011/api/v1/notification/create'
headers = {'x-access-token': 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJfaWQiOiI2MDQyMzA3ZDZhM2IyMjI2YjgyZmE0MmMiLCJyb2xlIjoiQURNSU4iLCJpYXQiOjE2MTUyODMyMTR9.FPv1C_AELB9UStxx-Jwj5Vax7OgRJr6Rr3sFPRxMa4M'}
files = {'imageUrl': open(image_path, 'rb')}
payload = {'title': "Dummy", 'status': 'threat',
           'type': 'known',
           'siteNumber': '4'}
session = requests.Session()
temp_ = session.post(notification_url, headers=headers, data=payload, files=files)
print(temp_.status_code, temp_.json())


# ----------------------- TENSORFLOW operations ------------------------




