import cv2
import pykinect_azure as pykinect
import numpy as np
import time
import threading

import serial
from serial.tools import list_ports

from math import sqrt, log

def joints2dist(joints):
    data = {}
    for j in joints:
      data[j.get_name()] = j.numpy()
    return data
  
def mapCvt(org, level, width):
    min_value = level-width/2
    img = org - min_value
    img = img/width*255
    img = np.where(img<0, 0, img)
    img = np.where(img>255, 255, img)   
    return img.astype(np.uint8)

def nolinear(x, b=0.5):
  a = -1/(b*b)
  y = a*x*x+1
  y = np.where(x<b, y, 0.0)
  return y

def linear(x, a=0.5):
  y = np.clip(1-x/a, 0, 1)
  return y

def step(x, threshold=0.5):
  y = np.where(x<threshold, 1.0, 0.0)
  return y

Weight_Method = {"nolinear":nolinear, "linear":linear, "step":step}

class Kinect:
  def __init__(self, light_model=False):
    pykinect.initialize_libraries(track_body=True)

    device_config = pykinect.default_configuration
    device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_720P
    device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED
    device_config.camera_fps = pykinect.K4A_FRAMES_PER_SECOND_30
    device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32
     
    self.device = pykinect.start_device(config=device_config)
    self.bodyTracker = pykinect.start_body_tracker(
      pykinect.K4ABT_LITE_MODEL if light_model 
        else pykinect.K4ABT_DEFAULT_MODEL)
    self.capture = None
    self.body_num = 0
    self.depth_img = None
    self.color_img = None
    
  def update(self, body=True):
    while True:
      self.capture = self.device.update()
      if body:
        self.body_frame = self.bodyTracker.update()
      
      
      self.update_body_closest()
      if self.body_num != 0:
        break
      time.sleep(0.1)

    _, segment_image = self.body_frame.get_body_index_map_image()
    _, depth_image = self.capture.get_depth_image()
    _, color_image = self.capture.get_color_image()
    color_skeleton = self.body_frame.draw_bodies(color_image, pykinect.K4A_CALIBRATION_TYPE_COLOR)
    
    self.depth_img = depth_image
    self.color_img = color_image
    
    masked_depth = np.where(segment_image==0, depth_image, 0)
    self.mask_img = np.where(segment_image==0, 1, 0)
    self.masked_depth = masked_depth
      
    neck, pelvis = self.target_joints["neck"], self.target_joints["pelvis"]
    
    self.joints = [self.target_joints[i] for i in ("neck", "pelvis", "left shoulder", "right shoulder")]
    
    #return masked_depth, (neck, pelvis), color_skeleton   
  
  def update_body_closest(self):
    self.body_num = self.body_frame.get_num_bodies()
    self.target_idx = None
    norm_max = 0
    for idx in range(self.body_num):
      body = self.body_frame.get_body2d(idx)
      body3d = self.body_frame.get_body(idx)
      joints3d = joints2dist(body3d.joints)
      joints = joints2dist(body.joints)
      neck = joints["neck"]
      pelvis = joints["pelvis"]
      vec_norm = np.linalg.norm(neck - pelvis)
      if vec_norm > norm_max:
        self.target_idx = idx
        self.target_joints = joints.copy()
        
        self.joints3d = joints3d.copy()
      
  def get_depth_image(self):
    return 
    
class SerialArduino:
  def __init__(self, port=None, baudorate=9600, timeout=2):
    ports = list(list_ports.comports())
    port_list = list(map(lambda d:d.device, ports))
    
    print(port_list)
    print(port in port_list)
    self.available = True
    if ports and port is None:
      port = port_list[0]
    elif port in port_list:
      port = port
    else:
      self.available = False
    
    if self.available:
      self.ser = serial.Serial(port, baudrate=baudorate, timeout=timeout)
      self.alive = True 
      self.thread = threading.Thread(target=self.readloop)
      self.thread.start()
      self.value = -1

  def readloop(self):
    while self.alive:
      data = self.ser.readline()
      if data != b'\r\n' and data != b'\n':
        self.value = float(data[:4])
    self.ser.close() 
        
  def readAsync(self):
    return self.value
    
  def read(self):
    while True:
      data = self.ser.readline()
      if data != b'\r\n' and data != b'\n':
        return float(data[:4])

  def terminate(self):
    if not(self.available):
      return
    self.alive = False
    