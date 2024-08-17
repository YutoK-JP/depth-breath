import cv2
import time
import numpy as np
import util.utils_old as utils_old
import sys

from pyqtgraph.Qt import QtCore
import pyqtgraph as pg
from PyQt5 import uic, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel

#ハイパーパラメータの指定
N=5                 #脊椎の分割数
GRAPH_WIDTH=150     #グラフの横軸の数
FPS = 30            #フレームレート
DEFAULT_EDGE = 0.1      #ガウシアン関数の分散σ
DEPTH_WIDTH = 180     #depthImageBox.frameGeometry().width()
COLOR_WIDTH = 440
COLOR_HEIGHT = 320
dt = 1/FPS
WEIGHT_METHOD = utils_old.Weight_Method
METHOD_NAMES = WEIGHT_METHOD.keys()
#port = "COM7"
port=None

waves_arduino=[]
waves_dict = {}
frame_times = []
for method_name in METHOD_NAMES:
  waves_dict[method_name] = [[]*i for i in range(N)]
  
app = pg.mkQApp("Recorder")
win = uic.loadUi("recorder_multi.ui")
pg.setConfigOptions(antialias=True)

depthImageBox = win.depth_map
colorImageBox = win.color_image

depthImageBox.setScene(QtWidgets.QGraphicsScene(0, 0, DEPTH_WIDTH, DEPTH_WIDTH, depthImageBox) )
depthImageBox.setRenderHint(QtGui.QPainter.Antialiasing, False)
colorImageBox.setScene(QtWidgets.QGraphicsScene(0, 20, COLOR_WIDTH, COLOR_HEIGHT, colorImageBox) )
colorImageBox.setRenderHint(QtGui.QPainter.Antialiasing, False)

plot_color=np.array([(100,100,255), (100,255,100), (255,100,100), (100,255,255), (255,100,255), (255,255,100)], dtype=np.float32)
figure_global = win.graph1_1.addPlot(title="global depth / aruduino measure")
figure_region = win.graph1_2.addPlot(title="region depth")
figure_fft = win.graph1_3.addPlot(title="region fft")
plot_global = figure_global.plot(pen='y')
plot_global_median = figure_global.plot(pen='r')
plot_region = [figure_region.plot(pen=plot_color[i]) for i in range(N)]
plot_fft = [figure_fft.plot(pen=plot_color[i]) for i in range(N)]


figure_global.setXRange(-6, 0, padding=0)
figure_region.setXRange(-6, 0, padding=0)

cutoff_slider = win.cutoff_slider
variance_slider = win.variance_slider
variance_slider.setValue(int(DEFAULT_EDGE*200))
variance_slider.sliderPosition = int(DEFAULT_EDGE*200)

check_boxes = [
  win.checkBox_1,
  win.checkBox_2,
  win.checkBox_3,
  win.checkBox_4,
  win.checkBox_5,
]

radio_buttons ={
  "nolinear":win.radioButton_Nolinear,
  "linear":win.radioButton_Linear,
  "step":win.radioButton_Step,
}

T=[]
Y_global_mean = []
Y_global_median = []
Y_pelvis_fft  = []
ptr=1

kinect = utils_old.Kinect()
if port is not None:
  arduino = utils_old.SerialArduino(port)
else:
  arduino = None

def lowpass(x, fc=10):
  freq = np.linspace(0, FPS, len(x))
  fs = FPS
  fc_upper = fs - fc
  F = np.fft.fft(x)
  G = F.copy()
  G[((freq > fc)&(freq< fc_upper))] = 0+0j
  g = np.fft.ifft(G)
  return g.real

   
def update():
  global ptr, kinect, start_time, T, Y_global_mean, Y_global_median, Y_pelvis_fft, plot_global, plot_region, plot_fft
  
  frame_start = time.time()
  #深度マップと胴体位置の更新
  depth_img, (neck, pelvis), color_image = kinect.update()
  if arduino is not None:
    waves_arduino.append(arduino.read())
  
  #region 深度マップと胴体位置のクロッピング(頭部や複数検出時の誤作動除去)
  h,w = depth_img.shape
  vertical_spine = int( max( abs(neck[0] - pelvis[0]), abs(neck[1] - pelvis[1]) ) /2 )
  center_spine = (neck+pelvis)//2
  left_top = (center_spine - vertical_spine).astype(np.uint16)
  right_bottom = (center_spine + vertical_spine).astype(np.uint16)
  cropped_depth = depth_img[
    max(0,left_top[1]) : min(w,right_bottom[1]),
    max(0,left_top[0]) : min(h,right_bottom[0])
  ]
  cropped_neck, cropped_pelvis = neck-left_top, pelvis-left_top
  #endregion
  
  #胴体の傾きの取得
  spine_vec = cropped_neck - cropped_pelvis
  slope = spine_vec[1]/spine_vec[0]

  #計算用の座標グリッド
  x_line = np.arange(cropped_depth.shape[0])
  y_line = np.arange(cropped_depth.shape[1])
  x_grid,y_grid = np.meshgrid(x_line, y_line)
  
  #領域分割用の１次関数
  gradation_split = x_grid + slope*y_grid
  
  #重み付け用の１次関数（軸からの距離）  
  #0 <= variance < 1
  variance = variance_slider.value()/200
  gradation_distance = np.abs((y_grid-cropped_neck[1])-(x_grid-cropped_neck[0])*slope)
  gradation_distance /= gradation_distance.max()
  
  #重みづけ手法の繰り返し
  weighted_maps = {}
  
  split_pos = np.stack((np.linspace(cropped_neck[0], cropped_pelvis[0], N+1), np.linspace(cropped_neck[1], cropped_pelvis[1], N+1)), 1)
  T.append(time.time()-start_time)
  for method_name in METHOD_NAMES:
    weight_map = WEIGHT_METHOD[method_name](gradation_distance, variance)
    weighted_depth = weight_map * cropped_depth
    
    depth_devided = []
    
    view_image = weighted_depth.copy()
    view_image = ((view_image-view_image.min())/(view_image.max()-view_image.min())).astype(np.float32)
    view_image = np.stack([view_image, view_image, view_image], axis=-1)
    weighted_maps[method_name] = view_image
    for i in range(N):
      point1, point2 = split_pos[i], split_pos[i+1]
      threshold1 = point1[0] + point1[1]*slope
      threshold2 = point2[0] + point2[1]*slope
      mask = np.where((gradation_split>threshold1)^(gradation_split>threshold2), True, False)
      region = (weight_map>0.0)&mask
      target_region = weighted_depth[region]
      view_image[region] *= plot_color[i]
      depth_devided.append(target_region)
      mean_region = target_region.mean()
      waves_dict[method_name][i].append(mean_region)
    
    #region 数値処理
    #mean_devided = np.array([region_depth.mean() for region_depth in depth_devided])
    #endregion
  global_mean = cropped_depth[cropped_depth>0].mean()
  global_median = np.median(cropped_depth[cropped_depth>0])
  Y_global_mean.append(global_mean) 
  Y_global_median.append(global_median)
  
  for k,button in radio_buttons.items():
    if button.isChecked():
      display_method = k
      break
  
  #時間列の加工
  T_np = np.array(T)
  T_graph = T_np-T_np.max()
  breath_rate = 0.0
  #region 表示処理
  if(len(T)>GRAPH_WIDTH):
    #全体の信号を処理
    T_graph=T_graph[-GRAPH_WIDTH:]
    #array_global =  np.array(waves_dict[display_method][-1][-GRAPH_WIDTH:])
    if arduino is not None:
      array_global = np.array(waves_arduino[-GRAPH_WIDTH:])
      amp_global_DC = array_global.mean()
      array_global_0base = array_global-amp_global_DC
      plot_global.setData(T_graph, array_global_0base)
    else:
      array_global =  np.array(Y_global_mean[-GRAPH_WIDTH:])
      amp_global_DC = array_global.mean()
      array_global_0base = array_global-amp_global_DC
      plot_global.setData(T_graph, array_global_0base)
      plot_global_median.setData(T_graph, Y_global_median[-GRAPH_WIDTH:])
    
    freq_cutoff = cutoff_slider.value()/10
    
    #領域ごとの信号を処理
    for i in range(N):
      if not(check_boxes[i].isChecked()):
        plot_region[i].clear()
        plot_fft[i].clear()
      else:
        array_region =  np.array(waves_dict[display_method][i][-GRAPH_WIDTH:])
        array_region_lp = lowpass(array_region, freq_cutoff)
        amp_region_DC = array_region_lp.mean()
        array_region_0base = array_region_lp-amp_region_DC
        plot_region[i].setData(T_graph, array_region_0base)
      
        #FFTの処理
        N_fft = len(T_graph)
        dt = (time.time()-start_time)/ptr
        F = np.fft.fft(array_region_0base)
        amp = np.abs(F/(N_fft/2))
        amp = amp/amp.max()
        freq = np.fft.fftfreq(N_fft, d=dt)
        amp = amp[:int(N_fft/2)]
        freq = freq[:int(N_fft/2)]
        plot_fft[i].setData(freq, amp)
        if i==N-2:
          breath_rate = freq[np.argmax(amp)]
  else:
    #plot_global.setData(T_graph, waves_dict[display_method][-1])
    if arduino is not None:
      plot_global.setData(T_graph, waves_arduino)
    else:
      plot_global.setData(T_graph, Y_global_mean)
      plot_global_median.setData(T_graph, Y_global_median)
    for i,p in enumerate(plot_region):
      p.setData(T_graph, waves_dict[display_method][i])  
    
  #endregion
  
  #region 深度マップ、その他情報の表示
  cv2.line(cropped_depth, cropped_neck.astype(np.uint32), cropped_pelvis.astype(np.uint32), 0, 8)
  """
  window_level, window_width = win.level_slider.value(), win.width_slider.value()
  depth_img_view = mapCvt(cropped_weight, window_level, window_width)
  image = QtGui.QImage(depth_img_view, DEPTH_WIDTH, DEPTH_WIDTH, DEPTH_WIDTH, QtGui.QImage.Format_Grayscale8)
  """
  depth_img_view = cv2.resize(weighted_maps[display_method], (DEPTH_WIDTH,DEPTH_WIDTH)).astype(np.uint8)
  depth_image = QtGui.QImage(depth_img_view, DEPTH_WIDTH, DEPTH_WIDTH, DEPTH_WIDTH*3, QtGui.QImage.Format_RGB888)
  depth_pixmap = QtGui.QPixmap.fromImage(depth_image)
  
  color_img_view = cv2.cvtColor(cv2.resize(color_image, (COLOR_WIDTH,COLOR_HEIGHT)).astype(np.uint8), cv2.COLOR_RGB2BGR)
  color_image = QtGui.QImage(color_img_view, COLOR_WIDTH, COLOR_HEIGHT, COLOR_WIDTH*3, QtGui.QImage.Format_RGB888)
  color_pixmap = QtGui.QPixmap.fromImage(color_image)
  
  depthImageBox.scene().clear()
  depthImageBox.scene().addPixmap(depth_pixmap)
  colorImageBox.scene().clear()
  colorImageBox.scene().addPixmap(color_pixmap)
  
  frame_times.append(time.time()-frame_start)
  win.Info.setText(
f"""aaa""")
  #endregion
  
  ptr += 1
  if(ptr==800):
    np.savez("proccessed_waves.npz", **waves_dict)
    np.save("global_waves.npy", np.array(waves_arduino if arduino is not None else Y_global_mean))
    np.save("global_median.npy", np.array(Y_global_median))
    sys.exit()
  

if __name__ =="__main__":
  qtTimer = QtCore.QTimer()
  qtTimer.timeout.connect(update)
  qtTimer.start(1000//FPS)
  win.show()
  start_time=time.time()
  pg.exec()