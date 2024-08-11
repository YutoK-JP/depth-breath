import cv2
import time
import numpy as np
from utils import Kinect, SerialArduino
import sys

from pyqtgraph.Qt import QtCore
import pyqtgraph as pg
from PyQt5 import uic, QtGui, QtWidgets

import os

TRACKING_CYCLE = 2
LIGHT_MODEL = True

N = 5  # 脊椎の分割数
GRAPH_WIDTH = 150  # グラフの横軸の数
FPS = 30  # フレームレート
DEPTH_WIDTH = 180  # procImageBox.frameGeometry().width()
COLOR_WIDTH = 280
COLOR_HEIGHT = 240
dt = 1 / FPS

frame_times = []

app = pg.mkQApp("Measure")
win = uic.loadUi("./recorder_v1.0.ui")
pg.setConfigOptions(antialias=True)

colorImageBox = win.ColorImage
procImageBox = win.ProcImage

procImageBox.setScene(
    QtWidgets.QGraphicsScene(0, 0, DEPTH_WIDTH, DEPTH_WIDTH, procImageBox)
)
procImageBox.setRenderHint(QtGui.QPainter.Antialiasing, False)

colorImageBox.setScene(
    QtWidgets.QGraphicsScene(0, 20, COLOR_WIDTH, COLOR_HEIGHT, colorImageBox)
)
colorImageBox.setRenderHint(QtGui.QPainter.Antialiasing, False)

figure1 = win.graph1.addPlot(title="global average")
figure2 = win.graph2.addPlot(title="chest - stomach")
figure3 = win.graph3.addPlot(title="ground truth (pressure sensor)")
plot1 = figure1.plot(pen="y")
plot3 = figure3.plot(pen="red")

figure2.showAxis("right")
figure2.getAxis("left").setLabel("stomach", color="#f00")
figure2.getAxis("right").setLabel("chest", color="#ff0")
fig2_ave = pg.ViewBox()
figure2.scene().addItem(fig2_ave)
figure2.getAxis("right").linkToView(fig2_ave)
fig2_ave.setXLink(figure2)


def updateViews():
    fig2_ave.setGeometry(figure2.vb.sceneBoundingRect())
    fig2_ave.linkedViewChanged(figure2.vb, fig2_ave.XAxis)


updateViews()
figure2.vb.sigResized.connect(updateViews)

array_time = []
array_global = []
array_torso = []
array_chest = []
array_stomach = []
array_pressure = []

ptr = 0

kinect = Kinect(light_model=LIGHT_MODEL)
arduino = SerialArduino(port="COM7")

print(bool(arduino))


def update():
  global ptr, kinect, start_time, array_time, array_chest
  global array_stomach, array_global, array_pressure, array_torso, plot1, plot3

  frame_start = time.time()
  kinect.update(ptr % TRACKING_CYCLE==0)
  (neck, pelvis, left_shoulder, right_shoulder) = kinect.joints
  depth_img = kinect.masked_depth
  color_image = kinect.color_img

  # 3dでの関節処理
  depth_neck = kinect.joints3d["neck"][2]
  depth_pelvis = kinect.joints3d["pelvis"][2]
  z_torso = (depth_neck + depth_pelvis) / 2

  # region 深度マップと胴体位置のクロッピング(頭部や複数検出時の誤作動除去)
  h, w = depth_img.shape
  vertical_spine = int(
    max(abs(neck[0] - pelvis[0]), abs(neck[1] - pelvis[1])) / 2)
  center_spine = (neck + pelvis) // 2
  left_top = (center_spine - vertical_spine).astype(np.uint16)
  right_bottom = (center_spine + vertical_spine).astype(np.uint16)
  cropped_depth = depth_img[
    max(0, left_top[1]): min(w, right_bottom[1]),
    max(0, left_top[0]): min(h, right_bottom[0]),
  ]
  cropped_neck, cropped_pelvis = neck - left_top, pelvis - left_top
  cropped_left_shoulder, cropped_right_shoulder = (
    left_shoulder - left_top,
    right_shoulder - left_top,
  )
  # endregion

  # 胴体の傾きの取得
  spine_vec = cropped_neck - cropped_pelvis
  slope = spine_vec[1] / spine_vec[0]

  # 計算用の座標グリッド
  x_line = np.arange(cropped_depth.shape[0])
  y_line = np.arange(cropped_depth.shape[1])
  x_grid, y_grid = np.meshgrid(x_line, y_line)

  # 領域分割用の１次関数
  gradation_split = x_grid + slope * y_grid
  split_pos = np.stack(
    (
      np.linspace(cropped_neck[0], cropped_pelvis[0], N + 1),
      np.linspace(cropped_neck[1], cropped_pelvis[1], N + 1),
    ),
    1,
  )

  # 重み付け用の１次関数（軸からの距離）
  gradation_distance = (
    y_grid - cropped_neck[1]) - (x_grid - cropped_neck[0]) * slope

  # 1時間数に沿った肩幅位置
  distance_right_shoulder = (cropped_right_shoulder[1] - cropped_neck[1]) - slope * (
    cropped_right_shoulder[0] - cropped_neck[0]
  )
  distance_left_shoulder = (cropped_left_shoulder[1] - cropped_neck[1]) - slope * (
    cropped_left_shoulder[0] - cropped_neck[0]
  )

  depth_torso = np.where(
    (
      (gradation_distance > distance_right_shoulder)
      & (gradation_distance < distance_left_shoulder)
      | (gradation_distance < distance_right_shoulder)
      & (gradation_distance > distance_left_shoulder)
    ),
    cropped_depth,
    0.0,
  )

  # 縦領域の選定
  border_upeer_stomach, border_lower_stomach = split_pos[3], split_pos[4]
  threshold_upeer_stomach = border_upeer_stomach[0] + \
    border_upeer_stomach[1] * slope
  threshold_lower_stomach = border_lower_stomach[0] + \
    border_lower_stomach[1] * slope
  mask_stomach = np.where(
    (gradation_split > threshold_upeer_stomach)
    ^ (gradation_split > threshold_lower_stomach),
    True,
    False,
)
  region_stomach = (depth_torso > 0.0) & mask_stomach
  stomach_depths = depth_torso[region_stomach]

  border_upeer_chest, border_lower_chest = split_pos[1], split_pos[2]
  threshold_upeer_chest = border_upeer_chest[0] + \
    border_upeer_chest[1] * slope
  threshold_lower_chest = border_lower_chest[0] + \
    border_lower_chest[1] * slope
  mask_chest = np.where(
    (gradation_split > threshold_upeer_chest)
    ^ (gradation_split > threshold_lower_chest),
    True,
    False,
  )
  region_chest = (depth_torso > 0.0) & mask_chest
  chest_depths = depth_torso[region_chest]

  array_time.append(time.time() - start_time)
  array_global.append(depth_torso.mean())
  array_stomach.append(stomach_depths.mean())
  array_chest.append(chest_depths.mean())
  array_torso.append(z_torso)

  if arduino.available:
    array_pressure.append(arduino.readAsync() / 1024)

  plot1.setData(array_time[-GRAPH_WIDTH:], array_global[-GRAPH_WIDTH:])
  plot3.setData(array_time[-GRAPH_WIDTH:], array_pressure[-GRAPH_WIDTH:])

  figure2.clear()
  fig2_ave.clear()
  figure2.plot(array_time[-GRAPH_WIDTH:],
                array_stomach[-GRAPH_WIDTH:], pen="r")
  #figure2.plot(array_time[-GRAPH_WIDTH:],
  #              array_torso[-GRAPH_WIDTH:], pen=pg.mkPen("#00ffff"))
  fig2_ave.addItem(
    pg.PlotCurveItem(array_time[-GRAPH_WIDTH:],
                      array_torso[-GRAPH_WIDTH:], pen="y")
  )

  # 表示用画像の処理
  view_image = depth_torso.copy().astype(np.float32)
  valid_min = np.where(view_image > 0, view_image, view_image.max()).min()
  view_image = np.where(
    view_image < valid_min,
    0,
    ((view_image - valid_min) / (view_image.max() - valid_min)),
  )
  view_image *= 255
  view_image = (np.stack([view_image, view_image, view_image], axis=-1)).astype(
    np.uint8
  )

  cv2.circle(view_image, cropped_right_shoulder.astype(
    np.uint16), 10, (255, 0, 0), 2)
  cv2.circle(view_image, cropped_left_shoulder.astype(
    np.uint16), 10, (255, 0, 0), 2)

  # region 深度マップ、その他情報の表示
  color_img_view = cv2.cvtColor(
    cv2.resize(color_image, (COLOR_WIDTH, COLOR_HEIGHT)), cv2.COLOR_RGB2BGR
  )
  color_image = QtGui.QImage(
    color_img_view,
    COLOR_WIDTH,
    COLOR_HEIGHT,
    COLOR_WIDTH * 3,
    QtGui.QImage.Format_RGB888,
  )
  color_pixmap = QtGui.QPixmap.fromImage(color_image)
  colorImageBox.scene().clear()
  colorImageBox.scene().addPixmap(color_pixmap)

  depth_img_view = cv2.resize(view_image, (DEPTH_WIDTH, DEPTH_WIDTH))
  depth_image = QtGui.QImage(
    depth_img_view,
    DEPTH_WIDTH,
    DEPTH_WIDTH,
    DEPTH_WIDTH * 3,
    QtGui.QImage.Format_RGB888,
  )
  depth_pixmap = QtGui.QPixmap.fromImage(depth_image)
  procImageBox.scene().clear()
  procImageBox.scene().addPixmap(depth_pixmap)
  win.Info.setText(f"""{array_global.__sizeof__()}""")
  # endregion
  ptr += 1


if __name__ == "__main__":
  qtTimer = QtCore.QTimer()
  qtTimer.timeout.connect(update)
  qtTimer.start(1000 // FPS)
  win.show()
  start_time = time.time()
  pg.exec()
  
  arduino.terminate()
  idx=1
  
  while True:
    filename = f"output\\data_{idx:02}.npz"
    if os.path.isfile(filename):
      idx += 1
      continue
    print(filename)
    if arduino.available:
      np.savez_compressed(
          filename,
          t=array_time,
          y_global=array_global,
          y_chest=array_chest,
          y_stomach=array_stomach,
          y_truth=array_pressure,
          y_torso=array_torso
      )
    else:
      np.savez_compressed(
          filename,
          t=array_time,
          y_global=array_global,
          y_chest=array_chest,
          y_stomach=array_stomach,
          y_torso=array_torso
      )
    break