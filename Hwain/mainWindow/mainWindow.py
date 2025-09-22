from PyQt5.QtWidgets import QShortcut, QApplication, QMainWindow, QGraphicsDropShadowEffect , QPushButton,QLineEdit, QLabel, QMessageBox, QInputDialog, QCheckBox, QStackedWidget, QAction, QFrame, QWidget, QVBoxLayout, QFileDialog
from pathlib import Path
from PyQt5.QtCore import QThread, QUrl, QTimer, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtGui import QIcon, QPixmap, QVector3D, QKeySequence
import pyqtgraph as pg
from pyqtgraph import PlotWidget, GridItem, AxisItem
#################
import numpy as np
from numpy import array, deg2rad, cos, sin, cross
from numpy.linalg import norm
from PyQt5.QtCore import QThread, QTimer, pyqtSignal
from PyQt5.QtWidgets import QLabel
import pyqtgraph.opengl as gl
import sys
import os
import pandas as pd
import time
#################
from numpy import zeros, array, cross, reshape, sin, cos, deg2rad, rad2deg
from numpy.random import rand
from numpy.linalg import norm

from matplotlib.pyplot import figure
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib import pyplot as plt

import traceback

from datetime import timedelta
from os.path import abspath, dirname, join, exists
from sys import exit, argv

import datahub

from pandas import read_csv

from . import widgetStyle as ws

COLUMN_NAMES = [
    "hours","mins","secs","tenmilis","E","N","U","v_E","v_N","v_U","a_p","a_y","a_r","q_0","q_1","q_2","q_3","w_p","w_y","w_r"
]

class PageWindow(QMainWindow):
    gotoSignal = pyqtSignal(str)
    def goto(self, name):
        self.gotoSignal.emit(name)

# class GraphViewer_Thread(QThread):
#     def __init__(self, mainwindow, datahub, update_ms=50, tail_len=500):
#         super().__init__()
#         self.mainwindow = mainwindow
#         self.datahub = datahub

#         # ---- 위젯/레이아웃(네 기존 ws.* 사용 유지) ----
#         self.view = QWebEngineView(self.mainwindow.container)
#         self.view.load(QUrl())
#         self.view.setGeometry(*ws.webEngine_geometry)
        
#         self.angleSpeed_title = QLabel(self.mainwindow.container)
#         self.angleSpeed_title.setText("<b>&#8226; Angle Speed</b>")
#         self.angleSpeed_title.setStyleSheet("color: white;")
#         self.pw_angleSpeed = PlotWidget(self.mainwindow.container)
        
#         self.accel_title = QLabel(self.mainwindow.container)
#         self.accel_title.setText("<b>&#8226; Acceleration</b>")
#         self.accel_title.setStyleSheet("color: white;")
#         self.pw_accel = PlotWidget(self.mainwindow.container)
        
#         self.speed_title = QLabel(self.mainwindow.container)
#         self.speed_title.setText("<b>&#8226; Speed (ENU)</b>")
#         self.speed_title.setStyleSheet("color: white;")
#         self.pw_speed = PlotWidget(self.mainwindow.container)
        
#         self.pw_angleSpeed.setGeometry(*ws.pw_angleSpeed_geometry)
#         self.pw_accel.setGeometry(*ws.pw_accel_geometry)
#         self.pw_speed.setGeometry(*ws.pw_speed_geometry)
          
#         self.angleSpeed_title.setGeometry(*ws.angleSpeed_title_geometry)
#         self.accel_title.setGeometry(*ws.accel_title_geometry)
#         self.speed_title.setGeometry(*ws.speed_title_geometry)

#         self.angleSpeed_title.setFont(ws.font_angleSpeed_title)
#         self.accel_title.setFont(ws.font_accel_title)
#         self.speed_title.setFont(ws.font_speed_title)

#         # grids
#         self.pw_angleSpeed.addItem(GridItem())
#         self.pw_accel.addItem(GridItem())
#         self.pw_speed.addItem(GridItem())

#         # axis labels
#         self.pw_angleSpeed.getPlotItem().getAxis('bottom').setLabel('Time (s)')
#         self.pw_angleSpeed.getPlotItem().getAxis('left').setLabel('deg/s')
#         self.pw_accel.getPlotItem().getAxis('bottom').setLabel('Time (s)')
#         self.pw_accel.getPlotItem().getAxis('left').setLabel('g')
#         self.pw_speed.getPlotItem().getAxis('bottom').setLabel('Time (s)')
#         self.pw_speed.getPlotItem().getAxis('left').setLabel('m/s')

#         # ranges
#         self.pw_angleSpeed.setYRange(-1000, 1000)
#         self.pw_accel.setYRange(-20, 20)
#         self.pw_speed.setYRange(-100, 1000)

#         # legends
#         self.pw_angleSpeed.getPlotItem().addLegend()
#         self.pw_accel.getPlotItem().addLegend()
#         self.pw_speed.getPlotItem().addLegend()

#         # curves
#         self.curve_rollSpeed  = self.pw_angleSpeed.plot(pen='r', name="roll speed")
#         self.curve_pitchSpeed = self.pw_angleSpeed.plot(pen='g', name="pitch speed")
#         self.curve_yawSpeed   = self.pw_angleSpeed.plot(pen='b', name="yaw speed")

#         self.curve_xaccel = self.pw_accel.plot(pen='r', name="x acc")
#         self.curve_yaccel = self.pw_accel.plot(pen='g', name="y acc")
#         self.curve_zaccel = self.pw_accel.plot(pen='b', name="z acc")

#         self.curve_e_speed = self.pw_speed.plot(pen='r', name='E speed (ENU)')
#         self.curve_n_speed = self.pw_speed.plot(pen='g', name='N speed (ENU)')
#         self.curve_u_speed = self.pw_speed.plot(pen='b', name='U speed (ENU)')

#         # buffers
#         self.tail_len = int(tail_len)
#         self.time       = np.zeros(self.tail_len)
#         self.rollSpeed  = np.zeros(self.tail_len)
#         self.pitchSpeed = np.zeros(self.tail_len)
#         self.yawSpeed   = np.zeros(self.tail_len)
#         self.xaccel     = np.zeros(self.tail_len)
#         self.yaccel     = np.zeros(self.tail_len)
#         self.zaccel     = np.zeros(self.tail_len)
#         self.espeed     = np.zeros(self.tail_len)
#         self.nspeed     = np.zeros(self.tail_len)
#         self.uspeed     = np.zeros(self.tail_len)

#         # timer (UI 스레드에서 갱신)
#         self.timer = QTimer(self.mainwindow)
#         self.timer.timeout.connect(self.update_data)
#         self.timer.start(int(update_ms))

#     @pyqtSlot()
#     def update_data(self):
#         # Datahub에서 스냅샷 추출(락으로 보호)
#         with self.datahub.lock:
#             n = len(self.datahub.vE_enu)
#             if n == 0:
#                 return

#             # 필요한 배열들만 복사
#             t           = self.datahub.t.copy()
#             rollSpeeds  = self.datahub.rollSpeeds.copy()
#             pitchSpeeds = self.datahub.pitchSpeeds.copy()
#             yawSpeeds   = self.datahub.yawSpeeds.copy()

#             Xaccels = self.datahub.Xaccels.copy()
#             Yaccels = self.datahub.Yaccels.copy()
#             Zaccels = self.datahub.Zaccels.copy()

#             vE = self.datahub.vE_enu.copy()
#             vN = self.datahub.vN_enu.copy()
#             vU = self.datahub.vU_enu.copy()

#             hrs  = self.datahub.hours.copy()
#             mins = self.datahub.mins.copy()
#             secs = self.datahub.secs.copy()
#             tenm = self.datahub.tenmilis.copy()

#         L = self.tail_len
#         # 뷰 버퍼 초기화
#         self.time.fill(0.0); self.rollSpeed.fill(0.0); self.pitchSpeed.fill(0.0); self.yawSpeed.fill(0.0)
#         self.xaccel.fill(0.0); self.yaccel.fill(0.0); self.zaccel.fill(0.0)
#         self.espeed.fill(0.0); self.nspeed.fill(0.0); self.uspeed.fill(0.0)

#         if n <= L:
#             k = n
#             self.rollSpeed[-k:]  = rollSpeeds[-k:]
#             self.pitchSpeed[-k:] = pitchSpeeds[-k:]
#             self.yawSpeed[-k:]   = yawSpeeds[-k:]

#             self.xaccel[-k:] = Xaccels[-k:]
#             self.yaccel[-k:] = Yaccels[-k:]
#             self.zaccel[-k:] = Zaccels[-k:]

#             self.espeed[-k:] = vE[-k:]
#             self.nspeed[-k:] = vN[-k:]
#             self.uspeed[-k:] = vU[-k:]

#             # 시간: 누적초 t가 있으면 tail 기준 0으로 정규화
#             if len(t) >= k:
#                 tt = t[-k:]
#                 self.time[-k:] = tt - tt[0]
#             else:
#                 t_abs = hrs[-k:]*3600.0 + mins[-k:]*60.0 + secs[-k:] + tenm[-k:]/100.0
#                 self.time[-k:] = t_abs - t_abs[0]
#         else:
#             s = -L
#             self.rollSpeed[:]  = rollSpeeds[s:]
#             self.pitchSpeed[:] = pitchSpeeds[s:]
#             self.yawSpeed[:]   = yawSpeeds[s:]

#             self.xaccel[:] = Xaccels[s:]
#             self.yaccel[:] = Yaccels[s:]
#             self.zaccel[:] = Zaccels[s:]

#             self.espeed[:] = vE[s:]
#             self.nspeed[:] = vN[s:]
#             self.uspeed[:] = vU[s:]

#             if len(t) >= L:
#                 tt = t[s:]
#                 self.time[:] = tt - tt[0]
#             else:
#                 t_abs = hrs[s:]*3600.0 + mins[s:]*60.0 + secs[s:] + tenm[s:]/100.0
#                 self.time[:] = t_abs - t_abs[0]

#         # 그리기
#         self.curve_rollSpeed.setData(x=self.time, y=self.rollSpeed)
#         self.curve_pitchSpeed.setData(x=self.time, y=self.pitchSpeed)
#         self.curve_yawSpeed.setData(x=self.time, y=self.yawSpeed)

#         self.curve_xaccel.setData(x=self.time, y=self.xaccel)
#         self.curve_yaccel.setData(x=self.time, y=self.yaccel)
#         self.curve_zaccel.setData(x=self.time, y=self.zaccel)

#         self.curve_e_speed.setData(x=self.time, y=self.espeed)
#         self.curve_n_speed.setData(x=self.time, y=self.nspeed)
#         self.curve_u_speed.setData(x=self.time, y=self.uspeed)

#     def graph_clear(self):
#         for arr in (self.time, self.rollSpeed, self.pitchSpeed, self.yawSpeed,
#                     self.xaccel, self.yaccel, self.zaccel, self.espeed, self.nspeed, self.uspeed):
#             arr.fill(0.0)

#         self.curve_rollSpeed.clear(); self.curve_pitchSpeed.clear(); self.curve_yawSpeed.clear()
#         self.curve_xaccel.clear();    self.curve_yaccel.clear();    self.curve_zaccel.clear()
#         self.curve_e_speed.clear();   self.curve_n_speed.clear();   self.curve_u_speed.clear()

class MapViewer_Thread(QThread):
    def __init__(self, mainwindow, datahub):
        super().__init__()
        self.mainwindow = mainwindow
        self.datahub = datahub

        # 기준 위치(초기 맵 중심). 필요시 프로젝트 현장 좌표로 바꿔줘.
        self.ref_lat = 37.45162
        self.ref_lon = 126.65058
        self.ref_h   = 0.0

        base_dir = Path(__file__).resolve().parent
        map_path = base_dir / "map.html"
        if not map_path.exists():
            raise FileNotFoundError(f"map.html을 찾을 수 없습니다: {map_path}")

        html = map_path.read_text(encoding="utf-8")
        new_width  = f"{ws.map_geometry[2]}px"
        new_height = f"{ws.map_geometry[3]}px"
        html = html.replace("width: 576px;",  f"width: {new_width};")
        html = html.replace("height: 345px;", f"height: {new_height};")
        map_path.write_text(html, encoding="utf-8")

        self.view = QWebEngineView(self.mainwindow.container)
        self.view.setGeometry(*ws.map_geometry)
        self.view.load(QUrl.fromLocalFile(str(map_path)))
        self.view.show()

    def run(self):
        self.view.loadFinished.connect(self.on_load_finished)

    def on_load_finished(self):
        page = self.view.page()
        self.script = f"""
        var lat = {self.ref_lat};
        var lng = {self.ref_lon};
        var map = L.map("map").setView([lat,lng], 17);
        L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}', {{
            attribution: 'Tiles &copy; Esri',
            maxZoom: 18,
        }}).addTo(map);
        var marker = L.marker([lat,lng]).addTo(map);
        var trigger_javascript = 0;
        function updateMarker(latnew, lngnew, trigger_python) {{
            marker.setLatLng([latnew, lngnew]);
            if(trigger_python >= 1 && trigger_javascript == 0) {{
                map.setView([latnew,lngnew], 15);
                trigger_javascript = 1;
            }}
        }}
        """
        page.runJavaScript(self.script)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_marker)
        self.timer.start(1000)

    # --- ENU(m) -> 위경도(deg) 근사 변환 (작은 영역용) ---
    @staticmethod
    def _enu_to_latlon(E, N, lat0_deg, lon0_deg):
        R = 6378137.0  # 지구 장반경 (m)
        lat0 = np.deg2rad(lat0_deg)
        dlat = (N / R) * (180.0 / np.pi)
        dlon = (E / (R * np.cos(lat0))) * (180.0 / np.pi)
        return lat0_deg + dlat, lon0_deg + dlon

    def update_marker(self):
        page = self.view.page()

        # 1) 위경도 필드가 있으면 그것부터 사용
        if hasattr(self.datahub, "latitudes") and hasattr(self.datahub, "longitudes"):
            if len(self.datahub.latitudes) == 0 or len(self.datahub.longitudes) == 0:
                return
            lat = float(self.datahub.latitudes[-1])
            lng = float(self.datahub.longitudes[-1])

        # 2) 없으면 ENU + 기준 위경도로 변환
        elif len(self.datahub.Easts) > 0 and len(self.datahub.Norths) > 0:
            E = float(self.datahub.Easts[-1])
            N = float(self.datahub.Norths[-1])
            lat, lng = self._enu_to_latlon(E, N, self.ref_lat, self.ref_lon)
        else:
            # 위치 정보가 아직 없음
            return

        trig = getattr(self.datahub, "trigger_python", 0)
        page.runJavaScript(f"updateMarker({lat:.8f}, {lng:.8f}, {int(trig)})")

class RocketViewer_Thread(QThread):
    update_signal = pyqtSignal()

    def __init__(self, mainwindow, datahub):
        super().__init__()
        self.mainwindow = mainwindow
        self.datahub = datahub
        self.pose = array([1.0, 0.0, 0.0])
        self.radius = 0.1
        self.normal = array([0.0, 0.0, 0.0])
        self.x = np.random.rand(1)
        self.y = np.random.rand(1)
        self.circle_point = np.zeros((3, 5))
        self._ref_lla = None
        self._ref_ecef = None
        self.altitude_mode = "enu"  # "enu" = 기준점 대비 상대고도(U), "msl" = 절대고도(LLA h)
        self.setup_ui()

    def setup_ui(self):
        # ▶ 중앙 컨테이너를 부모로 지정
        self.view = gl.GLViewWidget(self.mainwindow.container)
        self.view.setGeometry(*ws.model_geometry)
        self.view.setWindowTitle('Rocket Viewer')
        self.view.setCameraPosition(distance=10)

        # 축 추가
        self.add_axes()

        # 1) 스크립트 기준 폴더 구하기
        base_dir = Path(__file__).resolve().parent

        # 2) rocket.obj 경로 동적 구성 및 존재 여부 체크
        obj_path = base_dir / "rocket.obj"
        if not obj_path.exists():
            raise FileNotFoundError(f"rocket.obj을 찾을 수 없습니다: {obj_path}")

        # 3) OBJ 파일 로드 및 뷰에 추가
        self.rocket_mesh = self.load_and_display_obj(str(obj_path))
        self.view.addItem(self.rocket_mesh)

        # UI 레이블 초기화
        self.speed_label = QLabel("Speed ", self.mainwindow.container)
        self.speed_label.setGeometry(*ws.speed_label_geometry)
        self.speed_label.setStyleSheet("color: #FFFFFF;")
        self.speed_label.setFont(ws.font_speed_text)

        self.altitude_label = QLabel("Altitude ", self.mainwindow.container)
        self.altitude_label.setGeometry(*ws.altitude_label_geometry)
        self.altitude_label.setStyleSheet("color: #FFFFFF;")
        self.altitude_label.setFont(ws.font_altitude_text)

        self.roll_label = QLabel("Roll: ", self.mainwindow.container)
        self.roll_label.setGeometry(*ws.roll_label_geometry)
        self.roll_label.setStyleSheet("color: #FFFFFF;")
        self.roll_label.setFont(ws.font_roll_text)

        self.pitch_label = QLabel("Pitch: ", self.mainwindow.container)
        self.pitch_label.setGeometry(*ws.pitch_label_geometry)
        self.pitch_label.setStyleSheet("color: #FFFFFF;")
        self.pitch_label.setFont(ws.font_pitch_text)

        self.yaw_label = QLabel("Yaw: ", self.mainwindow.container)
        self.yaw_label.setGeometry(*ws.yaw_label_geometry)
        self.yaw_label.setStyleSheet("color: #FFFFFF;")
        self.yaw_label.setFont(ws.font_yaw_text)

        self.rollspeed_label = QLabel("Roll_speed: ", self.mainwindow.container)
        self.rollspeed_label.setGeometry(*ws.rollspeed_label_geometry)
        self.rollspeed_label.setStyleSheet("color: #FFFFFF;")
        self.rollspeed_label.setFont(ws.font_rollspeed_text)

        self.pitchspeed_label = QLabel("Pitch_speed: ", self.mainwindow.container)
        self.pitchspeed_label.setGeometry(*ws.pitchspeed_label_geometry)
        self.pitchspeed_label.setStyleSheet("color: #FFFFFF;")
        self.pitchspeed_label.setFont(ws.font_pitchspeed_text)

        self.yawspeed_label = QLabel("Yaw_speed: ", self.mainwindow.container)
        self.yawspeed_label.setGeometry(*ws.yawspeed_label_geometry)
        self.yawspeed_label.setStyleSheet("color: #FFFFFF;")
        self.yawspeed_label.setFont(ws.font_yawspeed_text)

        self.xacc_label = QLabel("X_acc: ", self.mainwindow.container)
        self.xacc_label.setGeometry(*ws.xacc_label_geometry)
        self.xacc_label.setStyleSheet("color: #FFFFFF;")
        self.xacc_label.setFont(ws.font_xacc_text)

        self.yacc_label = QLabel("Y_acc: ", self.mainwindow.container)
        self.yacc_label.setGeometry(*ws.yacc_label_geometry)
        self.yacc_label.setStyleSheet("color: #FFFFFF;")
        self.yacc_label.setFont(ws.font_yacc_text)

        self.zacc_label = QLabel("Z_acc: ", self.mainwindow.container)
        self.zacc_label.setGeometry(*ws.zacc_label_geometry)
        self.zacc_label.setStyleSheet("color: #FFFFFF;")
        self.zacc_label.setFont(ws.font_zacc_text)

        # 메인 스레드에서 타이머 설정
        self.timer = QTimer(self.mainwindow)
        self.timer.timeout.connect(self.update_pose)
        self.timer.start(33)  # 33ms마다 호출

    def add_axes(self):
        # X축 (빨간색)
        x_axis = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], [10, 0, 0]]), color=(1, 0, 0, 1), width=2, antialias=True)
        self.view.addItem(x_axis)

        # Y축 (초록색)
        y_axis = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], [0, 10, 0]]), color=(0, 1, 0, 1), width=2, antialias=True)
        self.view.addItem(y_axis)

        # Z축 (파란색)
        z_axis = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], [0, 0, 10]]), color=(0, 0, 1, 1), width=2, antialias=True)
        self.view.addItem(z_axis)

    def load_and_display_obj(self, filename):
        vertices, faces = self.load_obj(filename)
        centroid = np.mean(vertices, axis=0)
        mesh = gl.GLMeshItem(vertexes=vertices, faces=faces, drawEdges=False,
                                edgeColor=(1, 1, 1, 1), smooth=False)
        mesh.translate(-centroid[0], -centroid[1], -centroid[2])

        return mesh

    def load_obj(self, filename):
        vertices, faces = [], []
        with open(filename, 'r') as f:
            for line in f:
                if line.startswith('v '):
                    vertices.append(list(map(float, line.strip().split()[1:])))
                elif line.startswith('f '):
                    face = [int(part.split('/')[0]) - 1 for part in line.strip().split()[1:]]
                    faces.append(face)
        return np.array(vertices), np.array(faces)

    def update_pose(self):
        def _last(arr, default=np.nan):
            with self.datahub.lock:  # 락 보호
                try:
                    return float(arr[-1]) if len(arr) > 0 else float(default)
                except Exception:
                    return float(default)

        def _fmt(val, suffix=""):
            return f"{val:.2f}{suffix}" if np.isfinite(val) else "N/A"

        # 1) 오리엔테이션: Datahub가 이미 ENU 기준 Z–X–Y 오일러 제공
        roll_z  = _last(self.datahub.rolls)   # deg
        pitch_x = _last(self.datahub.pitchs)  # deg
        yaw_y   = _last(self.datahub.yaws)    # deg

        YAW_OFFSET = 90.0
        yaw_y_adj = yaw_y + YAW_OFFSET

        self.rocket_mesh.resetTransform()
        self.rocket_mesh.rotate(roll_z,  0, 0, 1)  # Z roll
        self.rocket_mesh.rotate(pitch_x, 1, 0, 0)  # X pitch
        self.rocket_mesh.rotate(yaw_y_adj,   0, 1, 0)  # Y yaw

        # 2) 속도: Datahub.speed (ENU 기반) 바로 사용
        spd = _last(self.datahub.speed)
        self.speed_label.setText(f"Speed {_fmt(spd,'m/s')}")

        # 3) 고도 (ENU U값)
        alt = _last(self.datahub.u_enu)
        self.altitude_label.setText(f"Altitude {_fmt(alt,'m')}")

        # 4) 각도 레이블
        self.roll_label.setText(f"Roll : {_fmt(roll_z,'°')}")
        self.pitch_label.setText(f"Pitch : {_fmt(pitch_x,'°')}")
        self.yaw_label.setText(f"Yaw : {_fmt(yaw_y_adj,'°')}")

        # 5) 각속도/가속도 (rad/s, m/s²)
        self.rollspeed_label.setText(f"Roll_speed : {_fmt(_last(self.datahub.rollSpeeds),'Rad/s')}")
        self.pitchspeed_label.setText(f"Pitch_speed : {_fmt(_last(self.datahub.pitchSpeeds),'Rad/s')}")
        self.yawspeed_label.setText(f"Yaw_speed : {_fmt(_last(self.datahub.yawSpeeds),'Rad/s')}")

        self.xacc_label.setText(f"X_acc : {_fmt(_last(self.datahub.Xaccels),'m/s²')}")
        self.yacc_label.setText(f"Y_acc : {_fmt(_last(self.datahub.Yaccels),'m/s²')}")
        self.zacc_label.setText(f"Z_acc : {_fmt(_last(self.datahub.Zaccels),'m/s²')}")

    def data_label_clear(self):
        """모든 데이터 레이블을 초기 상태로 리셋"""
        self.speed_label.setText("Speed 0.00 m/s")
        self.altitude_label.setText("Altitude 0.00 m")
        self.roll_label.setText("Roll : 0.00°")
        self.pitch_label.setText("Pitch : 0.00°")
        self.yaw_label.setText("Yaw : 0.00°")
        self.rollspeed_label.setText("Roll_speed : 0.00 Rad/s")
        self.pitchspeed_label.setText("Pitch_speed : 0.00 Rad/s")
        self.yawspeed_label.setText("Yaw_speed : 0.00 Rad/s")
        self.xacc_label.setText("X_acc : 0.00 m/s²")
        self.yacc_label.setText("Y_acc : 0.00 m/s²")
        self.zacc_label.setText("Z_acc : 0.00 m/s²")

    def run(self):
        self.exec_()

class MainWindow(PageWindow):
    def __init__(self, datahub):
        super().__init__()
        self.datahub = datahub

        # ▶ 중앙 컨테이너 생성
        self.container = QWidget(self)
        self.setCentralWidget(self.container)

        # ▶ 동적 기준 디렉터리 설정 (__file__ 이 있는 폴더)
        base_dir = Path(__file__).resolve().parent
        self.dir_path = str(base_dir)     # 문자열이 필요하면 str(), 아니면 Path 객체 그대로 써도 됩니다

        self.initUI()
        self.initGraph()

        """Start Thread"""
        self.mapviewer = MapViewer_Thread(self,datahub)
        # self.graphviewer = GraphViewer_Thread(self,datahub)
        self.rocketviewer = RocketViewer_Thread(self,datahub)

        self.initMenubar()

        self.mapviewer.start()
        # self.graphviewer.start()
        self.rocketviewer.start()

        self.resetcheck = 0

        self.csv_player = None
        self.replay_csv_path = None  # 사용자가 고를 CSV 경로
        
    def initUI(self):

        """Set Buttons"""
        self.start_button = QPushButton("Start",self.container)
        self.stop_button = QPushButton("Stop",self.container)
        self.reset_button = QPushButton("Reset",self.container)

        self.now_status = QLabel(ws.wait_status,self.container)
        self.rf_port_edit = QLineEdit("COM8",self.container)
        self.port_text = QLabel("Rx_Port:",self.container)
        self.baudrate_edit = QLineEdit("115200",self.container)
        self.baudrate_text = QLabel("Rx_Baudrate:",self.container)
        self.guide_text = QLabel(ws.guide,self.container)
        self.port_text.setStyleSheet("color: white;")
        self.baudrate_text.setStyleSheet("color: white;")

        self.start_button.setFont(ws.font_start_text)
        self.stop_button.setFont(ws.font_stop_text)
        self.reset_button.setFont(ws.font_reset_text)

        self.rf_port_edit.setStyleSheet("background-color: rgb(255,255,255);")
        self.baudrate_edit.setStyleSheet("background-color: rgb(255,255,255);")
        self.start_button.setStyleSheet("background-color: rgb(200,0,0); color: rgb(250, 250, 250);font-weight: bold; font-weight: bold; border-radius: 25px;")
        self.stop_button.setStyleSheet("background-color: rgb(0,0,139); color: rgb(250, 250, 250);font-weight: bold; font-weight: bold; border-radius: 25px;")
        self.reset_button.setStyleSheet("background-color: rgb(120,120,140); color: rgb(250, 250, 250);font-weight: bold; font-weight: bold; border-radius: 25px;")

        self.shadow_start_button = QGraphicsDropShadowEffect()
        self.shadow_stop_button = QGraphicsDropShadowEffect()
        self.shadow_reset_button = QGraphicsDropShadowEffect()
        self.shadow_start_button.setOffset(6)
        self.shadow_stop_button.setOffset(6)
        self.shadow_reset_button.setOffset(6)
        self.start_button.setGraphicsEffect(self.shadow_start_button)
        self.stop_button.setGraphicsEffect(self.shadow_stop_button)
        self.reset_button.setGraphicsEffect(self.shadow_reset_button)

        self.baudrate_text.setFont(ws.font_baudrate)
        self.port_text.setFont(ws.font_portText)
        self.guide_text.setFont(ws.font_guideText)

        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.reset_button.setEnabled(False)
        self.rf_port_edit.setEnabled(True)
        self.baudrate_edit.setEnabled(True)

        """Set Buttons Connection"""
        self.start_button.clicked.connect(self.start_button_clicked)
        self.stop_button.clicked.connect(self.stop_button_clicked)
        self.reset_button.clicked.connect(self.reset_button_clicked)

        """Set Geometry"""
        self.start_button.setGeometry(*ws.start_geometry)
        self.stop_button.setGeometry(*ws.stop_geometry)
        self.reset_button.setGeometry(*ws.reset_geometry)

        self._toggle_labels = {}

        self.port_text.setGeometry(*ws.port_text_geometry)
        self.rf_port_edit.setGeometry(*ws.port_edit_geometry)
        self.baudrate_text.setGeometry(*ws.baudrate_text_geometry)
        self.baudrate_edit.setGeometry(*ws.baudrate_edit_geometry)
        self.guide_text.setGeometry(*ws.cmd_geometry)
        self.now_status.setGeometry(*ws.status_geometry)
        self.now_status.setFont(ws.font_status_text)
        self.now_status.setStyleSheet("color:#FFFFFF;")
        
        base_dir = Path(__file__).resolve().parent

        # team logo (부모를 self.container 로 변경)
        logo_path = base_dir / 'team_logo.png'
        self.team_logo = QLabel(self.container)
        self.team_logo.setPixmap(
            QPixmap(str(logo_path))
            .scaled(*ws.team_logo_geometry[2:4], Qt.KeepAspectRatio)
        )
        self.team_logo.setGeometry(*ws.team_logo_geometry)

        # rudasys logo (부모를 self.container 로 변경)
        logo_path = base_dir / 'rudasys.png'
        self.rudasys_logo = QLabel(self.container)
        self.rudasys_logo.setPixmap(
            QPixmap(str(logo_path))
            .scaled(*ws.rudasys_logo_geometry[2:4], Qt.KeepAspectRatio)
        )
        self.rudasys_logo.setGeometry(*ws.rudasys_logo_geometry)

        # irri logo
        irri_path = base_dir / 'irri.png'
        self.irri_logo = QLabel(self.container)
        self.irri_logo.setPixmap(
            QPixmap(str(irri_path))
            .scaled(*ws.irri_logo_geometry[2:4], Qt.KeepAspectRatio)
        )
        self.irri_logo.setGeometry(*ws.irri_logo_geometry)

        # patch22 logo
        patch22_path = base_dir / '22patch.png'
        self.patch22_logo = QLabel(self.container)
        self.patch22_logo.setPixmap(
            QPixmap(str(patch22_path))
            .scaled(*ws.patch22_logo_geometry[2:4], Qt.KeepAspectRatio)
        )
        self.patch22_logo.setGeometry(*ws.patch22_logo_geometry)
    
    #상단 메뉴바
    def initMenubar(self):
        self.statusBar()

        menubar = self.menuBar()
        menubar.setNativeMenuBar(False)
        # 스타일시트 수정
        self.setStyleSheet("""
            QMenuBar {
                background-color: rgb(50,50,50);
                color: rgb(255,255,255);
                border: 1px solid rgb(50,50,50);
            }
            QMenu {
                background-color: rgb(255,255,255);  /* 메뉴 드롭다운 창의 배경색 */
                color: rgb(0,0,0);          /* 메뉴 항목의 글씨 색 */
                border: 1px solid rgb(50,50,50);  /* 메뉴 드롭다운 창의 테두리 색 */
            }
            QMenu::item::selected {
                background-color: rgb(50,50,50);  /* 선택된 메뉴 항목의 배경색 */
                color: rgb(255,255,255);          /* 선택된 메뉴 항목의 글씨 색 */
            }
        """)

    def initGraph(self):
        pass
        # self.xspeed_hide_checkbox = QCheckBox("v_x", self.container)
        # self.xspeed_hide_checkbox.setStyleSheet("color: white;")

        # self.yspeed_hide_checkbox = QCheckBox("v_y", self.container)
        # self.yspeed_hide_checkbox.setStyleSheet("color: white;")

        # self.zspeed_hide_checkbox = QCheckBox("v_z", self.container)
        # self.zspeed_hide_checkbox.setStyleSheet("color: white;")

        # self.rollspeed_hide_checkbox = QCheckBox("w_x",self.container)
        # self.rollspeed_hide_checkbox.setStyleSheet("color: white;")
        
        # self.pitchspeed_hide_checkbox = QCheckBox("w_y",self.container)
        # self.pitchspeed_hide_checkbox.setStyleSheet("color: white;")
        
        # self.yawspeed_hide_checkbox = QCheckBox("w_z",self.container)
        # self.yawspeed_hide_checkbox.setStyleSheet("color: white;")

        # self.xacc_hide_checkbox = QCheckBox("a_x",self.container)
        # self.xacc_hide_checkbox.setStyleSheet("color: white;")
        
        # self.yacc_hide_checkbox = QCheckBox("a_y",self.container)
        # self.yacc_hide_checkbox.setStyleSheet("color: white;")
        
        # self.zacc_hide_checkbox = QCheckBox("a_z",self.container)
        # self.zacc_hide_checkbox.setStyleSheet("color: white;")

        # self.xspeed_hide_checkbox.setGeometry(*ws.vx_checker_geometry)
        # self.yspeed_hide_checkbox.setGeometry(*ws.vy_checker_geometry)
        # self.zspeed_hide_checkbox.setGeometry(*ws.vz_checker_geometry)

        # self.rollspeed_hide_checkbox.setGeometry(*ws.rollS_checker_geomoetry)
        # self.pitchspeed_hide_checkbox.setGeometry(*ws.pitchS_checker_geomoetry)
        # self.yawspeed_hide_checkbox.setGeometry(*ws.yawS_checker_geomoetry)

        # self.xacc_hide_checkbox.setGeometry(*ws.ax_checker_geomoetry)
        # self.yacc_hide_checkbox.setGeometry(*ws.ay_checker_geomoetry)
        # self.zacc_hide_checkbox.setGeometry(*ws.az_checker_geomoetry)

        # self.xacc_hide_checkbox.setFont(ws.checker_font)
        # self.yacc_hide_checkbox.setFont(ws.checker_font)
        # self.zacc_hide_checkbox.setFont(ws.checker_font)

        # self.xspeed_hide_checkbox.stateChanged.connect(self.xspeed_hide_checkbox_state)
        # self.yspeed_hide_checkbox.stateChanged.connect(self.yspeed_hide_checkbox_state)
        # self.zspeed_hide_checkbox.stateChanged.connect(self.zspeed_hide_checkbox_state)
        # self.rollspeed_hide_checkbox.stateChanged.connect(self.rollspeed_hide_checkbox_state)
        # self.pitchspeed_hide_checkbox.stateChanged.connect(self.pitchspeed_hide_checkbox_state)
        # self.yawspeed_hide_checkbox.stateChanged.connect(self.yawspeed_hide_checkbox_state)
        # self.xacc_hide_checkbox.stateChanged.connect(self.xacc_hide_checkbox_state)
        # self.yacc_hide_checkbox.stateChanged.connect(self.yacc_hide_checkbox_state)
        # self.zacc_hide_checkbox.stateChanged.connect(self.zacc_hide_checkbox_state)

        # self.xspeed_hide_checkbox.setFont(ws.checker_font)
        # self.yspeed_hide_checkbox.setFont(ws.checker_font)
        # self.zspeed_hide_checkbox.setFont(ws.checker_font)
        # self.rollspeed_hide_checkbox.setFont(ws.checker_font)
        # self.pitchspeed_hide_checkbox.setFont(ws.checker_font)
        # self.yawspeed_hide_checkbox.setFont(ws.checker_font)

    # Run when start button is clicked
    def start_button_clicked(self):
        if self.resetcheck == 0:
            self.datahub.clear()
            
            # 스타일 시트를 이용해 배경색과 글씨 색 모두 설정
            self.setStyleSheet("""
                QMessageBox {
                    background-color: black;
                    color: white;
                }
                QLabel {
                    background-color: black;           
                    color: white;
                }
                QPushButton {
                    background-color: white;
                    color: black;
                }
                QInputDialog {
                    background-color: black;
                    color: black;
                }
                QLineEdit {
                    background-color: white;  /* QLineEdit의 배경색을 흰색으로 설정 */
                    color: black;  /* 입력 글씨 색상을 화이트로 설정 */
                }
             """)

            msg_box = QMessageBox(self)
            msg_box.setStyleSheet("QMessageBox {background-color: white;}")
            msg_box.information(self, "information", "Program Start")

            input_dialog = QInputDialog(self)
            input_dialog.setStyleSheet("QInputDialog {background-color: white;}")
            FileName, ok = input_dialog.getText(
                self, 'Input Dialog', 'Enter your File Name', QLineEdit.Normal, "Your File Name"
            )
            file_dir = os.path.join(os.path.dirname(self.dir_path), "log")
            os.makedirs(file_dir, exist_ok=True)
            file_path = os.path.join(file_dir, FileName) + ".csv"

            port_text = self.rf_port_edit.text().strip().upper()

            if port_text == "CSV":
                # 선택한 CSV 파일 열기…
                start_dir = os.path.join(dirname(self.dir_path), "log")
                if not os.path.isdir(start_dir):
                    start_dir = dirname(self.dir_path)

                csv_path, _ = QFileDialog.getOpenFileName(
                    self, "Select CSV to Replay", start_dir, "CSV Files (*.csv);;All Files (*)"
                )

                self.replay_csv_path = csv_path

                # 2) 통신 플래그/저장기 시작은 그대로 (UI 상태용)
                self.datahub.communication_start()
                self.datahub.datasaver_start()
                self.now_status.setText(ws.start_status)
                self.start_button.setEnabled(False)
                self.stop_button.setEnabled(True)
                self.rf_port_edit.setEnabled(False)
                self.baudrate_edit.setEnabled(False)
                self.shadow_start_button.setOffset(0)
                self.shadow_stop_button.setOffset(6)
                self.shadow_reset_button.setOffset(6)

                # 3) CSVPlayer 시작
                try:
                    self.csv_player = CSVPlayer(self.replay_csv_path, self.datahub, hz=5.0, parent=self)
                    # self.csv_player.sampleReady.connect(self.graphviewer.update_data)
                    self.csv_player.start()
                except Exception as e:
                    print(f"[CSVPlayer] start error: {e}")
                    QMessageBox.warning(self, "warning", "CSV 재생 시작 실패")
                    self.datahub.communication_stop()
                    self.datahub.datasaver_stop()
                    # UI 복구
                    self.start_button.setEnabled(True)
                    self.stop_button.setEnabled(False)
                    self.rf_port_edit.setEnabled(True)
                    self.baudrate_edit.setEnabled(True)
                return  # CSV 모드는 여기서 종료

            else:
                if os.path.exists(file_path):
                    QMessageBox.information(self, "information", "Same file already exist")

                if ok:
                    self.datahub.mySerialPort = self.rf_port_edit.text()
                    self.datahub.myBaudrate   = self.baudrate_edit.text()
                    self.datahub.file_Name    = FileName + '.csv'

                    # 1) 통신 시작
                    self.datahub.serial_port_error = -1
                    self.datahub.communication_start()

                    # 2) "알 수 없음" 상태로 초기화 후 Receiver의 판정 기다리기
                    t0 = time.perf_counter()
                    timeout_s = 2.5
                    while (self.datahub.serial_port_error == -1
                        and (time.perf_counter() - t0) < timeout_s):
                        QApplication.processEvents()
                        time.sleep(0.01)

                    status = self.datahub.serial_port_error

                    if status != 0:
                        # 실패(1) 또는 타임아웃(-1 그대로) → 에러 처리
                        QMessageBox.warning(self, "warning", "Check the Port or Baudrate again.")
                        self.datahub.communication_stop()
                        # UI 복구
                        self.start_button.setEnabled(True)
                        self.stop_button.setEnabled(False)
                        self.rf_port_edit.setEnabled(True)
                        self.baudrate_edit.setEnabled(True)
                    else:
                        # 성공
                        self.datahub.datasaver_start()
                        self.now_status.setText(ws.start_status)
                        self.start_button.setEnabled(False)
                        self.stop_button.setEnabled(True)
                        self.rf_port_edit.setEnabled(False)
                        self.baudrate_edit.setEnabled(False)
                        self.shadow_start_button.setOffset(0)
                        self.shadow_stop_button.setOffset(6)
                        self.shadow_reset_button.setOffset(6)

        else:
            # reset 후 재시작 분기도 동일하게 대기/판정 처리 권장
            self.datahub.communication_start()
            self.datahub.serial_port_error = -1
            t0 = time.perf_counter()
            timeout_s = 2.5
            while (self.datahub.serial_port_error == -1
                and (time.perf_counter() - t0) < timeout_s):
                QApplication.processEvents()
                time.sleep(0.01)

            status = self.datahub.serial_port_error
            if status != 0:
                QMessageBox.warning(self, "warning", "Check the Port or Baudrate again.")
                self.datahub.communication_stop()
                return

            self.now_status.setText(ws.start_status)
            self.now_status.setStyleSheet("color:#FFFFFF;")
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.rf_port_edit.setEnabled(False)
            self.baudrate_edit.setEnabled(False)
            self.shadow_start_button.setOffset(0)
            self.shadow_stop_button.setOffset(6)
            self.shadow_reset_button.setOffset(6)
            self.resetcheck = 0

    # Run when stop button is clicked
    def stop_button_clicked(self):
        # CSV 모드라면 재생 중지
        if self.csv_player is not None:
            try:
                self.csv_player.stop()
                self.csv_player.wait(1000)
            except Exception:
                pass
            self.csv_player = None

        self.datahub.communication_stop()
        self.now_status.setText(ws.stop_status)
        self.now_status.setStyleSheet("color:#FFFFFF;")
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.reset_button.setEnabled(True)
        self.rf_port_edit.setEnabled(True)
        self.baudrate_edit.setEnabled(True)
        self.shadow_start_button.setOffset(6)
        self.shadow_stop_button.setOffset(0)
        self.shadow_reset_button.setOffset(6)
        self.resetcheck = 1

    def reset_button_clicked(self):
        request = QMessageBox.question(self,'Message', 'Are you sure to reset?')
        if request == QMessageBox.Yes:
            # CSV 모드라면 재생 중지
            if self.csv_player is not None:
                try:
                    self.csv_player.stop()
                    self.csv_player.wait(1000)
                except Exception:
                    pass
            self.csv_player = None

            msg_box = QMessageBox(self)
            msg_box.setStyleSheet("QMessageBox {background-color: white;}")
            msg_box.information(self,"information","Program Reset")
            self.datahub.communication_stop()
            self.datahub.datasaver_stop()
            self.now_status.setText(ws.wait_status)
            self.now_status.setStyleSheet("color:#FFFFFF;")
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.reset_button.setEnabled(False)
            self.rf_port_edit.setEnabled(False)
            self.shadow_start_button.setOffset(6)
            self.shadow_stop_button.setOffset(0)
            self.shadow_reset_button.setOffset(0)
            # self.graphviewer.graph_clear()
            self.rocketviewer.data_label_clear()
            self.datahub.clear()
            self.resetcheck = 0
        else:
            msg_box = QMessageBox(self)
            msg_box.setStyleSheet("QMessageBox {background-color: white;}")
            msg_box.information(self,"information","Cancel")


    #curve hide check box is clicked
    # def xspeed_hide_checkbox_state(self,state):
    #     self.graphviewer.curve_xspeed.setVisible(state != Qt.Checked)
    # def yspeed_hide_checkbox_state(self,state):
    #     self.graphviewer.curve_yspeed.setVisible(state != Qt.Checked)
    # def zspeed_hide_checkbox_state(self,state):
    #     self.graphviewer.curve_zspeed.setVisible(state != Qt.Checked)        
    # def rollspeed_hide_checkbox_state(self,state):
    #     self.graphviewer.curve_rollSpeed.setVisible(state != Qt.Checked)
    # def pitchspeed_hide_checkbox_state(self,state):
    #     self.graphviewer.curve_pitchSpeed.setVisible(state != Qt.Checked)
    # def yawspeed_hide_checkbox_state(self,state):
    #     self.graphviewer.curve_yawSpeed.setVisible(state != Qt.Checked)
    # def xacc_hide_checkbox_state(self,state):
    #     self.graphviewer.curve_xaccel.setVisible(state != Qt.Checked)
    # def yacc_hide_checkbox_state(self,state):
    #     self.graphviewer.curve_yaccel.setVisible(state != Qt.Checked)
    # def zacc_hide_checkbox_state(self,state):
    #     self.graphviewer.curve_zaccel.setVisible(state != Qt.Checked)

class CSVPlayer(QThread):
    sampleReady = pyqtSignal()

    def __init__(self, csv_path, datahub, hz=5.0, parent=None):
        super().__init__(parent)
        self.csv_path = csv_path
        self.datahub = datahub
        self.hz = float(hz)
        self._running = True
        self._paused = False

        # 시간 처리용 내부 상태
        self._ten_div = 100.0   # tenmilis 스케일(기본 10ms → /100, ms면 /1000)
        self._epoch = None
        self._prev_raw = None
        self._t0 = None

        # 성능 튜닝 파라미터
        self._BATCH  = 200      # 샘플 N개마다 묶음 반영
        self._UI_FPS = 30.0     # UI 갱신 최대 FPS

    def stop(self):
        self._running = False
        # pause 루프에서 즉시 빠져나오도록
        self._paused = False

    def pause(self, yes=True):
        self._paused = bool(yes)

    def _abs_time_val(self, h, m, s, ten):
        return h*3600.0 + m*60.0 + s + ten/self._ten_div

    def _abs_time_row(self, row):
        h, m, s, ten = row[0], row[1], row[2], row[3]
        return self._abs_time_val(h, m, s, ten)

    def _decide_ten_div(self, cleaned):
        try:
            ten_max = max(r[3] for r in cleaned)
            self._ten_div = 1000.0 if ten_max > 120.0 else 100.0
        except Exception:
            self._ten_div = 100.0

    def _feed_monotonic_time_to_buf(self, h, m, s, ten, buf_t):
        """단조 증가 시간 t를 계산해 버퍼에만 push."""
        raw = self._abs_time_val(h, m, s, ten)

        if self._epoch is None:
            self._epoch = 0.0
            self._prev_raw = raw
            self._t0 = raw

        # 시계가 뒤로 감기면 epoch로 이어붙임
        if raw < self._prev_raw - 0.2:
            drop = self._prev_raw - raw
            self._epoch += 24*3600.0 if drop > 3600.0 else drop

        self._prev_raw = raw
        t_mono = (raw - self._t0) + self._epoch
        buf_t.append(float(t_mono))

    def run(self):
        try:
            # 1) CSV 로드 (헤더 유무와 무관하게 20열 강제 매핑)
            try:
                df = pd.read_csv(self.csv_path, header=None, names=COLUMN_NAMES, dtype=str)
            except Exception as e:
                #print("[CSVPlayer] FATAL while reading CSV:", e)
                traceback.print_exc()
                return

            # 2) 전처리: 숫자 캐스팅 & 열 개수 보정
            cleaned = []
            for row in df.itertuples(index=False, name=None):
                if len(row) < len(COLUMN_NAMES):
                    continue
                try:
                    vals = [float(x) for x in row[:len(COLUMN_NAMES)]]
                except Exception:
                    continue
                cleaned.append(vals)

            if not cleaned:
                #print("[CSVPlayer] no valid rows after cleaning")
                return

            # 3) tenmilis 스케일 판별 & 시간 기준 정렬
            self._decide_ten_div(cleaned)
            cleaned.sort(key=self._abs_time_row)

            # 4) 재생 루프 준비
            dt = 1.0 / max(1e-3, self.hz)
            next_t = time.perf_counter()

            buf_rows, buf_t = [], []

            # UI 스로틀
            ui_dt = 1.0 / self._UI_FPS
            next_ui = time.perf_counter()

            def flush_batch():
                """모아둔 t/rows를 한 번에 datahub에 반영 (락은 내부에서 처리)."""
                if not buf_rows:
                    return
                t_arr = np.asarray(buf_t, dtype=float)
                # 시간 → 데이터 순으로 배치 반영
                self.datahub.append_time_batch(t_arr)
                self.datahub.batch_update(buf_rows)
                buf_rows.clear()
                buf_t.clear()

            # 5) 메인 재생 루프
            for vals in cleaned:
                if not self._running:
                    break

                while self._paused and self._running:
                    time.sleep(0.005)

                # 단조시간 버퍼링
                try:
                    h, m, s, ten = vals[0], vals[1], vals[2], vals[3]
                    self._feed_monotonic_time_to_buf(h, m, s, ten, buf_t)
                except Exception as e:
                    pass
                    #print(f"[CSVPlayer] time gen error: {e}")

                # 데이터 버퍼링
                buf_rows.append(vals)
                if len(buf_rows) >= self._BATCH:
                    flush_batch()

                # UI 스로틀
                now = time.perf_counter()
                if now >= next_ui:
                    flush_batch()
                    self.sampleReady.emit()
                    next_ui = now + ui_dt

                # 재생 속도 제어
                next_t += dt
                sleep_for = next_t - time.perf_counter()
                if sleep_for > 0:
                    time.sleep(0.005)
                else:
                    next_t = time.perf_counter()

            # 6) 종료 전 잔여 플러시 + 최종 UI 갱신
            flush_batch()
            self.sampleReady.emit()

        except Exception as e:
            #print("[CSVPlayer] FATAL:", e)
            traceback.print_exc()

class TimeAxisItem(AxisItem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setLabel(text='Time', units=None)
        self.enableAutoSIPrefix(False)

    def tickStrings(self, values, scale, spacing):
        return [str(timedelta(milliseconds = millis))[:-4] for millis in values]

class window(QMainWindow):
    def __init__(self,datahub):
        self.app = QApplication(argv)
        super().__init__()
        self.datahub = datahub

        self.initUI()
        self.initWindows()

    def initUI(self):
        self.resize(*ws.full_size)
        self.setWindowTitle('I-link')
        self.setStyleSheet(ws.mainwindow_color) 

        path = abspath(__file__)
        dir_path = dirname(path)
        file_path = join(dir_path, 'window_logo.ico')
        self.setWindowIcon(QIcon(file_path))


    def initWindows(self):
        self.mainwindow = MainWindow(self.datahub)

        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)
        self.stacked_widget.addWidget(self.mainwindow)

    def start(self):
        self.show()
        
    def setEventLoop(self):
        exit(self.app.exec_())

    def closeEvent(self, event):
        self.datahub.communication_stop()
        self.datahub.datasaver_stop()

        event.accept()