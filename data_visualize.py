import os
import numpy as np
import matplotlib.pyplot as plt
import vispy
from vispy.scene import visuals
from vispy.scene.cameras import TurntableCamera
from vispy.scene import SceneCanvas
from pcprocessing import FilterIntentisyThr, FilterROI, FilterLidarVal, separate_LR, FilterROI7

class Vis():
  def __init__(self, data_folder, lane_folder, visulize_lane_line):
    self.index = 0
    self.lidar_paths, self.lane_paths = self.read_data(data_folder, lane_folder)

    self.canvas = SceneCanvas(keys='interactive',
                              show=True,
                              size=(1600, 900))
    self.canvas.events.key_press.connect(self._key_press)
    self.canvas.events.draw.connect(self._draw)

    self.grid = self.canvas.central_widget.add_grid()
    self.scan_view = vispy.scene.widgets.ViewBox(parent=self.canvas.scene,
                                                 camera=TurntableCamera(distance=30.0))
    self.grid.add_widget(self.scan_view)
    self.scan_vis = visuals.Markers()
    self.scan_view.add(self.scan_vis)
    visuals.XYZAxis(parent=self.scan_view.scene)

    self.visulize_lane_line = visulize_lane_line

    # lane_line
    self.lane_line = vispy.scene.visuals.Line(parent=self.scan_view.scene)
    self.lane_line.visible = False
    self.update_scan()

  def read_data(self, data_folder, lane_folder):
    lidar_files = sorted(os.listdir(data_folder))
    lidar_paths = [os.path.join(data_folder, f) for f in lidar_files]
    lane_paths = []
    for lidar_file in lidar_files:

      lane_path = os.path.join(lane_folder, lidar_file.replace("bin", "txt"))

      if os.path.isfile(lane_path):
        lane_paths.append(lane_path)
      else:
        lane_paths.append(None)
    return lidar_paths, lane_paths

  def read_points(self, lidar_path):
    pc = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)
    #pc = ground_points(pc)
    pc = FilterROI(pc)
    #pc = FilterROI7(pc)
    pc = FilterIntentisyThr(pc)
    #pc = AdapIntentisyThr(pc)
    pc = FilterLidarVal(pc)
    #pc = FilterLidarVal2(pc,57)
    #pc = Flatten(pc)
    #pc = separate_LR(pc)
    #pc = separate_LR(pc)[1]
    return pc

  def get_point_color_using_intensity(self, points):
    #print(points.shape)
    scale_factor = 10
    scaled_intensity = np.clip(points[:, 3] * scale_factor, 0, 255)
    scaled_intensity = scaled_intensity.astype(np.uint8)
    cmap = plt.get_cmap("viridis")
    # Initialize the matplotlib color map
    sm = plt.cm.ScalarMappable(cmap=cmap)

    # Obtain linear color range
    color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:, 2::-1]

    color_range = color_range.reshape(256, 3).astype(np.float32) / 255.0

    colors = color_range[scaled_intensity]
    return colors

  def plot_lane(self, left_lane_coef, right_lane_coef):
    self.lane_line.visible = True
    connect = []
    x_max = 40
    num = 80
    xs = np.linspace(start=-x_max, stop=x_max, num=num, endpoint=True)
    left_ys = left_lane_coef[-1]
    power = len(left_lane_coef) - 1
    for i, coef in enumerate(left_lane_coef[:-1]):
      left_ys += np.power(xs, power - i) * coef
    right_ys = right_lane_coef[-1]
    for i, coef in enumerate(right_lane_coef[:-1]):
      right_ys += np.power(xs, power - i) * coef

    left_lane = np.stack([xs, left_ys, np.zeros_like(xs, dtype=np.float32)], axis=-1)
    right_lane = np.stack([xs, right_ys, np.zeros_like(xs, dtype=np.float32)], axis=-1)
    lane = np.concatenate([left_lane, right_lane], axis=0)

    connect = [[i, i + 1] for i in range(len(xs) - 1)] +\
              [[i, i + 1] for i in range(len(xs), 2 * len(xs) - 1)]
    connect = np.array(connect)
    self.lane_line.set_data(pos=lane,
                           connect=connect,
                           color=np.array([1, 1, 1, 1]))

  def update_scan(self):
    lidar_path = self.lidar_paths[self.index]
    points = self.read_points(lidar_path)

    colors = self.get_point_color_using_intensity(points)

    self.canvas.title = f"Frame: {self.index} / {len(self.lidar_paths)} - {lidar_path}"
    self.scan_vis.set_data(points[:, :3],
                           face_color=colors,
                           edge_color=colors,
                           size=1.0)

    if not self.visulize_lane_line:
      return
    lane_path = self.lane_paths[self.index]
    if lane_path is not None:
      with open(lane_path, "r") as f:
        left_lane = f.readline()
        left_lane = [float(x) for x in left_lane.strip().split(";")]
        right_lane = f.readline()
        right_lane = [float(x) for x in right_lane.strip().split(";")]
      self.plot_lane(left_lane, right_lane)
    else:
      self.lane_line.visible = False

  def _key_press(self, event):
    if event.key == 'N':
      if self.index < len(self.lidar_paths) - 1:
        self.index += 1
      self.update_scan()

    if event.key == 'B':
      if self.index > 0:
        self.index -= 1
      self.update_scan()

    if event.key == 'Q':
      self.destroy()

  def destroy(self):
    # destroy the visualization
    self.canvas.close()
    vispy.app.quit()

  def _draw(self, event):
    if self.canvas.events.key_press.blocked():
      self.canvas.events.key_press.unblock()

  def run(self):
    self.canvas.app.run()


if __name__ == "__main__":
  data_folder = "C:/Users/SamSung/Desktop/projects2/Seoul_Robotics_lane_detection/Seoul_Robotics_lane_detection/pointclouds"
  lane_folder = "C:/Users/SamSung/Desktop/projects2/Seoul_Robotics_lane_detection/for_submission/sample_output"
  visulize_lane_line = False  # Visualize lane line if available
  vis = Vis(data_folder, lane_folder, visulize_lane_line)
  vis.run()
  #v = vis.read_data(data_folder, lane_folder)
