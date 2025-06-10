import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import json
import cv2
import numpy as np
from ultralytics import YOLO
from time import time
import math
from scipy.signal import savgol_filter  

def calculate_zones(camera_height, camera_angle, frame_height, num_zones=10, fov=60, tilt=0, yaw=0, sensor_size=0.0048, focal_length=0.0035):
    zones = []
    fov_rad = math.radians(fov)
    tilt_rad = math.radians(tilt)
    yaw_rad = math.radians(yaw)

    total_visible_length = 2 * camera_height * math.tan(fov_rad / 2) / math.cos(tilt_rad)
    ppm_base = frame_height / total_visible_length

    for i in range(num_zones):
        y_start = int(frame_height * (i / num_zones) ** 1.5)
        y_end = int(frame_height * ((i + 1) / num_zones) ** 1.5)
        y_center = (y_start + y_end) / 2
        pixel_relative = 1 - y_center / frame_height

        scale_factor = math.cos(tilt_rad) / (1 + pixel_relative * math.tan(tilt_rad))
        ppm = ppm_base * scale_factor * (focal_length / sensor_size) * math.cos(yaw_rad)
        zones.append({
            "y_range": (y_start, y_end),
            "ppm": round(ppm, 2)
        })

    return zones

def get_ppm_by_y_and_lane(y, x, zones, lanes):
    lane_factor = 1.0
    for lane in lanes:
        if lane["x_range"][0] <= x < lane["x_range"][1]:
            lane_factor = lane["ppm_scale"]
            break

    for zone in zones:
        if zone["y_range"][0] <= y < zone["y_range"][1]:
            return zone["ppm"] * lane_factor
    return zones[-1]["ppm"] * lane_factor

def estimate_speed(prev_point, current_point, time_diff, y_pos, x_pos, zones, lanes):
    if time_diff == 0:
        return 0
    ppm = get_ppm_by_y_and_lane(y_pos, x_pos, zones, lanes)
    distance_pixels = np.linalg.norm(np.array(current_point) - np.array(prev_point))
    distance_meters = distance_pixels / ppm
    return (distance_meters / time_diff) * 3.6  

def smooth_speed(speeds, window_size=5):
    if len(speeds) < window_size:
        return speeds[-1]
    smoothed = savgol_filter(speeds, window_size, 2)
    return smoothed[-1]

def process_video_with_config(video_path, speed_line_y, zones, lanes, show_zones=False, apply_distortion_correction=False, distortion_coeffs=None):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            messagebox.showerror("Error", f"Cannot open video file: {video_path}")
            return

        model = YOLO("/home/mykola/python/yolo/cp/speed_detection/runs/detect/train/weights/best.pt")
        track_history = {}
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (1020, 500))

            if apply_distortion_correction and distortion_coeffs is not None:
                frame = cv2.undistort(frame, np.array([[distortion_coeffs['fx'], 0, distortion_coeffs['cx']],
                                                       [0, distortion_coeffs['fy'], distortion_coeffs['cy']],
                                                       [0, 0, 1]]), np.array(distortion_coeffs['k']))

            cv2.line(frame, (0, speed_line_y), (1020, speed_line_y), (0, 0, 255), 2)

            for lane in lanes:
                cv2.line(frame, (lane["x_range"][0], 0), (lane["x_range"][0], 500), (255, 255, 0), 1)
                cv2.line(frame, (lane["x_range"][1], 0), (lane["x_range"][1], 500), (255, 255, 0), 1)

            if show_zones:
                for zone in zones:
                    y_start, y_end = zone["y_range"]
                    cv2.line(frame, (0, y_start), (1020, y_start), (0, 255, 255), 1)
                    cv2.putText(frame, f"PPM: {zone['ppm']}", (10, y_start + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            results = model.track(frame, persist=True, classes=[0])
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                ids = results[0].boxes.id.int().cpu().tolist()

                for box, obj_id in zip(boxes, ids):
                    x1, y1, x2, y2 = box
                    center = ((x1 + x2) / 2, (y1 + y2) / 2)
                    current_time = time()

                    if obj_id not in track_history:
                        track_history[obj_id] = {
                            "prev_pos": center,
                            "time": current_time,
                            "speed_history": [],
                            "passed": False
                        }
                    else:
                        prev_pos = track_history[obj_id]["prev_pos"]
                        prev_time = track_history[obj_id]["time"]
                        if not track_history[obj_id]["passed"] and prev_pos[1] < speed_line_y <= center[1]:
                            speed = estimate_speed(prev_pos, center, current_time - prev_time, int(center[1]), center[0], zones, lanes)
                            track_history[obj_id]["speed_history"].append(speed)
                            track_history[obj_id]["speed"] = smooth_speed(track_history[obj_id]["speed_history"])
                            track_history[obj_id]["passed"] = True

                        if track_history[obj_id]["passed"]:
                            cv2.putText(frame, f"{int(track_history[obj_id]['speed'])} km/h",
                                        (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    track_history[obj_id]["prev_pos"] = center
                    track_history[obj_id]["time"] = current_time
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

            cv2.imshow("Vehicle Speed Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            frame_count += 1
            if frame_count % 100 == 0:
                track_history = {k: v for k, v in track_history.items() if current_time - v["time"] < 5}

        cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")

class SpeedDetectionConfigurator:
    def __init__(self, root):
        self.root = root
        self.root.title("Vehicle Speed Detection Configurator")

        self.presets = {
            "Urban Traffic": {"height": 6.0, "angle": 35, "speed_line": 330, "fov": 60, "tilt": 0, "yaw": 0, "lanes": [{"x_range": (0, 1020), "ppm_scale": 1.0}]},
            "Overhead Bridge": {"height": 10.0, "angle": 50, "speed_line": 300, "fov": 60, "tilt": 0, "yaw": 0, "lanes": [{"x_range": (0, 1020), "ppm_scale": 1.0}]},
            "Multi-Lane Highway": {
                "height": 8.0, "angle": 40, "speed_line": 350, "fov": 70, "tilt": 5, "yaw": 0,
                "lanes": [
                    {"x_range": (0, 340), "ppm_scale": 1.1}, 
                    {"x_range": (340, 680), "ppm_scale": 1.0}, 
                    {"x_range": (680, 1020), "ppm_scale": 0.9}  
                ]
            }
        }

        self.config = {
            "video_path": "speed.mp4",
            "speed_line_y": 330,
            "camera_height": 6.0,
            "camera_angle": 35,
            "frame_height": 500,
            "fov": 60,
            "tilt": 0,
            "yaw": 0,
            "sensor_size": 0.0048, 
            "focal_length": 0.0035, 
            "num_zones": 10,
            "lanes": [{"x_range": (0, 1020), "ppm_scale": 1.0}],
            "distortion_correction": False,
            "distortion_coeffs": {"fx": 1000, "fy": 1000, "cx": 510, "cy": 250, "k": [0, 0, 0, 0]},
            "show_zones": False
        }

        self.create_widgets()
        self.update_zones_from_camera_params()
        self.load_config_to_ui()

    def create_widgets(self):
        tk.Label(self.root, text="Video Path:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        self.video_path_entry = tk.Entry(self.root, width=40)
        self.video_path_entry.grid(row=0, column=1, padx=5, pady=5)
        tk.Button(self.root, text="Browse", command=self.browse_video).grid(row=0, column=2, padx=5, pady=5)

        preset_frame = tk.Frame(self.root)
        preset_frame.grid(row=1, column=0, columnspan=3, pady=5)
        tk.Label(preset_frame, text="Video Preset:").pack(side="left")
        self.preset_var = tk.StringVar()
        self.preset_menu = ttk.Combobox(preset_frame, textvariable=self.preset_var, values=list(self.presets.keys()))
        self.preset_menu.pack(side="left")
        self.preset_menu.bind("<<ComboboxSelected>>", self.apply_preset)

        camera_frame = tk.LabelFrame(self.root, text="Camera Parameters", padx=5, pady=5)
        camera_frame.grid(row=2, column=0, columnspan=3, padx=5, pady=5, sticky="ew")

        tk.Label(camera_frame, text="Frame Height (px):").grid(row=0, column=4, padx=5, pady=2, sticky="e")
        self.frame_height_entry = tk.Entry(camera_frame, width=10)
        self.frame_height_entry.grid(row=0, column=5, padx=5, pady=2, sticky="w")
        self.frame_height_entry.insert(0, "500")

        tk.Label(camera_frame, text="Height (m):").grid(row=0, column=0, padx=5, pady=2, sticky="e")
        self.camera_height_entry = tk.Entry(camera_frame, width=10)
        self.camera_height_entry.grid(row=0, column=1, padx=5, pady=2, sticky="w")
        self.camera_height_entry.insert(0, "6.0") 

        tk.Label(camera_frame, text="Angle (deg):").grid(row=0, column=2, padx=5, pady=2, sticky="e")
        self.camera_angle_entry = tk.Entry(camera_frame, width=10)
        self.camera_angle_entry.grid(row=0, column=3, padx=5, pady=2, sticky="w")
        self.camera_angle_entry.insert(0, "35")

        tk.Label(camera_frame, text="FOV (deg):").grid(row=0, column=4, padx=5, pady=2, sticky="e")
        self.fov_entry = tk.Entry(camera_frame, width=10)
        self.fov_entry.grid(row=0, column=5, padx=5, pady=2, sticky="w")
        self.fov_entry.insert(0, "60")

        tk.Label(camera_frame, text="Tilt (deg):").grid(row=1, column=0, padx=5, pady=2, sticky="e")
        self.tilt_entry = tk.Entry(camera_frame, width=10)
        self.tilt_entry.grid(row=1, column=1, padx=5, pady=2, sticky="w")
        self.tilt_entry.insert(0, "0")

        tk.Label(camera_frame, text="Yaw (deg):").grid(row=1, column=2, padx=5, pady=2, sticky="e")
        self.yaw_entry = tk.Entry(camera_frame, width=10)
        self.yaw_entry.grid(row=1, column=3, padx=5, pady=2, sticky="w")
        self.yaw_entry.insert(0, "0")

        tk.Label(camera_frame, text="Sensor Size (mm):").grid(row=1, column=4, padx=5, pady=2, sticky="e")
        self.sensor_size_entry = tk.Entry(camera_frame, width=10)
        self.sensor_size_entry.grid(row=1, column=5, padx=5, pady=2, sticky="w")
        self.sensor_size_entry.insert(0, "4.8")

        tk.Label(camera_frame, text="Focal Length (mm):").grid(row=2, column=0, padx=5, pady=2, sticky="e")
        self.focal_length_entry = tk.Entry(camera_frame, width=10)
        self.focal_length_entry.grid(row=2, column=1, padx=5, pady=2, sticky="w")
        self.focal_length_entry.insert(0, "3.5")

        tk.Label(camera_frame, text="Num Zones:").grid(row=2, column=2, padx=5, pady=2, sticky="e")
        self.num_zones_spin = tk.Spinbox(camera_frame, from_=1, to=20, width=10)
        self.num_zones_spin.grid(row=2, column=3, padx=5, pady=2, sticky="w")
        self.num_zones_spin.delete(0, tk.END)
        self.num_zones_spin.insert(0, "10")

        tk.Button(camera_frame, text="Update Zones", command=self.update_zones_from_camera_params).grid(row=2, column=4, columnspan=2, padx=5, pady=2)

        lane_frame = tk.LabelFrame(self.root, text="Lane Configuration", padx=5, pady=5)
        lane_frame.grid(row=3, column=0, columnspan=3, padx=5, pady=5, sticky="ew")
        tk.Label(lane_frame, text="Number of Lanes:").grid(row=0, column=0, padx=5, pady=2, sticky="e")
        self.num_lanes_spin = tk.Spinbox(lane_frame, from_=1, to=6, width=10, command=self.update_lane_entries)
        self.num_lanes_spin.grid(row=0, column=1, padx=5, pady=2, sticky="w")
        self.lane_entries = []
        self.update_lane_entries()

        tk.Label(self.root, text="Speed Line Y:").grid(row=4, column=0, padx=5, pady=5, sticky="e")
        self.speed_line_y_spin = tk.Spinbox(self.root, from_=1, to=1000, width=10) 
        self.speed_line_y_spin.grid(row=4, column=1, padx=5, pady=5, sticky="w")
        self.speed_line_y_spin.insert(0, "330") 

        distortion_frame = tk.LabelFrame(self.root, text="Lens Distortion Correction", padx=5, pady=5)
        distortion_frame.grid(row=5, column=0, columnspan=3, padx=5, pady=5, sticky="ew")
        self.distortion_var = tk.BooleanVar()
        tk.Checkbutton(distortion_frame, text="Apply Distortion Correction", variable=self.distortion_var).grid(row=0, column=0, columnspan=2, padx=5, pady=2)
        tk.Label(distortion_frame, text="fx:").grid(row=1, column=0, padx=5, pady=2, sticky="e")
        self.fx_entry = tk.Entry(distortion_frame, width=10)
        self.fx_entry.grid(row=1, column=1, padx=5, pady=2, sticky="w")
        self.fx_entry.insert(0, "1000")
        tk.Label(distortion_frame, text="fy:").grid(row=1, column=2, padx=5, pady=2, sticky="e")
        self.fy_entry = tk.Entry(distortion_frame, width=10)
        self.fy_entry.grid(row=1, column=3, padx=5, pady=2, sticky="w")
        self.fy_entry.insert(0, "1000")
        tk.Label(distortion_frame, text="cx:").grid(row=2, column=0, padx=5, pady=2, sticky="e")
        self.cx_entry = tk.Entry(distortion_frame, width=10)
        self.cx_entry.grid(row=2, column=1, padx=5, pady=2, sticky="w")
        self.cx_entry.insert(0, "510")
        tk.Label(distortion_frame, text="cy:").grid(row=2, column=2, padx=5, pady=2, sticky="e")
        self.cy_entry = tk.Entry(distortion_frame, width=10)
        self.cy_entry.grid(row=2, column=3, padx=5, pady=2, sticky="w")
        self.cy_entry.insert(0, "250")
        tk.Label(distortion_frame, text="Distortion Coeffs (k1,k2,p1,p2):").grid(row=3, column=0, padx=5, pady=2, sticky="e")
        self.k_entry = tk.Entry(distortion_frame, width=20)
        self.k_entry.grid(row=3, column=1, columnspan=3, padx=5, pady=2, sticky="w")
        self.k_entry.insert(0, "0,0,0,0")

        vis_frame = tk.LabelFrame(self.root, text="Visualization", padx=5, pady=5)
        vis_frame.grid(row=6, column=0, columnspan=3, padx=5, pady=5, sticky="ew")
        self.show_zones_var = tk.BooleanVar()
        tk.Checkbutton(vis_frame, text="Show Zones", variable=self.show_zones_var).grid(row=0, column=0, padx=5, pady=2)

        button_frame = tk.Frame(self.root)
        button_frame.grid(row=7, column=0, columnspan=3, pady=10)
        tk.Button(button_frame, text="Save Config", command=self.save_config).pack(side="left", padx=10)
        tk.Button(button_frame, text="Load Config", command=self.load_config).pack(side="left", padx=10)
        tk.Button(button_frame, text="Run Detection", command=self.run_detection).pack(side="left", padx=10)

    def update_lane_entries(self):
        for x_start_entry, x_end_entry, ppm_scale_entry in self.lane_entries:
            x_start_entry.destroy()
            x_end_entry.destroy()
            ppm_scale_entry.destroy()

        self.lane_entries = []
        num_lanes = int(self.num_lanes_spin.get())
        lane_frame = self.num_lanes_spin.master
        for i in range(num_lanes):
            tk.Label(lane_frame, text=f"Lane {i+1} X-Range (start, end):").grid(row=i+1, column=0, padx=5, pady=2, sticky="e")
            x_start_entry = tk.Entry(lane_frame, width=10)
            x_start_entry.grid(row=i+1, column=1, padx=2, pady=2, sticky="w")
            x_start_entry.insert(0, str(i * 1020 // num_lanes))
            x_end_entry = tk.Entry(lane_frame, width=10)
            x_end_entry.grid(row=i+1, column=2, padx=2, pady=2, sticky="w")
            x_end_entry.insert(0, str((i + 1) * 1020 // num_lanes))
            tk.Label(lane_frame, text=f"PPM Scale:").grid(row=i+1, column=3, padx=5, pady=2, sticky="e")
            ppm_scale_entry = tk.Entry(lane_frame, width=10)
            ppm_scale_entry.grid(row=i+1, column=4, padx=2, pady=2, sticky="w")
            ppm_scale_entry.insert(0, "1.0")
            self.lane_entries.append((x_start_entry, x_end_entry, ppm_scale_entry))

    def apply_preset(self, event=None):
        name = self.preset_var.get()
        if name in self.presets:
            preset = self.presets[name]
            self.camera_height_entry.delete(0, tk.END)
            self.camera_height_entry.insert(0, str(preset["height"]))
            self.camera_angle_entry.delete(0, tk.END)
            self.camera_angle_entry.insert(0, str(preset["angle"]))
            self.fov_entry.delete(0, tk.END)
            self.fov_entry.insert(0, str(preset["fov"]))
            self.tilt_entry.delete(0, tk.END)
            self.tilt_entry.insert(0, str(preset["tilt"]))
            self.yaw_entry.delete(0, tk.END)
            self.yaw_entry.insert(0, str(preset["yaw"]))
            self.speed_line_y_spin.delete(0, tk.END)
            self.speed_line_y_spin.insert(0, str(preset["speed_line"]))
            self.num_lanes_spin.delete(0, tk.END)
            self.num_lanes_spin.insert(0, str(len(preset["lanes"])))
            self.update_lane_entries()
            for i, lane in enumerate(preset["lanes"]):
                self.lane_entries[i][0].delete(0, tk.END)
                self.lane_entries[i][0].insert(0, str(lane["x_range"][0]))
                self.lane_entries[i][1].delete(0, tk.END)
                self.lane_entries[i][1].insert(0, str(lane["x_range"][1]))
                self.lane_entries[i][2].delete(0, tk.END)
                self.lane_entries[i][2].insert(0, str(lane["ppm_scale"]))
            self.update_zones_from_camera_params()

    def update_zones_from_camera_params(self):
        try:
            h = float(self.camera_height_entry.get() or 0)
            a = float(self.camera_angle_entry.get() or 0)
            fov = float(self.fov_entry.get() or 60)  
            tilt = float(self.tilt_entry.get() or 0)
            yaw = float(self.yaw_entry.get() or 0)
            sensor_size = float(self.sensor_size_entry.get() or 4.8) / 1000  
            focal_length = float(self.focal_length_entry.get() or 3.5) / 1000  
            fh = int(self.frame_height_entry.get() or 500)  
            num_zones = int(self.num_zones_spin.get() or 10)  

            if h <= 0 or a <= 0 or fov <= 0 or fh <= 0 or num_zones <= 0:
                raise ValueError("All values must be positive")

            self.config["zones"] = calculate_zones(h, a, fh, num_zones, fov, tilt, yaw, sensor_size, focal_length)
            self.config["camera_height"] = h
            self.config["camera_angle"] = a
            self.config["fov"] = fov
            self.config["tilt"] = tilt
            self.config["yaw"] = yaw
            self.config["sensor_size"] = sensor_size
            self.config["focal_length"] = focal_length
            self.config["num_zones"] = num_zones
            self.config["frame_height"] = fh
            messagebox.showinfo("Success", "Zones updated")
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid input: {str(e)}. Please enter valid positive numbers.")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

    def browse_video(self):
        path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov")])
        if path:
            self.video_path_entry.delete(0, tk.END)
            self.video_path_entry.insert(0, path)

    def load_config_to_ui(self):
        self.video_path_entry.delete(0, tk.END)
        self.video_path_entry.insert(0, self.config["video_path"])
        self.camera_height_entry.delete(0, tk.END)
        self.camera_height_entry.insert(0, str(self.config["camera_height"]))
        self.camera_angle_entry.delete(0, tk.END)
        self.camera_angle_entry.insert(0, str(self.config["camera_angle"]))
        self.fov_entry.delete(0, tk.END)
        self.fov_entry.insert(0, str(self.config["fov"]))
        self.tilt_entry.delete(0, tk.END)
        self.tilt_entry.insert(0, str(self.config["tilt"]))
        self.yaw_entry.delete(0, tk.END)
        self.yaw_entry.insert(0, str(self.config["yaw"]))
        self.sensor_size_entry.delete(0, tk.END)
        self.sensor_size_entry.insert(0, str(self.config["sensor_size"] * 1000))
        self.focal_length_entry.delete(0, tk.END)
        self.focal_length_entry.insert(0, str(self.config["focal_length"] * 1000))
        self.frame_height_entry.delete(0, tk.END)
        self.frame_height_entry.insert(0, str(self.config["frame_height"]))
        self.speed_line_y_spin.delete(0, tk.END)
        self.speed_line_y_spin.insert(0, str(self.config["speed_line_y"]))
        self.num_zones_spin.delete(0, tk.END)
        self.num_zones_spin.insert(0, str(self.config["num_zones"]))
        self.num_lanes_spin.delete(0, tk.END)
        self.num_lanes_spin.insert(0, str(len(self.config["lanes"])))
        self.update_lane_entries()
        for i, lane in enumerate(self.config["lanes"]):
            self.lane_entries[i][0].delete(0, tk.END)
            self.lane_entries[i][0].insert(0, str(lane["x_range"][0]))
            self.lane_entries[i][1].delete(0, tk.END)
            self.lane_entries[i][1].insert(0, str(lane["x_range"][1]))
            self.lane_entries[i][2].delete(0, tk.END)
            self.lane_entries[i][2].insert(0, str(lane["ppm_scale"]))
        self.distortion_var.set(self.config["distortion_correction"])
        self.fx_entry.delete(0, tk.END)
        self.fx_entry.insert(0, str(self.config["distortion_coeffs"]["fx"]))
        self.fy_entry.delete(0, tk.END)
        self.fy_entry.insert(0, str(self.config["distortion_coeffs"]["fy"]))
        self.cx_entry.delete(0, tk.END)
        self.cx_entry.insert(0, str(self.config["distortion_coeffs"]["cx"]))
        self.cy_entry.delete(0, tk.END)
        self.cy_entry.insert(0, str(self.config["distortion_coeffs"]["cy"]))
        self.k_entry.delete(0, tk.END)
        self.k_entry.insert(0, ",".join(map(str, self.config["distortion_coeffs"]["k"])))
        self.show_zones_var.set(self.config["show_zones"])

    def get_config_from_ui(self):
        lanes = []
        for x_start_entry, x_end_entry, ppm_scale_entry in self.lane_entries:
            lanes.append({
                "x_range": (int(x_start_entry.get()), int(x_end_entry.get())),
                "ppm_scale": float(ppm_scale_entry.get())
            })
        k_coeffs = [float(k) for k in self.k_entry.get().split(",")]
        return {
            "video_path": self.video_path_entry.get(),
            "speed_line_y": int(self.speed_line_y_spin.get()),
            "camera_height": float(self.camera_height_entry.get()),
            "camera_angle": float(self.camera_angle_entry.get()),
            "fov": float(self.fov_entry.get()),
            "tilt": float(self.tilt_entry.get()),
            "yaw": float(self.yaw_entry.get()),
            "sensor_size": float(self.sensor_size_entry.get()) / 1000,
            "focal_length": float(self.focal_length_entry.get()) / 1000,
            "frame_height": int(self.frame_height_entry.get()),
            "num_zones": int(self.num_zones_spin.get()),
            "lanes": lanes,
            "distortion_correction": self.distortion_var.get(),
            "distortion_coeffs": {
                "fx": float(self.fx_entry.get()),
                "fy": float(self.fy_entry.get()),
                "cx": float(self.cx_entry.get()),
                "cy": float(self.cy_entry.get()),
                "k": k_coeffs
            },
            "show_zones": self.show_zones_var.get(),
            "zones": self.config.get("zones", [])
        }

    def save_config(self):
        self.config = self.get_config_from_ui()
        path = filedialog.asksaveasfilename(defaultextension=".json")
        if path:
            with open(path, "w") as f:
                json.dump(self.config, f, indent=4)
            messagebox.showinfo("Saved", "Configuration saved")

    def load_config(self):
        path = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if path:
            with open(path, "r") as f:
                self.config = json.load(f)
            self.load_config_to_ui()
            messagebox.showinfo("Loaded", "Configuration loaded")

    def run_detection(self):
        self.config = self.get_config_from_ui()
        import threading
        threading.Thread(
            target=process_video_with_config,
            args=(self.config["video_path"], self.config["speed_line_y"], self.config["zones"], self.config["lanes"],
                  self.config["show_zones"], self.config["distortion_correction"], self.config["distortion_coeffs"]),
            daemon=True
        ).start()

if __name__ == "__main__":
    root = tk.Tk()
    app = SpeedDetectionConfigurator(root)
    root.mainloop()
