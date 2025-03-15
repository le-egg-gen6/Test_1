import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import threading
import time
from pathlib import Path
from ultralytics import YOLO
import numpy as np

class YOLOv8App:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLOv8 Object Detection")
        self.root.geometry("1000x700")
        self.root.configure(padx=10, pady=10)
        
        # Model variables
        self.model_path = tk.StringVar(value="")
        self.data_yaml_path = tk.StringVar(value="")
        self.model_type = tk.StringVar(value="yolov8n.pt")
        self.epochs = tk.IntVar(value=100)
        self.batch_size = tk.IntVar(value=16)
        self.img_size = tk.IntVar(value=640)
        self.conf_threshold = tk.DoubleVar(value=0.25)
        self.device = tk.StringVar(value="0")
        
        # Detection variables
        self.image_path = tk.StringVar(value="")
        self.video_path = tk.StringVar(value="")
        self.output_dir = tk.StringVar(value="output")
        self.webcam_active = False
        self.webcam_thread = None
        
        # Create tabs
        self.tab_control = ttk.Notebook(root)
        
        # Tab 1: Training
        self.train_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.train_tab, text="Training")
        self._create_training_tab()
        
        # Tab 2: Image Detection
        self.image_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.image_tab, text="Image Detection")
        self._create_image_tab()
        
        # Tab 3: Video Detection
        self.video_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.video_tab, text="Video Detection")
        self._create_video_tab()
        
        # Tab 4: Webcam Detection
        self.webcam_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.webcam_tab, text="Webcam Detection")
        self._create_webcam_tab()
        
        self.tab_control.pack(expand=1, fill="both")
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        self.status_bar = ttk.Label(root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Initialize model
        self.model = None
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir.get(), exist_ok=True)
        
    def _create_training_tab(self):
        # Training frame
        training_frame = ttk.LabelFrame(self.train_tab, text="Model Training")
        training_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Model type selection
        ttk.Label(training_frame, text="Model Type:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        model_types = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"]
        model_combo = ttk.Combobox(training_frame, textvariable=self.model_type, values=model_types)
        model_combo.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Data YAML path
        ttk.Label(training_frame, text="Data YAML:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        data_entry = ttk.Entry(training_frame, textvariable=self.data_yaml_path, width=50)
        data_entry.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        ttk.Button(training_frame, text="Browse", command=self._browse_data_yaml).grid(row=1, column=2, padx=5, pady=5)
        
        # Training parameters
        ttk.Label(training_frame, text="Epochs:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Spinbox(training_frame, from_=1, to=1000, textvariable=self.epochs, width=10).grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(training_frame, text="Batch Size:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Spinbox(training_frame, from_=1, to=128, textvariable=self.batch_size, width=10).grid(row=3, column=1, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(training_frame, text="Image Size:").grid(row=4, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Spinbox(training_frame, from_=32, to=1280, textvariable=self.img_size, width=10).grid(row=4, column=1, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(training_frame, text="Device:").grid(row=5, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(training_frame, textvariable=self.device, width=10).grid(row=5, column=1, sticky=tk.W, padx=5, pady=5)
        ttk.Label(training_frame, text="(0 for GPU, cpu for CPU, 0,1,2 for multiple GPUs)").grid(row=5, column=2, sticky=tk.W, padx=5, pady=5)
        
        # Training button
        ttk.Button(training_frame, text="Start Training", command=self._start_training).grid(row=6, column=0, columnspan=3, pady=20)
        
        # Training log
        ttk.Label(training_frame, text="Training Log:").grid(row=7, column=0, sticky=tk.W, padx=5, pady=5)
        self.log_text = tk.Text(training_frame, height=10, width=80)
        self.log_text.grid(row=8, column=0, columnspan=3, padx=5, pady=5)
        log_scroll = ttk.Scrollbar(training_frame, command=self.log_text.yview)
        log_scroll.grid(row=8, column=3, sticky='nsew')
        self.log_text.configure(yscrollcommand=log_scroll.set)
        
    def _create_image_tab(self):
        # Image detection frame
        image_frame = ttk.LabelFrame(self.image_tab, text="Image Detection")
        image_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Model selection
        ttk.Label(image_frame, text="Model Path:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(image_frame, textvariable=self.model_path, width=50).grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        ttk.Button(image_frame, text="Browse", command=self._browse_model).grid(row=0, column=2, padx=5, pady=5)
        
        # Image selection
        ttk.Label(image_frame, text="Image Path:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(image_frame, textvariable=self.image_path, width=50).grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        ttk.Button(image_frame, text="Browse", command=self._browse_image).grid(row=1, column=2, padx=5, pady=5)
        
        # Confidence threshold
        ttk.Label(image_frame, text="Confidence:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Scale(image_frame, from_=0.0, to=1.0, orient=tk.HORIZONTAL, variable=self.conf_threshold, length=200).grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)
        ttk.Label(image_frame, textvariable=self.conf_threshold).grid(row=2, column=2, sticky=tk.W, padx=5, pady=5)
        
        # Output directory
        ttk.Label(image_frame, text="Output Dir:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(image_frame, textvariable=self.output_dir, width=50).grid(row=3, column=1, sticky=tk.W, padx=5, pady=5)
        ttk.Button(image_frame, text="Browse", command=self._browse_output_dir).grid(row=3, column=2, padx=5, pady=5)
        
        # Detect button
        ttk.Button(image_frame, text="Detect", command=self._detect_image).grid(row=4, column=0, columnspan=3, pady=10)
        
        # Image display
        self.image_display = ttk.Label(image_frame)
        self.image_display.grid(row=5, column=0, columnspan=3, padx=5, pady=5)
        
    def _create_video_tab(self):
        # Video detection frame
        video_frame = ttk.LabelFrame(self.video_tab, text="Video Detection")
        video_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Model selection
        ttk.Label(video_frame, text="Model Path:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(video_frame, textvariable=self.model_path, width=50).grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        ttk.Button(video_frame, text="Browse", command=self._browse_model).grid(row=0, column=2, padx=5, pady=5)
        
        # Video selection
        ttk.Label(video_frame, text="Video Path:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(video_frame, textvariable=self.video_path, width=50).grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        ttk.Button(video_frame, text="Browse", command=self._browse_video).grid(row=1, column=2, padx=5, pady=5)
        
        # Confidence threshold
        ttk.Label(video_frame, text="Confidence:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Scale(video_frame, from_=0.0, to=1.0, orient=tk.HORIZONTAL, variable=self.conf_threshold, length=200).grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)
        ttk.Label(video_frame, textvariable=self.conf_threshold).grid(row=2, column=2, sticky=tk.W, padx=5, pady=5)
        
        # Output directory
        ttk.Label(video_frame, text="Output Dir:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(video_frame, textvariable=self.output_dir, width=50).grid(row=3, column=1, sticky=tk.W, padx=5, pady=5)
        ttk.Button(video_frame, text="Browse", command=self._browse_output_dir).grid(row=3, column=2, padx=5, pady=5)
        
        # Detect button
        ttk.Button(video_frame, text="Process Video", command=self._process_video).grid(row=4, column=0, columnspan=3, pady=10)
        
        # Progress bar
        self.progress_var = tk.DoubleVar(value=0.0)
        ttk.Label(video_frame, text="Progress:").grid(row=5, column=0, sticky=tk.W, padx=5, pady=5)
        self.progress_bar = ttk.Progressbar(video_frame, variable=self.progress_var, length=400, mode='determinate')
        self.progress_bar.grid(row=5, column=1, columnspan=2, padx=5, pady=5, sticky=tk.W)
        
        # Video info
        self.video_info_var = tk.StringVar(value="")
        ttk.Label(video_frame, textvariable=self.video_info_var).grid(row=6, column=0, columnspan=3, padx=5, pady=5)
        
    def _create_webcam_tab(self):
        # Webcam detection frame
        webcam_frame = ttk.LabelFrame(self.webcam_tab, text="Webcam Detection")
        webcam_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Model selection
        ttk.Label(webcam_frame, text="Model Path:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(webcam_frame, textvariable=self.model_path, width=50).grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        ttk.Button(webcam_frame, text="Browse", command=self._browse_model).grid(row=0, column=2, padx=5, pady=5)
        
        # Confidence threshold
        ttk.Label(webcam_frame, text="Confidence:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Scale(webcam_frame, from_=0.0, to=1.0, orient=tk.HORIZONTAL, variable=self.conf_threshold, length=200).grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        ttk.Label(webcam_frame, textvariable=self.conf_threshold).grid(row=1, column=2, sticky=tk.W, padx=5, pady=5)
        
        # Camera selection
        self.camera_id = tk.IntVar(value=0)
        ttk.Label(webcam_frame, text="Camera ID:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Spinbox(webcam_frame, from_=0, to=10, textvariable=self.camera_id, width=5).grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)
        ttk.Label(webcam_frame, text="(0 for default camera)").grid(row=2, column=2, sticky=tk.W, padx=5, pady=5)
        
        # Control buttons
        button_frame = ttk.Frame(webcam_frame)
        button_frame.grid(row=3, column=0, columnspan=3, pady=10)
        
        self.start_webcam_btn = ttk.Button(button_frame, text="Start Webcam", command=self._start_webcam)
        self.start_webcam_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_webcam_btn = ttk.Button(button_frame, text="Stop Webcam", command=self._stop_webcam, state=tk.DISABLED)
        self.stop_webcam_btn.pack(side=tk.LEFT, padx=5)
        
        self.capture_frame_btn = ttk.Button(button_frame, text="Capture Frame", command=self._capture_frame, state=tk.DISABLED)
        self.capture_frame_btn.pack(side=tk.LEFT, padx=5)
        
        # Webcam display
        self.webcam_display_frame = ttk.Frame(webcam_frame, borderwidth=2, relief=tk.SUNKEN)
        self.webcam_display_frame.grid(row=4, column=0, columnspan=3, padx=5, pady=5)
        
        self.webcam_canvas = tk.Canvas(self.webcam_display_frame, width=640, height=480)
        self.webcam_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Status label
        self.webcam_status_var = tk.StringVar(value="Webcam inactive")
        ttk.Label(webcam_frame, textvariable=self.webcam_status_var).grid(row=5, column=0, columnspan=3, padx=5, pady=5)
        
        # FPS counter
        self.fps_var = tk.StringVar(value="FPS: 0")
        ttk.Label(webcam_frame, textvariable=self.fps_var).grid(row=6, column=0, columnspan=3, padx=5, pady=5)

if __name__ == "__main__":
    root = tk.Tk()
    app = YOLOv8App(root=root)
    app.mainloop()