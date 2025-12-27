import tkinter as tk
from tkinter import filedialog, Label, Button, Frame
from PIL import Image, ImageTk
from ultralytics import YOLO
import cv2
import numpy as np
import os
import threading


class HoleDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Road Hole Detection System")
        self.root.geometry("1400x900")
        self.root.configure(bg="#0f172a")

        # Colors (Modern dark theme)
        self.bg_primary = "#0f172a"
        self.bg_secondary = "#1e293b"
        self.bg_card = "#334155"
        self.accent = "#3b82f6"
        self.accent_hover = "#2563eb"
        self.text_primary = "#f1f5f9"
        self.text_secondary = "#94a3b8"
        self.success = "#10b981"
        self.warning = "#f59e0b"

        # Video playback variables
        self.video_playing = False
        self.current_video_path = None
        self.video_cap = None
        self.stop_video_flag = False

        # Load Model
        try:
            self.model = YOLO(r"C:\Users\asus\ML-2024-25\runs\detect\pothole_detector3\weights\best.pt")
            self.model_status = "Ready"
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
            self.model_status = "Model Not Found"

        self.setup_ui()

    def setup_ui(self):
        # Header Section
        header = Frame(self.root, bg=self.bg_secondary, height=100)
        header.pack(fill="x", padx=0, pady=0)
        header.pack_propagate(False)

        # Title
        title_label = Label(
            header,
            text="Road Hole Detection System",
            font=("Helvetica", 28, "bold"),
            bg=self.bg_secondary,
            fg=self.text_primary
        )
        title_label.pack(pady=10)

        # Subtitle
        subtitle_label = Label(
            header,
            text="Pothole Detection",
            font=("Helvetica", 12),
            bg=self.bg_secondary,
            fg=self.text_secondary
        )
        subtitle_label.pack()

        # Status Bar
        status_frame = Frame(self.root, bg=self.bg_secondary, height=40)
        status_frame.pack(fill="x", padx=0, pady=(0, 2))
        status_frame.pack_propagate(False)

        status_color = self.success if self.model else self.warning
        self.status_label = Label(
            status_frame,
            text=f"Model Status: {self.model_status}",
            font=("Helvetica", 10),
            bg=self.bg_secondary,
            fg=status_color
        )
        self.status_label.pack(side="left", padx=20)

        self.detection_count = Label(
            status_frame,
            text="Detected Holes: --",
            font=("Helvetica", 10),
            bg=self.bg_secondary,
            fg=self.text_secondary
        )
        self.detection_count.pack(side="right", padx=20)

        # Main Content Area
        content_frame = Frame(self.root, bg=self.bg_primary)
        content_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Upload Section (Center)
        upload_section = Frame(content_frame, bg=self.bg_primary)
        upload_section.pack(pady=30)

        button_container = Frame(upload_section, bg=self.bg_primary)
        button_container.pack()

        self.btn_upload = Button(
            button_container,
            text="Upload Image or Video",
            command=self.upload_file,
            font=("Helvetica", 14, "bold"),
            bg=self.accent,
            fg="white",
            activebackground=self.accent_hover,
            activeforeground="white",
            relief="flat",
            padx=40,
            pady=15,
            cursor="hand2",
            bd=0
        )
        self.btn_upload.pack(side="left", padx=5)

        self.btn_stop = Button(
            button_container,
            text="Stop Video",
            command=self.stop_video,
            font=("Helvetica", 14, "bold"),
            bg="#ef4444",
            fg="white",
            activebackground="#dc2626",
            activeforeground="white",
            relief="flat",
            padx=40,
            pady=15,
            cursor="hand2",
            bd=0,
            state="disabled"
        )
        self.btn_stop.pack(side="left", padx=5)

        # Bind hover effects
        self.btn_upload.bind("<Enter>", lambda e: self.btn_upload.config(bg=self.accent_hover) if self.btn_upload[
                                                                                                      'state'] == 'normal' else None)
        self.btn_upload.bind("<Leave>", lambda e: self.btn_upload.config(bg=self.accent) if self.btn_upload[
                                                                                                'state'] == 'normal' else None)

        # Image/Video Comparison Container
        self.comparison_frame = Frame(content_frame, bg=self.bg_primary)
        self.comparison_frame.pack(fill="both", expand=True, pady=20)

        # Left Card - Original
        left_card = Frame(self.comparison_frame, bg=self.bg_card, relief="flat", bd=0)
        left_card.pack(side="left", fill="both", expand=True, padx=10)

        left_header = Label(
            left_card,
            text="Original",
            font=("Helvetica", 14, "bold"),
            bg=self.bg_card,
            fg=self.text_primary,
            pady=15
        )
        left_header.pack(fill="x")

        self.lbl_input = Label(
            left_card,
            text="No file uploaded",
            font=("Helvetica", 11),
            bg=self.bg_secondary,
            fg=self.text_secondary,
            relief="flat",
            pady=50
        )
        self.lbl_input.pack(fill="both", expand=True, padx=15, pady=(0, 15))

        # Right Card - Detection Result
        right_card = Frame(self.comparison_frame, bg=self.bg_card, relief="flat", bd=0)
        right_card.pack(side="right", fill="both", expand=True, padx=10)

        right_header = Label(
            right_card,
            text="Detection Results",
            font=("Helvetica", 14, "bold"),
            bg=self.bg_card,
            fg=self.text_primary,
            pady=15
        )
        right_header.pack(fill="x")

        self.lbl_output = Label(
            right_card,
            text="Awaiting detection...",
            font=("Helvetica", 11),
            bg=self.bg_secondary,
            fg=self.text_secondary,
            relief="flat",
            pady=50
        )
        self.lbl_output.pack(fill="both", expand=True, padx=15, pady=(0, 15))

        # Footer
        footer = Frame(self.root, bg=self.bg_secondary, height=50)
        footer.pack(fill="x", side="bottom")
        footer.pack_propagate(False)

        footer_label = Label(
            footer,
            text="Powered by YOLOv8 | Built with Python & Tkinter | Supports Images & Videos",
            font=("Helvetica", 9),
            bg=self.bg_secondary,
            fg=self.text_secondary
        )
        footer_label.pack(expand=True)

    def upload_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[
                ("Media Files", "*.jpg *.jpeg *.png *.bmp *.mp4 *.avi *.mov *.mkv"),
                ("Image Files", "*.jpg *.jpeg *.png *.bmp"),
                ("Video Files", "*.mp4 *.avi *.mov *.mkv")
            ],
            title="Select a road image or video"
        )

        if file_path:
            # Stop any currently playing video
            self.stop_video()

            # Determine file type
            ext = os.path.splitext(file_path)[1].lower()

            if ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                self.process_image(file_path)
            elif ext in ['.mp4', '.avi', '.mov', '.mkv']:
                self.process_video(file_path)

    def process_image(self, image_path):
        # Update button state
        self.btn_upload.config(text="...", state="disabled")
        self.root.update()

        # Display Original Image
        self.display_image(image_path, self.lbl_input)

        # Run Prediction
        self.predict_holes_image(image_path)

        # Reset button
        self.btn_upload.config(text="Upload Another File", state="normal")

    def process_video(self, video_path):
        if self.model is None:
            self.lbl_output.config(
                text="⚠Model not found\nPlease check model path",
                fg=self.warning
            )
            return

        self.current_video_path = video_path
        self.stop_video_flag = False

        # Enable stop button
        self.btn_stop.config(state="normal")
        self.btn_upload.config(state="disabled")

        # Start video processing in a separate thread
        thread = threading.Thread(target=self.play_video_with_detection, daemon=True)
        thread.start()

    def play_video_with_detection(self):
        self.video_playing = True
        cap = cv2.VideoCapture(self.current_video_path)

        if not cap.isOpened():
            self.lbl_output.config(
                text="Cannot open video file",
                fg=self.warning
            )
            self.video_playing = False
            return

        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        frame_delay = int(1000 / fps)  # milliseconds

        frame_count = 0
        total_detections = 0

        while cap.isOpened() and not self.stop_video_flag:
            ret, frame = cap.read()

            if not ret:
                break

            frame_count += 1

            # Process every frame (you can skip frames for better performance: frame_count % 2 == 0)
            results = self.model.predict(source=frame, save=False, verbose=False)

            # Get detection count
            num_detections = len(results[0].boxes)
            total_detections += num_detections

            # Update detection count
            self.detection_count.config(
                text=f"Current Frame: {num_detections} holes | Total: {total_detections}",
                fg=self.warning if num_detections > 0 else self.success
            )

            # Plot results
            res_plotted = results[0].plot()

            # Convert frames to PIL
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)

            res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
            res_pil = Image.fromarray(res_rgb)

            # Update displays
            self.update_label_with_image(frame_pil, self.lbl_input)
            self.update_label_with_image(res_pil, self.lbl_output)

            # Control playback speed
            self.root.update()
            self.root.after(frame_delay)

        cap.release()
        self.video_playing = False

        # Reset UI
        self.btn_upload.config(state="normal")
        self.btn_stop.config(state="disabled")

        if not self.stop_video_flag:
            self.lbl_output.config(
                text=f"✓ Video processing complete\nTotal detections: {total_detections}",
                fg=self.success
            )

    def stop_video(self):
        if self.video_playing:
            self.stop_video_flag = True
            self.btn_stop.config(state="disabled")

    def predict_holes_image(self, image_path):
        if self.model is None:
            self.lbl_output.config(
                text="Model not found\nPlease check model path",
                fg=self.warning
            )
            return

        # Run inference
        results = self.model.predict(source=image_path, save=False)

        # Get detection count
        num_detections = len(results[0].boxes)
        self.detection_count.config(
            text=f"Detected Holes: {num_detections}",
            fg=self.warning if num_detections > 0 else self.success
        )

        # Plot results
        res_plotted = results[0].plot()
        res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(res_rgb)

        # Update output label
        self.update_label_with_image(img_pil, self.lbl_output)

    def display_image(self, path, label_widget):
        img = Image.open(path)
        self.update_label_with_image(img, label_widget)

    def update_label_with_image(self, img_pil, label_widget):
        # Resize to fit container
        img_pil.thumbnail((600, 600), Image.Resampling.LANCZOS)

        img_tk = ImageTk.PhotoImage(img_pil)

        label_widget.config(image=img_tk, text="", bg=self.bg_secondary)
        label_widget.image = img_tk


if __name__ == "__main__":
    root = tk.Tk()
    app = HoleDetectionApp(root)
    root.mainloop()