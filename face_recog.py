import cv2
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
from PIL import Image, ImageTk
import os
import pickle
import threading
import time
from datetime import datetime

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🎭 Face Recognition Studio")
        self.root.geometry("1200x800")
        self.root.configure(bg='#2c3e50')
        
        # Initialize variables
        self.camera = None
        self.is_running = False
        self.current_frame = None
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.known_faces = {}
        self.face_id_counter = 0
        
        # Create directories
        self.create_directories()
        
        # Load existing model if available
        self.load_model()
        
        # Setup GUI
        self.setup_gui()
        
        # Apply modern styling
        self.apply_styling()
        
    def create_directories(self):
        """Create necessary directories for storing data"""
        os.makedirs('faces_data', exist_ok=True)
        os.makedirs('captured_faces', exist_ok=True)
        os.makedirs('models', exist_ok=True)
        
    def setup_gui(self):
        """Setup the main GUI layout"""
        # Main title
        title_frame = tk.Frame(self.root, bg='#2c3e50')
        title_frame.pack(pady=20)
        
        title_label = tk.Label(title_frame, text="🎭 Face Recognition Studio", 
                              font=('Arial', 24, 'bold'), fg='#ecf0f1', bg='#2c3e50')
        title_label.pack()
        
        subtitle_label = tk.Label(title_frame, text="Real-time face detection and recognition powered by OpenCV", 
                                 font=('Arial', 12), fg='#bdc3c7', bg='#2c3e50')
        subtitle_label.pack()
        
        # Main container
        main_container = tk.Frame(self.root, bg='#2c3e50')
        main_container.pack(expand=True, fill='both', padx=20, pady=10)
        
        # Left panel for camera feed
        self.setup_camera_panel(main_container)
        
        # Right panel for controls and info
        self.setup_control_panel(main_container)
        
        # Status bar
        self.setup_status_bar()
        
    def setup_camera_panel(self, parent):
        """Setup camera display panel"""
        camera_frame = tk.Frame(parent, bg='#34495e', relief='raised', bd=2)
        camera_frame.pack(side='left', expand=True, fill='both', padx=(0, 10))
        
        # Camera label
        camera_title = tk.Label(camera_frame, text="📹 Live Camera Feed", 
                               font=('Arial', 16, 'bold'), fg='#ecf0f1', bg='#34495e')
        camera_title.pack(pady=10)
        
        # Video display
        self.video_label = tk.Label(camera_frame, bg='#2c3e50', text="Camera Off\n\nClick 'Start Camera' to begin", 
                                   font=('Arial', 14), fg='#bdc3c7')
        self.video_label.pack(expand=True, fill='both', padx=20, pady=20)
        
    def setup_control_panel(self, parent):
        """Setup control panel with buttons and information"""
        control_frame = tk.Frame(parent, bg='#34495e', relief='raised', bd=2)
        control_frame.pack(side='right', fill='y', padx=(10, 0))
        control_frame.configure(width=350)
        
        # Controls title
        controls_title = tk.Label(control_frame, text="🎮 Controls", 
                                 font=('Arial', 16, 'bold'), fg='#ecf0f1', bg='#34495e')
        controls_title.pack(pady=(20, 10))
        
        # Camera controls
        self.setup_camera_controls(control_frame)
        
        # Face recognition controls
        self.setup_recognition_controls(control_frame)
        
        # Information panel
        self.setup_info_panel(control_frame)
        
    def setup_camera_controls(self, parent):
        """Setup camera control buttons"""
        camera_controls = tk.Frame(parent, bg='#34495e')
        camera_controls.pack(pady=10)
        
        self.start_btn = tk.Button(camera_controls, text="📹 Start Camera", 
                                  command=self.start_camera, width=15, height=2)
        self.start_btn.pack(pady=5)
        
        self.stop_btn = tk.Button(camera_controls, text="⏹️ Stop Camera", 
                                 command=self.stop_camera, width=15, height=2)
        self.stop_btn.pack(pady=5)
        
        self.capture_btn = tk.Button(camera_controls, text="📸 Capture Face", 
                                    command=self.capture_face, width=15, height=2)
        self.capture_btn.pack(pady=5)
        
    def setup_recognition_controls(self, parent):
        """Setup face recognition control buttons"""
        recognition_frame = tk.Frame(parent, bg='#34495e')
        recognition_frame.pack(pady=20)
        
        recognition_title = tk.Label(recognition_frame, text="🧠 Recognition", 
                                    font=('Arial', 14, 'bold'), fg='#ecf0f1', bg='#34495e')
        recognition_title.pack(pady=(0, 10))
        
        self.train_btn = tk.Button(recognition_frame, text="🎓 Train Model", 
                                  command=self.train_model, width=15, height=2)
        self.train_btn.pack(pady=5)
        
        self.load_faces_btn = tk.Button(recognition_frame, text="📁 Load Faces", 
                                       command=self.load_faces, width=15, height=2)
        self.load_faces_btn.pack(pady=5)
        
        self.save_model_btn = tk.Button(recognition_frame, text="💾 Save Model", 
                                       command=self.save_model, width=15, height=2)
        self.save_model_btn.pack(pady=5)
        
    def setup_info_panel(self, parent):
        """Setup information display panel"""
        info_frame = tk.Frame(parent, bg='#2c3e50', relief='sunken', bd=2)
        info_frame.pack(pady=20, padx=10, fill='x')
        
        info_title = tk.Label(info_frame, text="📊 Detection Info", 
                             font=('Arial', 12, 'bold'), fg='#ecf0f1', bg='#2c3e50')
        info_title.pack(pady=(10, 5))
        
        # Face count
        self.face_count_label = tk.Label(info_frame, text="Faces Detected: 0", 
                                        font=('Arial', 11), fg='#3498db', bg='#2c3e50')
        self.face_count_label.pack(pady=2)
        
        # Recognition results
        self.recognition_text = tk.Text(info_frame, height=8, width=35, 
                                       bg='#34495e', fg='#ecf0f1', font=('Arial', 9))
        self.recognition_text.pack(pady=10)
        
        # Known faces count
        self.known_faces_label = tk.Label(info_frame, text="Known Faces: 0", 
                                         font=('Arial', 11), fg='#2ecc71', bg='#2c3e50')
        self.known_faces_label.pack(pady=2)
        
    def setup_status_bar(self):
        """Setup status bar at the bottom"""
        self.status_var = tk.StringVar()
        self.status_var.set("Ready - Click 'Start Camera' to begin face detection")
        
        status_bar = tk.Label(self.root, textvariable=self.status_var, 
                             relief='sunken', anchor='w', bg='#34495e', fg='#ecf0f1')
        status_bar.pack(side='bottom', fill='x')
        
    def apply_styling(self):
        """Apply modern styling to the application"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure button styles
        button_style = {
            'font': ('Arial', 10, 'bold'),
            'bg': '#3498db',
            'fg': 'white',
            'activebackground': '#2980b9',
            'activeforeground': 'white',
            'relief': 'flat',
            'borderwidth': 0
        }
        
        # Apply to all buttons
        buttons = [self.start_btn, self.stop_btn, self.capture_btn, 
                  self.train_btn, self.load_faces_btn, self.save_model_btn]
        
        for button in buttons:
            button.configure(**button_style)
            
    def start_camera(self):
        """Start the camera feed"""
        try:
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                raise Exception("Could not open camera")
                
            self.is_running = True
            self.status_var.set("Camera started - Face detection active")
            
            # Start video feed in separate thread
            self.video_thread = threading.Thread(target=self.video_loop)
            self.video_thread.daemon = True
            self.video_thread.start()
            
        except Exception as e:
            messagebox.showerror("Camera Error", f"Failed to start camera: {str(e)}")
            self.status_var.set("Camera error - Please check your camera connection")
            
    def stop_camera(self):
        """Stop the camera feed"""
        self.is_running = False
        if self.camera:
            self.camera.release()
            self.camera = None
            
        # Reset video display
        self.video_label.configure(image='', text="Camera Off\n\nClick 'Start Camera' to begin")
        self.status_var.set("Camera stopped")
        self.face_count_label.configure(text="Faces Detected: 0")
        
    def video_loop(self):
        """Main video processing loop"""
        while self.is_running and self.camera:
            ret, frame = self.camera.read()
            if not ret:
                break
                
            self.current_frame = frame.copy()
            
            # Process frame for face detection
            processed_frame = self.process_frame(frame)
            
            # Convert to display format
            display_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            display_frame = cv2.resize(display_frame, (640, 480))
            
            # Convert to PhotoImage
            image = Image.fromarray(display_frame)
            photo = ImageTk.PhotoImage(image)
            
            # Update display
            self.video_label.configure(image=photo, text='')
            self.video_label.image = photo
            
            time.sleep(0.03)  # ~30 FPS
            
    def process_frame(self, frame):
        """Process frame for face detection and recognition"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        # Update face count
        self.face_count_label.configure(text=f"Faces Detected: {len(faces)}")
        
        # Clear previous recognition results
        self.recognition_text.delete(1.0, tk.END)
        
        for i, (x, y, w, h) in enumerate(faces):
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Try to recognize face
            face_roi = gray[y:y + h, x:x + w]
            
            try:
                if len(self.known_faces) > 0:
                    label, confidence = self.recognizer.predict(face_roi)
                    
                    if confidence < 50:  # Good match
                        name = self.known_faces.get(label, f"Person {label}")
                        color = (0, 255, 0)  # Green
                        status = "✅ Recognized"
                    else:
                        name = "Unknown"
                        color = (0, 0, 255)  # Red
                        status = "❓ Unknown"
                        
                    # Draw label
                    cv2.putText(frame, f"{name} ({confidence:.1f})", 
                               (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    
                    # Update recognition display
                    self.recognition_text.insert(tk.END, 
                        f"Face {i+1}: {name}\n"
                        f"Confidence: {confidence:.1f}%\n"
                        f"Status: {status}\n"
                        f"Time: {datetime.now().strftime('%H:%M:%S')}\n\n")
                        
                else:
                    cv2.putText(frame, f"Face {i+1}", 
                               (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                    
            except Exception as e:
                cv2.putText(frame, f"Face {i+1}", 
                           (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                
        return frame
        
    def capture_face(self):
        """Capture current face for training"""
        if not self.is_running or self.current_frame is None:
            messagebox.showwarning("Warning", "Please start the camera first")
            return
            
        # Ask for person's name
        name = tk.simpledialog.askstring("Person Name", "Enter person's name:")
        if not name:
            return
            
        gray = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            messagebox.showwarning("Warning", "No faces detected in current frame")
            return
            
        # Save the largest face
        largest_face = max(faces, key=lambda face: face[2] * face[3])
        x, y, w, h = largest_face
        face_roi = gray[y:y + h, x:x + w]
        
        # Create directory for this person
        person_dir = f"faces_data/{name}"
        os.makedirs(person_dir, exist_ok=True)
        
        # Save face image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{person_dir}/{timestamp}.jpg"
        cv2.imwrite(filename, face_roi)
        
        # Also save full capture
        capture_filename = f"captured_faces/{name}_{timestamp}.jpg"
        cv2.imwrite(capture_filename, self.current_frame)
        
        messagebox.showinfo("Success", f"Face captured and saved for {name}")
        self.status_var.set(f"Face captured for {name}")
        
    def load_faces(self):
        """Load faces from directory for training"""
        if not os.path.exists('faces_data'):
            messagebox.showwarning("Warning", "No faces data directory found")
            return
            
        faces = []
        labels = []
        self.known_faces = {}
        label_id = 0
        
        for person_name in os.listdir('faces_data'):
            person_path = f"faces_data/{person_name}"
            if not os.path.isdir(person_path):
                continue
                
            self.known_faces[label_id] = person_name
            
            for image_name in os.listdir(person_path):
                image_path = f"{person_path}/{image_name}"
                face_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                faces.append(face_img)
                labels.append(label_id)
                
            label_id += 1
            
        if len(faces) == 0:
            messagebox.showwarning("Warning", "No face images found")
            return
            
        messagebox.showinfo("Success", f"Loaded {len(faces)} face images for {len(self.known_faces)} people")
        self.known_faces_label.configure(text=f"Known Faces: {len(self.known_faces)}")
        self.status_var.set(f"Loaded {len(faces)} face images")
        
        return faces, labels
        
    def train_model(self):
        """Train the face recognition model"""
        try:
            data = self.load_faces()
            if not data:
                return
                
            faces, labels = data
            
            self.status_var.set("Training model... Please wait")
            self.root.update()
            
            # Train the recognizer
            self.recognizer.train(faces, np.array(labels))
            
            messagebox.showinfo("Success", "Model trained successfully!")
            self.status_var.set("Model training completed")
            
        except Exception as e:
            messagebox.showerror("Training Error", f"Failed to train model: {str(e)}")
            self.status_var.set("Model training failed")
            
    def save_model(self):
        """Save the trained model"""
        try:
            self.recognizer.save('models/face_recognizer.yml')
            
            # Save known faces mapping
            with open('models/known_faces.pkl', 'wb') as f:
                pickle.dump(self.known_faces, f)
                
            messagebox.showinfo("Success", "Model saved successfully!")
            self.status_var.set("Model saved")
            
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save model: {str(e)}")
            
    def load_model(self):
        """Load existing trained model"""
        try:
            if os.path.exists('models/face_recognizer.yml'):
                self.recognizer.read('models/face_recognizer.yml')
                
            if os.path.exists('models/known_faces.pkl'):
                with open('models/known_faces.pkl', 'rb') as f:
                    self.known_faces = pickle.load(f)
                    
                self.known_faces_label.configure(text=f"Known Faces: {len(self.known_faces)}")
                self.status_var.set("Existing model loaded successfully")
                
        except Exception as e:
            print(f"Could not load existing model: {e}")
            
    def on_closing(self):
        """Handle application closing"""
        self.stop_camera()
        self.root.destroy()

def main():
    # Create main application window
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    
    # Handle window closing
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    # Start the application
    root.mainloop()

if __name__ == "__main__":
    # Required imports check
    try:
        import tkinter.simpledialog
        main()
    except ImportError as e:
        print(f"Missing required package: {e}")
        print("Please install required packages:")
        print("pip install opencv-python opencv-contrib-python pillow numpy")