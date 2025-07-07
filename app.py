import os
from dotenv import load_dotenv
import re
import psutil
import cv2
import time
import uuid
import torch
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, send_file, send_from_directory, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from torchvision import transforms, models
from torch import nn
from werkzeug.security import generate_password_hash, check_password_hash
from flask_mail import Mail, Message
from datetime import datetime
from sqlalchemy.exc import IntegrityError

DEBUG_MODE = os.getenv("DEBUG_MODE", "off").lower() == "on"
def debug_log(msg):
    if DEBUG_MODE:
        print(msg)

start_time = time.time()
app = Flask(__name__, template_folder='prototype')
load_dotenv()
debug_log(f"✅ Memory usage at Flask startup: {round(psutil.virtual_memory().used / 1024**2, 2)} MB")
app.secret_key = os.getenv("SECRET_KEY")  # 修改为自己的密钥
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
db = SQLAlchemy(app)
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.getenv("MAIL_USERNAME")
app.config['MAIL_PASSWORD'] = os.getenv("MAIL_PASSWORD")
mail = Mail(app)

def is_valid_email(email):
    pattern = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
    return re.match(pattern, email)

# === 用户模型 ===
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    otp = db.Column(db.String(6))           # 存储验证码
    is_verified = db.Column(db.Boolean, default=False)  # 邮箱是否验证成功


class History(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    filename = db.Column(db.String(255))
    result = db.Column(db.String(10))
    confidence = db.Column(db.Float)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    face_images = db.Column(db.Text)         # 存逗号分隔 Cloudinary 链接
    crop_faces = db.Column(db.Text)         # 同上
    original_filename = db.Column(db.String(255))  # 原始文件名（如 id0_0001.mp4）

    user = db.relationship('User', backref='history')


# === CNN + LSTM PyTorch 模型 ===
class CNN_LSTM(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        base_cnn = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        base_cnn.fc = nn.Identity()
        self.cnn = base_cnn
        self.lstm = nn.LSTM(input_size=512, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, T, C, H, W = x.size()
        x = x.view(B * T, C, H, W)
        with torch.no_grad():
            cnn_feats = self.cnn(x)
        cnn_feats = cnn_feats.view(B, T, -1)
        lstm_out, _ = self.lstm(cnn_feats)
        out = self.fc(lstm_out[:, -1, :])
        return out.squeeze(dim=1)

# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = CNN_LSTM().to(device)
# model.load_state_dict(torch.load(r"Train Model/cnn_lstm_deepfake_self_train.pth", map_location=device))
# model.eval()

model = None  # global variable for lazy loading

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def get_model():
    global model
    if model is None:
        debug_log("📦 Loading model...")
        m = CNN_LSTM().to(device)
        m.load_state_dict(torch.load("Train Model/best_unbalanced_model.pth", map_location=device))
        m.eval()
        model = m
        debug_log("✅ Model loaded successfully.")
    return model

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/auth', methods=['GET', 'POST'])
def auth():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        action = request.form.get('action')
        email = request.form.get('email')
        confirm_password = request.form.get('confirm_password')
        

        user = User.query.filter_by(username=username).first()

        if action == 'register':
            if user:
                flash('Username already exists')
            elif User.query.filter_by(email=email).first():
                flash('Email already registered')
            elif not is_valid_email(email):
                flash('Invalid email format')
            elif password != confirm_password:
                flash('Passwords do not match')
            elif not re.match(r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d).{8,}$', password):
                flash('Password must be at least 8 characters long and contain uppercase, lowercase letters, and numbers')
            else:
                otp_code = str(np.random.randint(100000, 999999))  # 6-digit OTP

                try:
                    msg = Message("Your Verification Code",
                                  sender=app.config['MAIL_USERNAME'],
                                  recipients=[email])
                    msg.body = f"Your OTP code is: {otp_code}"
                    mail.send(msg)
                except Exception as e:
                    print(f"[MAIL ERROR] {e}")
                    flash('Error sending email. Please try again.')
                    return render_template('auth.html')

                hashed_pw = generate_password_hash(password)
                new_user = User(username=username, email=email, password=hashed_pw,
                                otp=otp_code, is_verified=False)
                db.session.add(new_user)
                try:
                    db.session.add(new_user)
                    db.session.commit()
                    flash('OTP sent to your email. Please verify to activate your account.')
                    return redirect(url_for('verify'))
                except IntegrityError:
                    db.session.rollback()
                    flash('This email is already registered. Please log in instead.')
                    return redirect(url_for('auth'))

        elif action == 'login':
            if user and check_password_hash(user.password, password):
                if not getattr(user, 'is_verified', False):
                    flash('Please verify your email before logging in.')
                else:
                    session['user'] = username
                    return redirect(url_for('home'))
            else:
                flash('Invalid credentials')

    return render_template('auth.html')


@app.route('/verify', methods=['GET', 'POST'])
def verify():
    if request.method == 'POST':
        email = request.form['email']
        otp_input = request.form['otp']

        user = User.query.filter_by(email=email).first()
        if not user:
            flash("Email not found")
        elif user.otp != otp_input:
            flash("Incorrect OTP code")
        else:
            user.is_verified = True
            db.session.commit()
            flash("Verification successful. You can now log in.")
            return redirect(url_for('auth', action='login'))

    return render_template('verify.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('home'))

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if not file.filename.endswith(('.mp4', '.avi', '.mov')):
            return "Unsupported file type."

        unique_folder_name = str(uuid.uuid4())
        video_folder = os.path.join(app.config['UPLOAD_FOLDER'], unique_folder_name)
        os.makedirs(video_folder, exist_ok=True)

        video_path = os.path.join(video_folder, file.filename)
        file.save(video_path)

        # 分析视频
        face_images, crop_faces, original_video_path, preview_path, results = analyze_video(video_path, unique_folder_name)

        # ✅ 仅在用户已登录时才保存记录
        if 'user' in session:
            current_user = User.query.filter_by(username=session['user']).first()
            if current_user:
                db.session.add(History(
                    user_id=current_user.id,
                    filename=preview_path,
                    original_filename=file.filename,
                    face_images=",".join(face_images),
                    crop_faces=",".join(crop_faces),
                    result=results[0]['result'],
                    confidence=results[0]['confidence']
                ))
                db.session.commit()

        return render_template('result.html',
            face_images=face_images,
            crop_faces=crop_faces,
            video_path=original_video_path,
            preview_path=preview_path,
            results=results
        )

    return render_template('index.html')


@app.route('/result')
def result_page():
    return render_template('result.html')

@app.route('/history')
def history():
    if 'user' not in session:
        return redirect(url_for('auth', action='login'))
    
    current_user = User.query.filter_by(username=session['user']).first()
    records = History.query.filter_by(user_id=current_user.id).order_by(History.timestamp.desc()).all()
    return render_template('history.html', records=records)

@app.route('/history/<int:history_id>')
def history_detail(history_id):
    if 'user' not in session:
        return redirect(url_for('auth', action='login'))

    record = History.query.get_or_404(history_id)
    current_user = User.query.filter_by(username=session['user']).first()

    if record.user_id != current_user.id:
        flash("Access denied.")
        return redirect(url_for('history'))

    folder = record.filename.split('/')[2]  # static/uploads/{uuid}/result_preview.mp4

    face_images = [
        url_for('uploaded_file',
                unique_folder_name=folder,
                subfolder='frames',
                filename=img.split('/')[-1])
        for img in record.face_images.split(',')
    ]

    crop_faces = [
        url_for('uploaded_file',
                unique_folder_name=folder,
                subfolder='frames_crop',
                filename=img.split('/')[-1])
        for img in record.crop_faces.split(',')
    ]

    preview_path = url_for('uploaded_video',
                           unique_folder_name=folder,
                           filename=os.path.basename(record.filename))

    return render_template('history_detail.html',
        record=record,
        face_images=face_images,
        crop_faces=crop_faces,
        preview_path=preview_path,
        results=[{
            'confidence': record.confidence,
            'result': record.result
        }]
    )

@app.route('/uploads/<unique_folder_name>/<filename>')
def uploaded_video(unique_folder_name, filename):
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_folder_name, filename)
    return send_file(video_path, mimetype='video/mp4')

@app.route('/uploads/<unique_folder_name>/<subfolder>/<filename>')
def uploaded_file(unique_folder_name, subfolder, filename):
    return send_from_directory(os.path.join(app.config['UPLOAD_FOLDER'], unique_folder_name, subfolder), filename)

def analyze_video(filepath, unique_folder_name, num_frames=20):
    debug_log("📊 Start analyzing video...")
    debug_log(f"💡 RAM at start: {round(psutil.virtual_memory().used / 1024**2, 2)} MB")
    debug_log(f"💡 CPU usage: {psutil.cpu_percent(interval=0.5)}%")
    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        print(f"ERROR: Could not open video file: {filepath}")
        return [], []

    base_folder = os.path.join(app.config['UPLOAD_FOLDER'], unique_folder_name)
    frames_folder = os.path.join(base_folder, 'frames')
    os.makedirs(frames_folder, exist_ok=True)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, num_frames * 4, dtype=int)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    face_images = []
    images_tensor = []
    raw_frames = []
    face_boxes = []

    crop_folder = os.path.join(base_folder, 'frames_crop')
    os.makedirs(crop_folder, exist_ok=True)

    count = 0
    crop_face_images = []
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if i in frame_indices:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            if len(faces) > 0:
                x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
                face = frame[y:y+h, x:x+w]
                face_resized = cv2.resize(face, (224, 224))

                crop_filename = f"crop_{i:03d}.jpg"
                crop_filepath = os.path.join(crop_folder, crop_filename)
                cv2.imwrite(crop_filepath, face_resized)

                crop_face_images.append(crop_filepath.replace("\\", "/"))


                img_tensor = transform(Image.fromarray(cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)))
                images_tensor.append(img_tensor)
                raw_frames.append(frame.copy())
                face_boxes.append((x, y, w, h))
                count += 1

                if count >= num_frames:
                    break

                if count % 3 == 0:  # 每隔3帧输出一次
                    debug_log(f"🧠 Frame {i}: RAM {round(psutil.virtual_memory().used / 1024**2, 2)} MB | CPU {psutil.cpu_percent()}%")

    cap.release()

    if len(images_tensor) < num_frames:
        print("[WARNING] Not enough face frames for prediction.")
        return face_images, []

    video_tensor = torch.stack(images_tensor).unsqueeze(0).to(device)
    with torch.no_grad():
        # confidence = model(video_tensor).item()
        debug_log("📦 Predicting...")
        debug_log(f"💡 RAM before prediction: {round(psutil.virtual_memory().used / 1024**2, 2)} MB")
        debug_log(f"💡 CPU: {psutil.cpu_percent()}%")
        confidence = get_model()(video_tensor).item()
    result = "Fake" if confidence >= 0.5 else "Real"
    color = (0, 0, 255) if result == "Fake" else (0, 255, 0)
    label_text = f"{result} ({confidence:.2f})"

    results = []
    for i in range(num_frames):
        frame = raw_frames[i]
        x, y, w, h = face_boxes[i]
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        filename = f"frame_{i:03d}.jpg"
        filepath_out = os.path.join(frames_folder, filename)
        cv2.imwrite(filepath_out, frame)

        face_images.append(filepath_out.replace("\\", "/"))

        results.append({
            "face_image": face_images[-1],
            "confidence": confidence,
            "result": result
        })

    output_video_path = os.path.join(base_folder, 'result_preview.mp4')
    h, w, _ = raw_frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_video_path, fourcc, 5.0, (w, h))
    for i in range(len(face_images)):
        frame_path = os.path.join(frames_folder, f"frame_{i:03d}.jpg")
        frame = cv2.imread(frame_path)
        out.write(frame)
    out.release()

    cloud_preview_url = output_video_path.replace("\\", "/")

    debug_log("📹 Output video done.")
    debug_log(f"✅ Final RAM:, {round(psutil.virtual_memory().used / 1024**2, 2)}, MB")
    debug_log(f"✅ Final CPU:, {psutil.cpu_percent()}, %")

    # ✅ 返回 Cloudinary 视频链接
    return face_images, crop_face_images, filepath.replace("\\", "/"), cloud_preview_url, results

debug_log(f"⏱️ Total processing time: {round(time.time() - start_time, 2)}s")

if __name__ == "__main__":
    app.run(debug=True)
