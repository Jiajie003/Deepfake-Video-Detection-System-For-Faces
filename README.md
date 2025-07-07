# Deepfake Detection System for Video-Based Facial Verification

A Flask-based web app that lets users upload a video and checks its authenticity (real vs. deepfake) using a self-trained CNN+LSTM model.

---

## 📂 Project Structure

```bash
FYP_DeepFake/
├── dataset/ ← raw Celeb-DF videos downloaded via Google Form
├── Train Model/
│ ├── frames/ ← extracted frames output
│ │ ├── real/ ← subfolders, each with 20 extracted frames from a real video
│ │ └── fake/ ← subfolders, each with 20 extracted frames from a fake video
│ ├── extract_face_frames_224x224_20frames.py
│ ├── generate_train_index_fixed_path.py
│ ├── self train/
│ │ ├── split_train_val.py ← unbalanced split
│ │ └── train_index.json
│ ├── new train/
│ │ └── split_train_val_balance.py
│ └── Train_And_Eval_Unbalanced.py
├── create_db.py
├── app.py
├── check_gpu_available.py        ← utility to verify CUDA/PyTorch setup
├── .env
├── .gitignore
└── requirements.txt
```

---

## 🔧 Prerequisites

- Python 3.8 or higher  
- Git  
- Google account (to request and download Celeb-DF dataset)
- NVIDIA GPU + CUDA Toolkit (optional, for GPU acceleration)
- (Optional) venv or conda for environment isolation

---

## 📥 Dataset Download

1. Visit the [Celeb-DF v2 README](https://github.com/yuezunli/celeb-deepfakeforensics/blob/master/Celeb-DF-v1/README.md).  
2. Scroll down, open the request form, and submit it to receive the Google Drive link.  
3. Download and extract the dataset into the project root under dataset/.

---

## 🛠️ Environment Setup

```bash
git clone https://github.com/Jiajie003/Deepfake-Video-Detection-System-For-Faces.git
cd Deepfake-Video-Detection-System-For-Faces
python -m venv env
# Windows:
env\Scripts\activate
# macOS/Linux:
source env/bin/activate

pip install -r requirements.txt
```
---
## 🔍 Verify GPU & CUDA Setup
A small script check_gpu_available.py is provided to confirm your PyTorch/CUDA installation:
```bash
python check_gpu_available.py
```
-  ✅ If you see “CUDA is available!”, you can train on GPU.
-  ❌ Otherwise, run:
```bash
nvidia-smi
```
-  to check your driver’s CUDA version (e.g. “CUDA Version: 12.9”).


## ⚙️ Choosing the Right CUDA Build for PyTorch
1. Check driver’s CUDA version
```bash
nvidia-smi
```

2. Install matching PyTorch wheel
-  For CUDA ≥ 12.1:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
-  Otherwise, use CUDA 11.8 build:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

3. Re-run the GPU check
```bash
python check_gpu_available.py
```

Verify that the printed PyTorch version includes +cu118 or +cu121 and that CUDA is available!


---




## 🚀 Training the Model
1. Extract 20 face-focused frames per video
```bash
cd Train Model
python extract_face_frames_224x224_20frames.py \
  --input ../dataset/ \
  --output ./frames/
```


## 2. Generate training index
```bash
python generate_train_index_fixed_path.py \
  --frames_dir ./frames/ \
  --output ./self train/train_index.json
```

## 3. Split into unbalanced train/validation sets
```bash
cd self train
python split_train_val.py \
  --index train_index.json \
  --train_out train_split.json \
  --val_out val_split.json
```


## 4. Create a balanced split
```bash
cd ../new train
python split_train_val_balance.py \
  --index ../self train/train_index.json \
  --train_out train_split_balanced.json \
  --val_out val_split_balanced.json
```

## 5. Train and evaluate (unbalanced)
```bash
cd ..
python Train_And_Eval_Unbalanced.py \
  --train_json self train/train_split.json \
  --val_json self_train/val_split.json \
  --save_model best_unbalanced_model.pth
```
---

### 🔑 Generate a Gmail App Password
1. Go to your Google Account’s security settings:  
   https://myaccount.google.com/security  
2. Under **“Signing in to Google”**, click **App passwords**.  
3. You may be prompted to re-enter your Google password or complete 2-step verification.  
4. In the **“Select app”** dropdown choose **Mail**, and in **“Select device”** choose **Other (Custom name)**.  
5. Enter a name (e.g. `DeepfakeApp`) and click **Generate**.  
6. Google will show you a 16-character password. Copy it.  
7. Paste that password into your `.env` as the value for `MAIL_PASSWORD`:
```ini
MAIL_USERNAME='your.email@gmail.com'
MAIL_PASSWORD='your_16_character_app_password'
```

---

This will guide anyone through creating and using a Gmail App Password for your Flask-Mail setup.


## 🌐 Running the Web App
1. Initialize the database
```bash
python create_db.py
```

2. Configure email settings in .env
```bash
# Email configuration
MAIL_USERNAME='123_Sample@gmail.com'
MAIL_PASSWORD='aaaa bbbb cccc dddd'
```

3. Start the Flask server
```bash
python app.py
```

4. Open your browser and navigate to http://127.0.0.1:5000/
