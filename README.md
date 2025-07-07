# Deepfake Detection System for Video-Based Facial Verification

A Flask-based web app that lets users upload a video and checks its authenticity (real vs. deepfake) using a self-trained CNN+LSTM model.

---

## 📂 Project Structure

```bash
FYP_DeepFake/
├── dataset/ ← raw Celeb-DF videos downloaded via Google Form
├── Train_Model/
│ ├── frames/ ← extracted frames output
│ │ ├── real/ ← subfolders, each with 20 extracted frames from a real video
│ │ └── fake/ ← subfolders, each with 20 extracted frames from a fake video
│ ├── extract_face_frames_224x224_20frames.py
│ ├── generate_train_index_fixed_path.py
│ ├── self_train/
│ │ ├── split_train_val.py ← unbalanced split
│ │ └── train_index.json
│ ├── new_train/
│ │ └── split_train_val_balance.py
│ └── Train_And_Eval_Unbalanced.py
├── create_db.py
├── app.py
├── .env
├── .gitignore
└── requirements.txt
```

---

## 🔧 Prerequisites

- Python 3.8 or higher  
- Git  
- Google account (to request and download Celeb-DF dataset)  
- (Optional) `venv` or `conda` for environment isolation

---

## 📥 Dataset Download

1. Visit the [Celeb-DF v2 README](https://github.com/yuezunli/celeb-deepfakeforensics/blob/master/Celeb-DF-v1/README.md).  
2. Scroll down, open the request form, and submit it to receive the Google Drive link.  
3. Download and extract the dataset into the project root under `dataset/`.

---

## 🛠️ Environment Setup

```bash
git clone https://github.com/your-username/Deepfake-Video-Detection-System-For-Faces.git
cd Deepfake-Video-Detection-System-For-Faces
python -m venv env
# Windows:
env\Scripts\activate
# macOS/Linux:
source env/bin/activate

pip install -r requirements.txt
```



## 🚀 Training the Model
1. Extract 20 face-focused frames per video
```bash
cd Train_Model
python extract_face_frames_224x224_20frames.py \
  --input ../dataset/ \
  --output ./frames/

```


## 2. Generate training index
```bash
python generate_train_index_fixed_path.py \
  --frames_dir ./frames/ \
  --output ./self_train/train_index.json
```

## 3. Split into unbalanced train/validation sets
```bash
cd self_train
python split_train_val.py \
  --index train_index.json \
  --train_out train_split.json \
  --val_out val_split.json
```


## 4. Create a balanced split
```bash
cd ../new_train
python split_train_val_balance.py \
  --index ../self_train/train_index.json \
  --train_out train_split_balanced.json \
  --val_out val_split_balanced.json
```

## 5. Train and evaluate (unbalanced)
```bash
cd ..
python Train_And_Eval_Unbalanced.py \
  --train_json self_train/train_split.json \
  --val_json self_train/val_split.json \
  --save_model best_unbalanced_model.pth
```

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
