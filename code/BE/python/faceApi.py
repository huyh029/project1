from flask import Flask, request, jsonify
import mysql.connector
from flask_cors import CORS
import base64
import secrets
from deepface import DeepFace
import cv2
import numpy as np

app = Flask(__name__)
CORS(app)

# ========================
# KẾT NỐI MYSQL
# ========================
def get_db_connection():
    return mysql.connector.connect(
        host='127.0.0.1',
        user='root',
        password='',
        database='userdb'
    )

# ========================
# EGISTER USER
# ========================
@app.route('/api/register', methods=['POST'])
def register():
    data = request.json
    username = data.get('username')
    password = data.get('password')
    img_data = data.get('img')

    if not username or not password or not img_data:
        return jsonify({'error': 'username, password và ảnh là bắt buộc'}), 400

    # Tách Base64 nếu có prefix data:image/png;base64,
    img_str = img_data.split(",")[1] if "," in img_data else img_data

    # Kiểm tra Base64 hợp lệ
    try:
        base64.b64decode(img_str)
    except Exception:
        return jsonify({'error': 'Ảnh không hợp lệ, phải là Base64'}), 400

    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # ✅ Kiểm tra username đã tồn tại chưa
        cursor.execute("SELECT * FROM user WHERE userName = %s", (username,))
        existing_user = cursor.fetchone()
        if existing_user:
            cursor.close()
            conn.close()
            return jsonify({'error': 'Tên người dùng đã tồn tại, vui lòng chọn tên khác'}), 409

        # Nếu chưa tồn tại thì thêm mới
        cursor.execute(
            "INSERT INTO user (userName, password, img) VALUES (%s, %s, %s)",
            (username, password, img_str)
        )
        conn.commit()
        cursor.close()
        conn.close()

        return jsonify({'message': '✅ Đăng ký thành công!'}), 201

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ========================
# LOGIN + KIỂM TRA KHUÔN MẶT
# ========================
@app.route('/api/login', methods=['POST'])
def login():
    data = request.json
    username = data.get('username')
    password = data.get('password')
    img_data = data.get('img')

    if not username or not password or not img_data:
        return jsonify({'error': 'username, password và ảnh bắt buộc'}), 400

    img_str = img_data.split(",")[1] if "," in img_data else img_data

    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            "SELECT * FROM user WHERE userName=%s AND password=%s",
            (username, password)
        )
        user = cursor.fetchone()
        cursor.close()
        conn.close()

        if not user:
            return jsonify({'error': 'Username hoặc password sai'}), 401

        # Chuyển Base64 → ảnh OpenCV
        nparr_input = np.frombuffer(base64.b64decode(img_str), np.uint8)
        img_input = cv2.imdecode(nparr_input, cv2.IMREAD_COLOR)

        nparr_db = np.frombuffer(base64.b64decode(user['img']), np.uint8)
        img_db = cv2.imdecode(nparr_db, cv2.IMREAD_COLOR)

        # So sánh bằng DeepFace
        try:
            result = DeepFace.verify(img_input, img_db, enforce_detection=True)
        except Exception as e:
            return jsonify({'error': f'Lỗi khi nhận diện khuôn mặt: {e}'}), 400

        if result['verified']:
            token = secrets.token_hex(16)
            return jsonify({'message': '✅ Login thành công', 'token': token})
        else:
            return jsonify({'error': 'Khuôn mặt không trùng khớp'}), 401

    except Exception as e:
        return jsonify({'error': str(e)}), 500
# ========================
# PHÂN TÍCH KHUÔN MẶT
# ========================
def convert_numpy(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(i) for i in obj]
    elif isinstance(obj, np.float32) or isinstance(obj, np.float64):
        return float(obj)
    elif isinstance(obj, np.int32) or isinstance(obj, np.int64):
        return int(obj)
    else:
        return obj

@app.route('/api/analyze', methods=['POST'])
def analyze_face():
    data = request.json
    img_data = data.get('img')

    if not img_data:
        return jsonify({'error': 'Ảnh là bắt buộc'}), 400

    img_str = img_data.split(",")[1] if "," in img_data else img_data

    try:
        nparr = np.frombuffer(base64.b64decode(img_str), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        result = DeepFace.analyze(img, actions=['age','gender','emotion','race'], enforce_detection=True)
        result = convert_numpy(result)

        return jsonify({'result': result})

    except Exception as e:
        return jsonify({'error': f'Lỗi khi phân tích khuôn mặt: {e}'}), 400

# ========================
# Chạy Flask
# ========================
if __name__ == '__main__':
    app.run(debug=True)
