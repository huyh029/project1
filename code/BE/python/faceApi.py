from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import secrets
from deepface import DeepFace
import cv2
import numpy as np
from pymongo import MongoClient
app = Flask(__name__)
CORS(app)

# ========================
# KẾT NỐI MONGODB
# ========================
mongo_uri = "mongodb+srv://huyh01480_db_user:zxvAwzAhr8yk3lWe@cluster0.n8pboqq.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(mongo_uri)

db = client["userdb"]           # database
users_col = db["users"]         # collection tương đương bảng user


# ========================
# REGISTER USER
# ========================


from deepface import DeepFace
import cv2
import numpy as np
import base64

@app.route('/api/register', methods=['POST'])
def register():
    data = request.json

    username = data.get('username')
    password = data.get('password')
    img_data = data.get('img')
    
    fullName = data.get('fullName')
    email = data.get('email')
    phone = data.get('phone')
    gender = data.get('gender')

    # Kiểm tra dữ liệu bắt buộc
    if not username or not password or not img_data:
        return jsonify({'error': 'username, password và ảnh là bắt buộc'}), 400

    img_str = img_data.split(",")[1] if "," in img_data else img_data

    # Kiểm tra Base64 hợp lệ
    try:
        decoded = base64.b64decode(img_str)
    except Exception:
        return jsonify({'error': 'Ảnh không hợp lệ, phải là Base64'}), 400

    # Chuyển base64 → ảnh OpenCV
    try:
        nparr = np.frombuffer(decoded, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({'error': 'Ảnh không thể đọc được'}), 400

    except Exception:
        return jsonify({'error': 'Không thể xử lý ảnh'}), 400

    # ===== KIỂM TRA ẢNH CÓ MẶT KHÔNG =====
    try:
        DeepFace.extract_faces(img, enforce_detection=True)
    except Exception:
        return jsonify({'error': 'Ảnh không có khuôn mặt hợp lệ'}), 400

    try:
        # Kiểm tra username trùng
        existing_user = users_col.find_one({"userName": username})
        if existing_user:
            return jsonify({'error': 'Tên người dùng đã tồn tại'}), 409

        # Dữ liệu người dùng chỉ gồm các trường yêu cầu
        user_info = {
            "userName": username,
            "password": password,
            "img": img_str,
            "fullName": fullName,
            "email": email,
            "phone": phone,
            "gender": gender,
        }

        # Lưu vào MongoDB
        users_col.insert_one(user_info)

        return jsonify({'message': 'Đăng ký thành công!', 'user': username}), 201

    except Exception as e:
        return jsonify({'error': str(e)}), 500
# LOGIN + KIỂM TRA KHUÔN MẶT
# ========================
@app.route('/api/login', methods=['POST'])
def login():
    data = request.json
    username = data.get('username')
    password = data.get('password')
    img_data = data.get('img')

    if not username or not password or not img_data:
        return jsonify({'error': 'username, password và ảnh là bắt buộc'}), 400

    img_str = img_data.split(",")[1] if "," in img_data else img_data

    try:
        # Lấy user từ MongoDB
        user = users_col.find_one({"userName": username, "password": password})

        if not user:
            return jsonify({'error': 'Username hoặc password sai'}), 401

        # Base64 → ảnh
        nparr_input = np.frombuffer(base64.b64decode(img_str), np.uint8)
        img_input = cv2.imdecode(nparr_input, cv2.IMREAD_COLOR)

        nparr_db = np.frombuffer(base64.b64decode(user['img']), np.uint8)
        img_db = cv2.imdecode(nparr_db, cv2.IMREAD_COLOR)

        # So sánh mặt
        try:
            result = DeepFace.verify(img_input, img_db, enforce_detection=True)
        except Exception as e:
            return jsonify({'error': f'Lỗi khi nhận diện khuôn mặt: {e}'}), 400

        # Tính %
        distance = result.get("distance", 1)
        similarity = max(0, (1 - distance)) * 100
        similarity = round(similarity, 2)

        if result['verified']:
            token = secrets.token_hex(16)

            # Xóa _id vì không serializable
            user_info = {
                "userName": user.get("userName"),
                "fullName": user.get("fullName"),
                "email": user.get("email"),
                "phone": user.get("phone"),
                "gender": user.get("gender"),
                "img": user.get("img")
            }

            return jsonify({
                'message': 'Login thành công!',
                'token': token,
                'similarity': similarity,
                'user': user_info
            })
        else:
            return jsonify({
                'error': 'Khuôn mặt không trùng khớp',
                'similarity': similarity
            }), 401

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
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
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

@app.route('/api/compare', methods=['POST'])
def compare_faces():
    data = request.json
    img1_data = data.get("img1")
    img2_data = data.get("img2")

    if not img1_data or not img2_data:
        return jsonify({'error': 'Cần 2 ảnh base64 để so sánh'}), 400

    # Tách phần base64 (nếu có prefix data:image/png;base64,)
    img1_str = img1_data.split(",")[1] if "," in img1_data else img1_data
    img2_str = img2_data.split(",")[1] if "," in img2_data else img2_data

    try:
        # Base64 → OpenCV image
        nparr1 = np.frombuffer(base64.b64decode(img1_str), np.uint8)
        img1 = cv2.imdecode(nparr1, cv2.IMREAD_COLOR)

        nparr2 = np.frombuffer(base64.b64decode(img2_str), np.uint8)
        img2 = cv2.imdecode(nparr2, cv2.IMREAD_COLOR)

        # So sánh khuôn mặt bằng DeepFace
        try:
            result = DeepFace.verify(img1, img2, enforce_detection=True)
        except Exception as e:
            return jsonify({'error': f'Lỗi nhận diện: {e}'}), 400

        # Tính phần trăm giống nhau
        distance = result.get("distance", 1)
        similarity = max(0, (1 - distance)) * 100
        similarity = round(similarity, 2)

        return jsonify({
            "verified": result.get("verified", False),
            "distance": distance,
            "similarity": similarity
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ========================
# RUN
# ========================
if __name__ == '__main__':
    app.run(debug=True)
