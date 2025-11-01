# Hướng dẫn chạy và sử dụng ứng dụng Streamlit

## 1. Chuẩn bị & clone mã nguồn từ GitHub

```bash
git clone https://github.com/VinhIT2019/projectPolyNomial_05.git
cd projectPolyNomial_05
```

Trong thư mục `projectPolyNomial_05` có các file quan trọng để chạy web Streamlit:

* `app.py`
* `requirements.txt`

---

## 2. Tạo và kích hoạt môi trường ảo (virtual environment)

### Trên Windows

```powershell
python -m venv venv
.\venv\Scripts\Activate
```

### Trên Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Cài đặt thư viện

Trong môi trường ảo `venv`, chạy lệnh:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 4. Chạy ứng dụng Streamlit

Vẫn trong môi trường ảo `venv`, chạy lệnh:

```bash
streamlit run app.py
```

Ứng dụng sẽ mở giao diện web Streamlit để bạn tương tác.

---

## 5. Hướng dẫn sử dụng giao diện web

### 5.1. Tải dữ liệu (Upload Dataset)

Ở **sidebar (cột trái)**, bấm nút **"Tải dataset"**.

Ứng dụng sẽ:

* Hiển thị kích thước dữ liệu (số dòng × số cột)
* Hiển thị preview 5 dòng đầu tiên

---

### 5.2. Chọn Target và Features

Sau khi tải dữ liệu:

* **Target (biến cần dự đoán):**

  * Mặc định là **cột cuối cùng** trong dataset.

* **Features (đầu vào mô hình):**

  * Mặc định là **tất cả các cột còn lại**, ngoại trừ cột target.

Bạn có thể thay đổi lựa chọn này ngay trên giao diện.

---

### 5.3. Chọn mô hình và tuỳ chỉnh tham số

Trong phần **"Chọn mô hình"** ở sidebar, có 5 lựa chọn:

1. `LinearRegression (no polynomial)`
   Hồi quy tuyến tính thuần (không tạo biến đa thức).

2. `Poly4 + LinearRegression`
   Sinh đặc trưng đa thức bậc 4 rồi fit LinearRegression.

3. `Poly4 + RidgeCV`
   Sinh đặc trưng đa thức bậc 4 + RidgeCV.

4. `Poly4 + LassoCV`
   Sinh đặc trưng đa thức bậc 4 + LassoCV.

5. `Poly4 + ElasticNetCV`
   Sinh đặc trưng đa thức bậc 4 + ElasticNetCV (kết hợp L1 và L2).

Các tham số có thể chỉnh:

* `test_size`: tỉ lệ dữ liệu dùng làm tập test.
* `random_state`: seed để tái lập kết quả.
* Các tham số riêng của từng mô hình.

---

### 5.4. Train / Evaluate model

Nhấn nút **"Train / Evaluate model"** để chạy huấn luyện và đánh giá.

Ứng dụng thực hiện các bước sau:

1. **Chia train/test**
   Tách dữ liệu theo `test_size`.

2. **Huấn luyện mô hình đã chọn**

3. **Báo cáo chỉ số đánh giá:**

   * R² trên tập train
   * R² trên tập test
   * RMSE trên tập test

4. **Đánh giá Cross-Validation nâng cao:**

   * Repeated K-Fold (ví dụ K=10, lặp lại N lần)
   * Báo cáo:

     * R² CV
     * RMSE CV

5. **Biểu đồ dự đoán:**
   So sánh giá trị thật vs giá trị dự đoán (scatter plot).

6. **Bảng hệ số tuyến tính (Model Coefficients):**
   Hiển thị các hệ số quan trọng nhất (ví dụ top feature có ảnh hưởng dương / âm).
