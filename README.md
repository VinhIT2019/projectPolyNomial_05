\# Hướng dẫn chạy và sử dụng ứng dụng Streamlit



\## 1. Chuẩn bị \& clone mã nguồn từ GitHub



```bash

git clone https://github.com/VinhIT2019/projectPolyNomial\_05.git

cd projectPolyNomial\_05

```



Trong thư mục `projectPolyNomial\_05` có các file quan trọng để chạy web Streamlit:



\* `app.py`

\* `requirements.txt` 



---



\## 2. Tạo và kích hoạt môi trường ảo (virtual environment)



\### Trên Windows 



```powershell

python -m venv venv

.\\venv\\Scripts\\Activate

```



\### Trên Linux / macOS 



```bash

python3 -m venv venv

source venv/bin/activate

```



---



\## 3. Cài đặt thư viện 



Trong môi trường ảo `venv`, chạy lệnh:



```bash

pip install --upgrade pip

pip install -r requirements.txt

```



---



\## 4. Chạy ứng dụng Streamlit



Vẫn trong môi trường ảo `venv`, chạy lệnh:



```bash

streamlit run app.py

```



Ứng dụng sẽ mở giao diện web Streamlit để bạn tương tác.



---



\## 5. Hướng dẫn sử dụng giao diện web



\### 5.1. Tải dữ liệu (Upload Dataset)



Ở \*\*sidebar (cột trái)\*\*, bấm nút \*\*"Tải dataset"\*\*.



\* Ứng dụng sẽ:



&nbsp; \* Hiển thị kích thước dữ liệu (số dòng × số cột)

&nbsp; \* Hiển thị preview 5 dòng đầu tiên



---



\### 5.2. Chọn Target và Features



Sau khi tải dữ liệu:



\* \*\*Target (biến cần dự đoán):\*\*



&nbsp; \* Mặc định là \*\*cột cuối cùng\*\* trong dataset.



\* \*\*Features (đầu vào mô hình):\*\*



&nbsp; \* Mặc định là \*\*tất cả các cột còn lại\*\*, ngoại trừ cột target.



Bạn có thể thay đổi lựa chọn này ngay trên giao diện.



---



\### 5.3. Chọn mô hình và tuỳ chỉnh tham số



Trong phần \*\*"Chọn mô hình"\*\* ở sidebar, có 5 lựa chọn:



1\. `LinearRegression (no polynomial)`

&nbsp;  Hồi quy tuyến tính thuần, không tạo đặc trưng đa thức.



2\. `Poly3 + LinearRegression`

&nbsp;  Sinh đặc trưng đa thức bậc 3.



3\. `Poly4 + RidgeCV`

&nbsp;  Sinh đặc trưng đa thức bậc 4 + RidgeCV.



4\. `Poly4 + LassoCV`

&nbsp;  Sinh đặc trưng đa thức bậc 4 + LassoCV.



5\. `Poly4 + ElasticNetCV`

&nbsp;  Sinh đặc trưng đa thức bậc 4 + ElasticNetCV (kết hợp L1 và L2).



Tùy chỉnh các tham số:



\* `test\_size`: tỉ lệ dữ liệu dùng làm tập test.

\* `random\_state`: seed để tái lập kết quả.

\* Các tham số riêng của từng mô hình

---



\### 5.4. Train \& đánh giá mô hình



Nhấn nút \*\*"Train / Evaluate model"\*\* để chạy huấn luyện và đánh giá.



Ứng dụng thực hiện các bước sau:



1\. \*\*Chia train/test\*\*

&nbsp;  Tách dữ liệu theo `test\_size`.



2\. \*\*Huấn luyện mô hình đã chọn\*\*



3\. \*\*Báo cáo chỉ số đánh giá:\*\*



&nbsp;  \* R² trên tập train

&nbsp;  \* R² trên tập test

&nbsp;  \* RMSE trên tập test

&nbsp;    

4\. \*\*Đánh giá Cross-Validation nâng cao:\*\*



&nbsp;  \* Dùng RepeatedKFold với:



&nbsp;    \* K-fold = 10

&nbsp;    \* lặp lại 3 lần

&nbsp;  \* Hiển thị giá trị trung bình ± độ lệch chuẩn cho:



&nbsp;    \* R² CV

&nbsp;    \* RMSE CV



5\. \*\*Biểu đồ dự đoán:\*\*



6\. \*\*Bảng hệ số tuyến tính (Model Coefficients):\*\*



