# Image-Captioning
# Image-Captioning

## Giới thiệu
Dự án Image Captioning là một ứng dụng sử dụng mô hình học sâu để tạo chú thích tự động cho hình ảnh. Mô hình này kết hợp giữa mạng nơ-ron tích chập (CNN) để trích xuất đặc trưng từ hình ảnh và mạng nơ-ron hồi quy (RNN) với cơ chế Attention để tạo ra các chú thích văn bản tương ứng.

Repo chỉ mang tính chất học tậptập
## Dataset
Dự án sử dụng tập dữ liệu Flickr8k, bao gồm 8000 hình ảnh và mỗi hình ảnh đi kèm với 5 chú thích mô tả. Tập dữ liệu này rất phổ biến trong các bài toán về Image Captioning.

## Cách giải quyết bài toán
1. **Trích xuất đặc trưng từ hình ảnh**: Sử dụng mạng ResNet50 đã được huấn luyện trước để trích xuất các đặc trưng từ hình ảnh.
2. **Tạo chú thích**: Sử dụng mạng LSTM với cơ chế Attention để tạo ra các chú thích dựa trên các đặc trưng hình ảnh đã trích xuất.
3. **Huấn luyện mô hình**: Mô hình được huấn luyện trên tập dữ liệu Flickr8k với các chú thích tương ứng.

## Hướng dẫn cài đặt

### Yêu cầu hệ thống
- Python 3.x
- pip

### Các bước cài đặt

1. **Clone repo**
    ```sh
    git clone <repository-url>
    cd <repository-directory>
    ```

2. **Cài đặt các thư viện cần thiết**
    ```sh
    pip install -r requirements.txt
    ```

3. **Chuẩn bị dữ liệu**
    - Tải tập dữ liệu Flickr8k và giải nén vào thư mục [data](http://_vscodecontentref_/0).
    - Đảm bảo rằng các hình ảnh nằm trong thư mục [Images](http://_vscodecontentref_/1) và file chú thích [captions.txt](http://_vscodecontentref_/2) nằm trong thư mục [data](http://_vscodecontentref_/3).

4. **Xây dựng từ vựng**
    ```sh
    python build_vocab.py
    ```

5. **Huấn luyện mô hình**
    ```sh
    python train.py
    ```

6. **Chạy ứng dụng web**
    ```sh
    python app.py
    ```

7. **Truy cập ứng dụng**
    - Mở trình duyệt và truy cập `http://127.0.0.1:5000/` để sử dụng ứng dụng.

## Sử dụng
- Tải lên một hình ảnh và ứng dụng sẽ tự động tạo chú thích cho hình ảnh đó.

## Thư mục dự án
- [app.py](http://_vscodecontentref_/4): Ứng dụng web Flask.
- [build_vocab.py](http://_vscodecontentref_/5): Tạo từ vựng từ tập dữ liệu chú thích.
- [data](http://_vscodecontentref_/6): Thư mục chứa dữ liệu.
- [data_procesing.py](http://_vscodecontentref_/7): Xử lý dữ liệu và tạo dataset.
- [model.py](http://_vscodecontentref_/8): Định nghĩa mô hình Encoder-Decoder với Attention.
- [train.py](http://_vscodecontentref_/9): Huấn luyện mô hình.
- [templates](http://_vscodecontentref_/10): Thư mục chứa template HTML.
- [static](http://_vscodecontentref_/11): Thư mục chứa các file tĩnh (hình ảnh tải lên).
