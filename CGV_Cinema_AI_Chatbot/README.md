# CHATBOT TƯ VẤN CHỌN PHIM CGV - BÀI TẬP LỚN MÔN TRÍ TUỆ NHÂN TẠO

## 1. TỔNG QUAN DỰ ÁN
Hệ thống Chatbot tư vấn chọn phim chiếu rạp CGV Cinemas được xây dựng dựa trên các nguyên lý và thuật toán của môn Trí tuệ Nhân tạo:
- **Cơ sở tri thức (Knowledge Representation)**: Quản lý 65 bộ phim chiếu rạp CGV với các thuộc tính thể loại, nhãn độ tuổi P, C13, C16, C18, cụm rạp, suất chiếu, định dạng 2D, 3D, IMAX, rating, giá vé, combo bắp nước, mô tả cốt truyện (`description`) và kho từ khóa liên tưởng ngữ nghĩa siêu rộng hàng ngàn từ (`keywords`).
- **Huấn luyện từ khóa giao tiếp tự nhiên (Natural Speech Vocabulary Training)**: Tự động huấn luyện và mở rộng 60-80+ từ khóa tự nhiên cho **TOÀN BỘ 65 bộ phim** trong `data/movies.json` bao gồm các từ ngữ giao tiếp người thật (xả stress, cười bể bụng, khóc hết nước mắt, thót tim, chữa lành tâm hồn...), tên diễn viên/đạo diễn (Trấn Thành, Tuấn Trần...) và liên tưởng đồ vật, con vật, phương tiện (con khỉ, vượn, máy bay, rô bốt, quật mộ, phong thủy...).
- **Xử lý ngôn ngữ tự nhiên (NLP & IR)**: Nhận diện ý định Intent, trích xuất thực thể NER, tiền xử lý văn bản tiếng Việt, quy đổi cụm từ hành động/tình tiết tự nhiên và phân tích từ điển cảm xúc/tâm trạng (buồn, stress, chán đời...).
- **Tìm kiếm phim theo liên tưởng ngữ nghĩa (Semantic Concept & Content-based IR)**: Sử dụng thuật toán TF-IDF + Cosine Similarity kết hợp điểm thưởng trùng khớp từ khóa (Keyword Bonus) để so khớp câu hỏi tự nhiên của người dùng với kho từ khóa mở rộng (ví dụ: "con khỉ" -> Godzilla x Kong; "máy bay" -> Chàng Nữ Phi Công; "Trấn Thành" -> Mai; "rô bốt biến hình" -> Transformers Một).
- **Giao tiếp tự nhiên, Trực diện & Đủ ý (Direct Natural NLG)**:
  + Chatbot xưng hô "mình - bạn", trả lời trực diện vào thẳng vấn đề ngay câu đầu tiên: *"Dạ theo như yêu cầu tìm '[Từ khóa]' của bạn thì đó chính là bộ phim [Tên Phim Top 1] đúng không nhỉ? 🎬"*.
  + Loại bỏ hoàn toàn các câu văn mẫu robot khô khan và các danh sách câu hỏi dồn dập.
- **Hỏi lại từng bước một (Step-by-Step Single-Question Interaction)**:
  + Chỉ hỏi **DUY NHẤT 1 câu** tự nhiên ở mỗi lượt giao tiếp (hỏi người đi cùng / có trẻ em không ➔ đợi khách trả lời ➔ mới hỏi câu tiếp theo).
  + Tự động nhắc nhở giới hạn độ tuổi (`C13`, `C16`, `C18`) một cách duyên dáng và đề xuất chuyển sang phim Hoạt hình / Gia đình nhãn `P` khi có trẻ em đi cùng.
- **Suy luận logic hình thức (Horn Clauses)**: Áp dụng các mệnh đề Horn để lọc loại trừ cứng các phim vi phạm quy định độ tuổi, sai rạp hoặc sai khung giờ.
- **Hệ khuyến nghị lai và Trí tuệ nhân tạo giải thích được (Explainable AI - XAI)**: Kết hợp chỉ số Jaccard Similarity, Cosine TF-IDF, điểm Rating và mức độ trùng khớp từ khóa để xếp hạng Top 3 và sinh lời giải thích XAI ngắn gọn, minh bạch.
- **Giao diện người dùng**: Ứng dụng web Streamlit phong cách CGV với giao diện VIP Concierge, hiệu ứng chữ chạy streaming và công tắc bật tắt chế độ Gemini AI.

---

## 2. ÁNH XẠ THUẬT TOÁN, CÔNG NGHỆ VÀ VỊ TRÍ FILE

- **Cơ sở tri thức (Knowledge Representation)**:
  + File dữ liệu JSON: `data/movies.json` (lưu trữ 65 bộ phim chứa description và hàng ngàn keywords liên tưởng tự nhiên)
  + File sinh dữ liệu: `data/generate_movies.py` (script khởi tạo và kiểm tra CSDL)
  + File quản lý tri thức: `src/knowledge_base.py` (lớp `MovieKnowledgeBase` quản lý truy vấn)

- **Xử lý Ngôn ngữ Tự nhiên & Thu hồi Thông tin (NLP, IR & Slot Filling)**:
  + File chính: `src/nlp_processor.py` (lớp `NLPProcessor`)
  + Hàm tiền xử lý văn bản: `preprocess_text` (chữ thường, bỏ dấu câu, bỏ từ dừng)
  + Hàm bỏ dấu tiếng Việt: `remove_accents` (hỗ trợ tìm kiếm không dấu / gõ gần đúng)
  + Hàm nhận diện ý định: `recognize_intent` (phân loại các intent)
  + Hàm trích xuất thực thể: `extract_entities` (bóc tách tuổi 2 chiều, số người, trẻ em, vị bắp, thể loại, rạp, giờ)
  + Hàm phân tích cảm xúc: `detect_emotion` (bắt các trạng thái tâm lý buồn, stress, chán)
  + Hàm tư vấn ngược lại: `detect_missing_slots` (phát hiện câu hỏi chung chung để tư vấn gợi mở)

- **Suy luận Logic mệnh đề Horn (Horn Clauses Filter)**:
  + File chính: `src/reasoning_engine.py` (lớp `ReasoningEngine`)
  + Hàm suy luận luật tuổi: `check_age_rule` (loại trừ phim C13, C16, C18 khi chưa đủ tuổi)
  + Hàm suy luận luật rạp: `check_cinema_rule` (loại phim không chiếu ở rạp yêu cầu)
  + Hàm suy luận luật giờ: `check_time_rule` (lọc suất chiếu trong khung giờ)
  + Hàm lọc tổng hợp: `filter_candidates` (trả về danh sách ứng viên hợp lệ)

- **Thuật toán Gợi ý và Explainable AI (Recommender System và XAI)**:
  + File chính: `src/recommender.py` (lớp `MovieRecommender`)
  + Thuật toán Jaccard: `_compute_genre_similarity` (so khớp thể loại)
  + Thuật toán Cosine TF-IDF: `_compute_text_cosine_similarity` (đo độ tương đồng nội dung và từ khóa không dấu/gần đúng)
  + Trích xuất từ khóa trùng: `_extract_matching_keywords` (bóc tách từ khóa khớp trực tiếp)
  + Mô hình AI giải thích XAI: `_generate_explanation` (sinh lý do ngắn gọn, cô đọng)
  + Hàm xếp hạng đa tiêu chí: `rank_and_recommend` (tính điểm tổng hợp và cộng thưởng từ khóa khớp)

- **Bộ não điều phối và Mô hình Dual-Mode**:
  + File chính: `src/chatbot.py` (lớp `CGVChatbot`)
  + Hàm điều phối: `process_message` (tích hợp tư vấn ngược lại và xếp hạng IR)
  + Hàm sinh đối thoại Offline: `_generate_natural_offline_response` (văn phong trực diện, ngắn gọn nhưng đủ ý, hỏi lại từng câu một)
  + Hàm gọi Gemini AI: `_generate_llm_response` (kết nối `gemini-3.6-flash` qua thư viện `google-genai`)

- **Giao diện Web và Kiểm thử**:
  + File giao diện: `app.py` (ứng dụng Streamlit, giao diện VIP Concierge và công tắc bật tắt Gemini)
  + File kiểm thử tự động: `tests/test_system.py` (chạy 8 bộ test case tự động đạt 100% PASS)
  + File khởi chạy nhanh: `run.bat` (script bật 1-click trên Windows)
  + File báo cáo học thuật: `BAOCAO_BTL.md` (tài liệu báo cáo chi tiết và câu hỏi phản biện)

---

## 3. PHÂN CÔNG CÔNG VIỆC NHÓM 5 NGƯỜI

- **Thành viên 1 (Trưởng nhóm - System Architect)**:
  Phụ trách kiến trúc tổng thể, module `src/chatbot.py`, `requirements.txt`, `README.md`.

- **Thành viên 2 (Data và Knowledge Engineer)**:
  Phụ trách mô hình tri thức, kho từ khóa liên tưởng natural keywords, `data/generate_movies.py`, `data/movies.json` và `src/knowledge_base.py`.

- **Thành viên 3 (NLP và IR Engineer)**:
  Phụ trách tiền xử lý ngôn ngữ, nhận diện intent, trích xuất NER, phân tích cảm xúc và tính năng tư vấn ngược lại trong `src/nlp_processor.py`.

- **Thành viên 4 (Reasoning và Recommender Specialist)**:
  Phụ trách động cơ suy luận Horn Clauses, thuật toán TF-IDF so khớp từ khóa liên tưởng và giải thích XAI trong `src/reasoning_engine.py` và `src/recommender.py`.

- **Thành viên 5 (Frontend và QA Tester)**:
  Phụ trách giao diện `app.py`, kiểm thử tự động 8/8 bộ test case trong `tests/test_system.py`, `run.bat` và `BAOCAO_BTL.md`.

---

## 4. HƯỚNG DẪN KHỞI CHẠY

### Cách 1: Khởi chạy nhanh trên Windows
Nhấp đúp chuột vào file `run.bat` trong thư mục dự án.

### Cách 2: Chạy bằng dòng lệnh Terminal hoặc PowerShell
- **Bước 1**: Cài đặt thư viện
  ```bash
  pip install -r requirements.txt
  ```
- **Bước 2**: Chạy kiểm thử tự động
  ```bash
  python tests/test_system.py
  ```
- **Bước 3**: Khởi chạy ứng dụng web
  ```bash
  streamlit run app.py
  ```
  *(Mở trình duyệt tại địa chỉ http://localhost:8501)*
