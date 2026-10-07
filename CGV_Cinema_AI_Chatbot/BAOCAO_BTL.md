# BÁO CÁO BÀI TẬP LỚN MÔN TRÍ TUỆ NHÂN TẠO
## Đề Tài: "Xây Dựng Chatbot Tư Vấn Chọn Phim Cho Rạp CGV"

---

## PHẦN 1: GIỚI THIỆU & ĐẶT VẤN ĐỀ

Trong kỷ nguyên bùng nổ thông tin và dịch vụ giải trí số, khán giả thường gặp phải hiện tượng **"Quá tải lựa chọn" (Choice Overload)** khi truy cập các hệ thống rạp chiếu phim với hàng chục bộ phim, đa dạng thể loại, khung giờ và quy định độ tuổi khác nhau.

Đề tài **"Xây dựng Chatbot tư vấn chọn phim cho rạp CGV"** ứng dụng các kỹ thuật cốt lõi của **Trí tuệ nhân tạo (AI)** nhằm:
- Cho phép người dùng tương tác tự nhiên bằng tiếng Việt theo cảm xúc hoặc câu hỏi ngẫu nhiên.
- Tìm kiếm phim theo mô tả liên tưởng ngữ nghĩa đồ vật, con vật, phương tiện và kho từ khóa mở rộng (Semantic Concept & Content-based IR) bằng thuật toán TF-IDF + Cosine Similarity.
- Tư vấn ngược lại chủ động (Proactive Reverse Advising) khi nhận thấy thông tin bị thiếu hoặc cần làm rõ nhu cầu khách hàng.
- Tự động suy luận logic để lọc các ứng viên thỏa mãn quy định pháp luật về độ tuổi và tính khả thi về rạp, suất chiếu.
- Sử dụng thuật toán khuyến nghị thông minh để chọn ra Top 3 bộ phim tối ưu nhất kèm lời giải thích minh bạch (Explainable AI - XAI).

---

## PHẦN 2: KIẾN TRÚC HỆ THỐNG VÀ CƠ SỞ TRI THỨC

### 1. Kiến trúc phân tầng (4-Tier Architecture)
Hệ thống được thiết kế thành 4 tầng module hóa độc lập:
1. **Tầng Giao Diện (Presentation Layer - `app.py`)**: Ứng dụng Web Streamlit cung cấp giao diện tương tác 2 chiều (Chat Window & Movie Cards Panel), hỗ trợ giao diện VIP Concierge và ô tích hợp Gemini API Key linh hoạt.
2. **Tầng Xử Lý Ngôn Ngữ Tự Nhiên & IR (NLP & IR Layer - `nlp_processor.py`)**: Chuẩn hóa câu thoại tiếng Việt, nhận dạng ý định (Intent), trích xuất thực thể (NER đa chiều), phân tích cảm xúc (Emotion Lexicon) và kiểm tra thiếu dữ kiện để hỏi ngược lại (Proactive Slot Filling).
3. **Tầng Suy Luận Logic & Khuyến Nghị (Reasoning & Recommendation Layer - `reasoning_engine.py`, `recommender.py`)**: Thực thi mệnh đề Horn, tìm kiếm ngữ nghĩa liên tưởng theo kho từ khóa keywords/description (TF-IDF Cosine Match không dấu & gần đúng) và thuật toán Content-based Hybrid Ranking.
4. **Tầng Sinh Lời Thoại Kép (Dual-Mode Dialogue Generation - `chatbot.py`)**:
   - *Chế độ 1 (Mặc định)*: Bộ sinh lời thoại tự nhiên nội bộ (Offline Natural NLG) - văn phong tư vấn viên CGV thân thiện, có duyên, 100% không cần mạng.
   - *Chế độ 2 (Mở rộng)*: Tích hợp Gemini LLM Hybrid khi người dùng/thầy cô muốn kiểm tra khả năng đối thoại tự do của mô hình ngôn ngữ lớn.
5. **Tầng Cơ Sở Tri Thức (Knowledge Base Layer - `knowledge_base.py`, `movies.json`)**: Lưu trữ và quản lý dữ liệu 65 Phim với đầy đủ thông tin chi tiết, mô tả cốt truyện và hàng ngàn từ khóa liên tưởng ngữ nghĩa.

```
[User Input] 
      │
      ▼
[NLP Processor: Intent, NER, Emotion & Slot Filling]
      │ (Intent, Entities, Proactive Question)
      ▼
[Knowledge Base (65 Phim)] ──► [Reasoning Engine: Horn Clauses Filter]
                                              │ (Eligible Candidates)
                                              ▼
[Streamlit UI] ◄── [Top 3 Movies + XAI] ◄── [Recommender: Content-Based IR (TF-IDF) + Rating]
```

### 2. Mô hình biểu diễn Cơ sở tri thức (Knowledge Representation)
Mỗi bộ phim $M$ trong cơ sở tri thức được mô hình hóa dưới dạng một bộ thuộc tính có cấu trúc:
$$M = \langle \text{id, title, genres, age\_rating, duration, rating, cinemas, showtimes, formats, description, keywords} \rangle$$
- **genres**: Tập hợp các thể loại $\subseteq \{\text{Hành động, Kinh dị, Hài kịch, Lãng mạn, Hoạt hình, Viễn tưởng, Trinh thám, Tâm lý, Gia đình, Phiêu lưu}\}$.
- **age_rating**: Nhãn độ tuổi chính thức của Cục Điện ảnh: $\{\text{P (Mọi độ tuổi)}, \text{C13}, \text{C16}, \text{C18}\}$.
- **cinemas**: Danh sách cụm rạp CGV lớn (`CGV Vincom Center`, `CGV Landmark 81`, `CGV Aeon Mall`, `CGV Bà Triệu`, `CGV Crescent Mall`).
- **description**: Đoạn văn tóm tắt cốt truyện chi tiết.
- **keywords**: Kho từ/cụm từ khóa liên tưởng ngữ nghĩa mở rộng (con vật, phương tiện, đồ vật, bối cảnh, tình tiết, cảm xúc, đạo diễn, diễn viên).

---

## PHẦN 3: CÁC KỸ THUẬT AI ÁP DỤNG TRONG HỆ THỐNG

### 1. Xử lý ngôn ngữ tự nhiên (NLP) & Tư vấn ngược lại (Proactive Advising)
- **Tiền xử lý tiếng Việt & Bỏ dấu (`preprocess_text`, `remove_accents`)**: Chuyển chữ thường, chuẩn hóa NFC, loại bỏ dấu câu, chuẩn hóa khoảng trắng thừa và hỗ trợ tìm kiếm không dấu/gõ gần đúng.
- **Nhận diện ý định (Intent Recognition)** & **Phân tích Cảm xúc (`detect_emotion`)**:
  Phân loại câu người dùng vào các nhóm hành vi và trạng thái cảm xúc tự nhiên (buồn, stress, chán đời, thất tình, mệt mỏi...).
- **Tư vấn ngược lại chủ động (`detect_missing_slots`)**:
  Khi câu hỏi của người dùng chung chung hoặc thiếu thông tin quan trọng (ví dụ: *"Nên mua combo nào?"* hay *"Nay buồn quá"*), hệ thống tự động sinh câu hỏi gợi mở thân thiện nhằm thu thập đủ dữ kiện (số người đi, có trẻ em không, rạp mong muốn, khung giờ).

### 2. Cơ chế suy luận logic dựa trên Mệnh đề Horn (Horn Clauses Inference)
Hệ thống sử dụng **suy luận diễn dịch (Deductive Reasoning)** với các ràng buộc cứng (Hard Constraints) dưới dạng luật Horn:

#### a) Luật ràng buộc độ tuổi (Legal Age Restriction):
$$\forall M, \forall \text{UserAge}, \quad \text{UserAge} < \text{AgeLimit}(\text{Rating}(M)) \implies \neg \text{Eligible}(M)$$

#### b) Luật cụm rạp khả dụng (Cinema Availability):
$$\text{RequestedCinemas} \neq \emptyset \land (\text{Cinemas}(M) \cap \text{RequestedCinemas} = \emptyset) \implies \neg \text{Eligible}(M)$$

#### c) Luật khung giờ chiếu (Showtime Feasibility):
$$\forall s \in \text{Showtimes}(M), \quad |s - \text{RequestedTime}| > \Delta t \implies \neg \text{Eligible}(M)$$

### 3. Tìm kiếm Phim qua Liên tưởng Ngữ nghĩa & Kho từ khóa (Semantic Concept IR)
Trên tập các phim ứng viên đã vượt qua bộ lọc suy luận logic, hệ thống tính toán điểm số so khớp **Content-Based Similarity** bằng mô hình TF-IDF + Cosine Similarity:

$$\text{Sim}_{\text{Content}}(Q, M) = \frac{\vec{TF}(Q) \cdot \vec{TF}(Doc_M)}{\|\vec{TF}(Q)\| \|\vec{TF}(Doc_M)\|}$$

Trong đó $Doc_M$ là văn bản hợp nhất bao gồm: $Title + Keywords + Description + Cast + Director + Genres$.
Nếu có từ khóa liên tưởng/cụm từ trùng khớp trực tiếp ($\text{MatchedKeywords} \neq \emptyset$), hệ thống thưởng thêm điểm khớp trọng số trực tiếp để ưu tiên tuyệt đối các bộ phim trùng khớp chi tiết đồ vật, con vật, phương tiện.

$$\text{Score}(M) = w_1 \cdot \text{Sim}_{\text{Genre}}(M) + w_2 \cdot \text{Sim}_{\text{Content}}(M) + w_3 \cdot \frac{\text{Rating}(M)}{10} + w_4 \cdot \text{Match}_{\text{Cinema}}(M)$$

---

## PHẦN 4: BẢNG HỎI - ĐÁP PHẢN BIỆN DÀNH CHO BẢO VỆ ĐỒ ÁN

### Câu 1: Em hãy giải thích Mệnh đề Horn (Horn Clause) được ứng dụng ở đâu trong bài toán này?
> **Trả lời:**  
> Mệnh đề Horn là mệnh đề logic mệnh đề có tối đa một nguyên tử dương ($P_1 \land P_2 \land \dots \land P_n \implies Q$). Nhóm áp dụng Horn Clauses vào module `reasoning_engine.py` để lọc các ràng buộc cứng:  
> 1. Luật độ tuổi: "NẾU người dùng dưới 18 tuổi VÀ phim dán nhãn C18 THÌ loại trừ phim".  
> 2. Luật rạp: "NẾU người dùng chọn cụm rạp R VÀ phim không chiếu tại R THÌ loại trừ phim".  
> Việc dùng Horn Clauses giúp việc suy luận đạt độ phức tạp thời gian tuyến tính $O(N)$, loại bỏ các phương án không hợp lệ trước khi thực hiện tìm kiếm ngữ nghĩa.

### Câu 2: Làm thế nào Chatbot hiểu được các câu hỏi liên tưởng đồ vật, con vật, phương tiện mà người dùng không cần gõ tên phim hay thể loại?
> **Trả lời:**  
> Hệ thống kết hợp 2 kỹ thuật:  
> 1. **Mở rộng vốn từ liên tưởng (Concept Mapping)**: Trong CSDL `movies.json`, mỗi bộ phim được gán hàng chục từ khóa liên tưởng thực tế (ví dụ: *Godzilla x Kong* gán `khủng long, con vượn, khỉ đột, thằn lằn`; *Chàng Nữ Phi Công* gán `máy bay, sân bay, buồng lái`; *Inside Out 2* gán `bộ não, suy nghĩ, đầu óc`; *Transformers* gán `rô bốt, xe hơi biến hình`).  
> 2. **Thuật toán Cosine Similarity + TF-IDF**: So khớp vector tần suất từ giữa câu hỏi tự do của khách và tập từ khóa mở rộng, từ đó đưa ra bộ phim chuẩn xác nhất ở Top 1.

---

## PHẦN 5: KẾT LUẬN

Đề tài đã hoàn thành xuất sắc toàn bộ các mục tiêu đặt ra:
- Xây dựng cơ sở tri thức 65 phim phong phú chứa mô tả và hàng ngàn từ khóa liên tưởng ngữ nghĩa.
- Hoàn thành bộ kiểm thử tự động 8/8 test suites vượt qua xuất sắc 100%.
- Tích hợp thành công tìm kiếm phim theo liên tưởng ngữ nghĩa (Semantic Concept IR với TF-IDF Cosine Match).
- Tích hợp tính năng tư vấn ngược lại chủ động không dập khuôn.
- Giao diện Streamlit VIP Concierge trực quan, thẩm mỹ cao theo chuẩn thương hiệu CGV.
