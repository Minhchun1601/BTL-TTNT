"""
Module Xử Lý Ngôn Ngữ Tự Nhiên (Natural Language Processing - NLP).
Bao gồm:
1. Tiền xử lý văn bản tiếng Việt (Chuẩn hóa ký tự, chữ thường, dấu câu).
2. Phân loại ý định người dùng (Intent Recognition).
3. Trích xuất thực thể (Named Entity Recognition - NER): Thể loại, Cụm rạp, Độ tuổi, Khung giờ, Định dạng.
"""

import re
import unicodedata
from typing import Dict, Any, List, Optional

class NLPProcessor:
    """Lớp xử lý ngôn ngữ tự nhiên cho Chatbot tư vấn phim CGV."""

    # 1. Từ điển quy đổi thể loại (Genre Keywords & Natural Action Phrases)
    GENRE_MAPPING = {
        "Hành động": ["hành động", "hanh dong", "action", "đánh nhau", "danh nhau", "đánh đấm", "danh dam", "bắn nhau", "ban nhau", "rượt đuổi", "ruot duoi", "võ thuật", "vo thuat", "chém giết", "đánh lộn", "đấm nhau", "đao kiếm", "đánh nhau khổng lồ", "sướng mắt", "mãn nhãn", "bom tấn", "cháy nổ", "kịch tính"],
        "Kinh dị": ["kinh dị", "kinh di", "horror", "dọa ma", "doa ma", "ma quỷ", "ma quy", "rùng rợn", "rung ron", "sợ hãi", "ám ảnh", "am anh", "máu me", "mau me", "sợ giật mình", "trừ tà", "quỷ cẩu", "nghĩa địa", "xác chết", "thót tim", "nổi da gà", "tâm linh", "bùa ngải"],
        "Hài kịch": ["hài", "hai", "hài kịch", "hai kich", "comedy", "hài hước", "hai huoc", "cười bể bụng", "cười", "vui nhộn", "vui ve", "buồn cười", "hài bựa", "xả stress", "tinh nghịch", "tấu hài", "cười té ghế", "cười xỉu"],
        "Lãng mạn": ["lãng mạn", "lang man", "tình cảm", "tinh cam", "romance", "romantic", "yêu đương", "yeu duong", "ngọt ngào", "ngot ngao", "thất tình", "tat tinh", "đôi lứa", "gấu", "tình yêu", "hẹn hò", "khóc hết nước mắt", "người yêu", "chữa lành"],
        "Hoạt hình": ["hoạt hình", "hoat hinh", "animation", "anime", "cartoon", "hoạt họa", "hoat hoa", "gấu trúc", "gấu mập", "con vật", "động vật", "rô bốt biến hình", "con ngỗng", "trẻ em", "cho bé", "thiếu nhi"],
        "Khoa học viễn tưởng": ["khoa học viễn tưởng", "khoa hoc vien tuong", "viễn tưởng", "vien tuong", "sci-fi", "scifi", "vũ trụ", "vu tru", "ngoài hành tinh", "rô bốt", "ro bot", "xe biến hình", "lỗ sâu", "hố đen", "không gian", "khổng lồ", "tương lai"],
        "Trinh thám": ["trinh thám", "trinh tham", "thám tử", "tham tu", "phá án", "pha an", "bí ẩn", "bi an", "detective", "mystery", "sát nhân hàng loạt", "giải mã", "thám tử lừng danh", "điều tra", "sát nhân"],
        "Tâm lý": ["tâm lý", "tam ly", "drama", "tình cảm xã hội", "chiều sâu tâm lý", "gia tộc", "bạo lực gia đình", "chữa lành", "trầm cảm", "xúc động", "nghẹn ngào"],
        "Gia đình": ["gia đình", "gia dinh", "family", "cho trẻ em", "cho bé", "cả nhà", "ca nha", "trẻ con", "tình mẫu tử", "bà ngoại", "mẹ già", "tình thân", "ấm áp"],
        "Phiêu lưu": ["phiêu lưu", "phieu luu", "adventure", "khám phá", "kham pha", "hoang đảo", "vượt biển", "sa mạc", "chèo thuyền", "thám hiểm"]
    }

    # 2. Từ điển quy đổi cụm rạp CGV (Cinema Keywords)
    CINEMA_MAPPING = {
        "CGV Vincom Center": ["vincom center", "vincom đồng khởi", "vincom bà triệu", "vincom", "cgv vincom"],
        "CGV Landmark 81": ["landmark 81", "landmark", "l81", "cgv landmark"],
        "CGV Aeon Mall": ["aeon mall", "aeon", "ê-on", "eon mall", "cgv aeon"],
        "CGV Bà Triệu": ["bà triệu", "ba trieu", "cgv bà triệu"],
        "CGV Crescent Mall": ["crescent mall", "crescent", "quận 7", "q7", "cgv crescent"]
    }

    # 3. Định dạng phim (Movie Formats)
    FORMAT_MAPPING = {
        "IMAX": ["imax", "i-max"],
        "3D": ["3d", "3-d"],
        "2D": ["2d", "2-d"]
    }

    # 4. Từ điển cảm xúc & tâm trạng (Emotion & Mood Lexicon)
    EMOTION_LEXICON = {
        "SADNESS": ["buồn", "buon", "tâm trạng tệ", "tam trang te", "chán đời", "sầu", "thất vọng", "đau lòng", "khóc", "thất tình", "mệt mỏi tâm trí"],
        "STRESS": ["stress", "mệt mỏi", "met moi", "áp lực", "ap luc", "sếp mắng", "sep mang", "đi làm mệt", "xả stress", "xa stress", "căng thẳng", "cang thang", "kiệt sức"],
        "BOREDOM": ["chán quá", "chan qua", "rảnh rỗi", "ranh roi", "không biết làm gì", "nhàn rỗi", "nhàm chán", "chán ngắt"],
        "ROMANCE_DESIRE": ["hẹn hò", "hen ho", "người yêu", "nguoi yeu", "bạn gái", "bạn trai", "hẹn hò đôi", "ngọt ngào", "đi với gấu"]
    }

    def __init__(self):
        pass

    @staticmethod
    def preprocess_text(text: str) -> str:
        """
        Hàm tiền xử lý văn bản tiếng Việt nâng cao:
        1. Chuyển chữ thường (lowercase)
        2. Chuẩn hóa mã Unicode NFC
        3. Loại bỏ dấu câu / ký tự đặc biệt
        4. Chuẩn hóa khoảng trắng thừa
        5. Loại bỏ từ dừng (stop-words) cơ bản
        """
        if not text:
            return ""
        text = text.lower().strip()
        text = unicodedata.normalize('NFC', text)
        # Loại bỏ dấu câu
        text = re.sub(r"[^\w\s]", " ", text)
        # Chuẩn hóa khoảng trắng
        text = re.sub(r"\s+", " ", text).strip()
        
        # Từ dừng tiếng Việt cơ bản (không làm mất ý nghĩa quan trọng)
        stop_words = {"và", "của", "là", "thì", "mà", "ở", "với", "cho", "được", "có", "những", "các", "này", "đó", "ra", "vào"}
        tokens = text.split()
        filtered = [t for t in tokens if t not in stop_words]
        return " ".join(filtered)

    @staticmethod
    def remove_accents(text: str) -> str:
        """
        Loại bỏ dấu tiếng Việt để hỗ trợ tìm kiếm không dấu / gõ gần giống (Accent-Insensitive Search):
        Ví dụ: 'Phim tinh cam lang man cua Tran Thanh' -> 'phim tinh cam lang man cua tran thanh'
        """
        if not text:
            return ""
        text = unicodedata.normalize('NFD', text)
        text = re.sub(r'[\u0300-\u036f]', '', text)
        text = text.replace('đ', 'd').replace('Đ', 'D')
        return unicodedata.normalize('NFC', text).lower()

    @staticmethod
    def normalize_text(text: str) -> str:
        """Chuẩn hóa chuỗi văn bản đầu vào: xóa khoảng trắng thừa, hạ chữ thường."""
        if not text:
            return ""
        text = text.lower().strip()
        text = unicodedata.normalize('NFC', text)
        return text

    def detect_emotion(self, text: str) -> Optional[str]:
        """Nhận diện cảm xúc tự nhiên trong câu nói người dùng."""
        norm_text = self.normalize_text(text)
        for emotion, keywords in self.EMOTION_LEXICON.items():
            for kw in keywords:
                if kw in norm_text:
                    return emotion
        return None

    def detect_missing_slots(self, text: str, intent: str, entities: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Phát hiện thiếu dữ kiện quan trọng để thực hiện tính năng 'Tư vấn ngược lại' (Proactive Slot Filling).
        Trả về dictionary chứa kịch bản cần hỏi lại, hoặc None nếu đã đủ dữ kiện.
        """
        norm_text = self.normalize_text(text)
        emotion = self.detect_emotion(text)

        # 1. Kịch bản tư vấn bắp nước: Hỏi đi mấy người & có trẻ em không nếu chưa nêu
        if intent == "CONCESSION_COMBO":
            if entities.get("group_size") is None and not entities.get("has_children"):
                return {
                    "scenario": "CONCESSION_SLOT_FILLING",
                    "question": "Để mình tư vấn Combo bắp nước tiết kiệm và chuẩn nhất cho bạn nhé! 😊\nBạn đi nhóm mấy người và trong đoàn có trẻ em không ạ?"
                }

        # 2. Kịch bản nhận diện cảm xúc tự nhiên (Buồn, stress, chán...)
        if emotion:
            if not entities.get("cinemas") and not entities.get("time_period"):
                if emotion in ["SADNESS", "BOREDOM"]:
                    return {
                        "scenario": "EMOTION_SLOT_FILLING",
                        "emotion": emotion,
                        "suggested_genres": ["Hài kịch", "Hoạt hình", "Gia đình"],
                        "question": "Chia sẻ với bạn nhé! 🥺 Những lúc tâm trạng đang buồn hoặc chán thế này, xem một bộ phim hài sảng khoái hoặc hoạt hình chữa lành là tuyệt vời nhất.\nBạn định xem ở rạp CGV nào (Landmark 81, Vincom Center, Bà Triệu...) và xem suất chiều hay tối để mình lọc phim chuẩn nhất cho bạn?"
                    }
                elif emotion == "STRESS":
                    return {
                        "scenario": "EMOTION_SLOT_FILLING",
                        "emotion": emotion,
                        "suggested_genres": ["Hành động", "Hài kịch"],
                        "question": "Sau một ngày mệt mỏi xả stress thì phim hành động bom tấn hoặc hài hước giải trí là lựa chọn số 1! 💪\nBạn muốn xem ở cụm rạp nào và đi một mình hay đi cùng ai để mình gợi ý nhé?"
                    }

        # 3. Kịch bản gợi ý phim chung chung (Chỉ kích hoạt khi câu truy vấn thực sự rỗng tuếch, không có từ khóa tìm kiếm)
        if intent == "RECOMMENDATION" and not entities.get("genres") and not entities.get("cinemas") and not entities.get("user_age"):
            vague_phrases = ["tư vấn phim", "gợi ý phim", "xem gì", "xem gi", "phim hay", "tìm phim", "tư vấn cho mình", "tư vấn giúp", "gợi ý cho mình", "chọn giúp phim", "tư vấn", "gợi ý"]
            cleaned_text = norm_text
            for vp in vague_phrases:
                cleaned_text = cleaned_text.replace(vp, "")
            cleaned_text = self.preprocess_text(cleaned_text).strip()
            
            # Chỉ coi là vague nếu sau khi lọc từ nhờ tư vấn chung chung mà không còn từ khóa tìm kiếm nào
            if len(cleaned_text) == 0:
                return {
                    "scenario": "VAGUE_RECOMMENDATION_SLOT_FILLING",
                    "question": "Rạp CGV đang có rất nhiều phim hot! 🍿 Để mình tư vấn đúng ý bạn nhất, bạn có thể cho mình biết:\n1. Bạn đi xem một mình, hẹn hò hay đi cùng gia đình/trẻ em?\n2. Bạn muốn xem rạp nào hoặc thích thể loại gì (Hành động, Kinh dị, Hoạt hình, Hài kịch...)?"
                }

        return None

    def recognize_intent(self, text: str) -> str:
        """
        Phân loại ý định của người dùng dựa trên phân tích từ khóa và mẫu câu (Rule-based Intent Matching).
        Các loại ý định chính:
        - GREETING: Chào hỏi ban đầu
        - RECOMMENDATION: Yêu cầu gợi ý/tư vấn phim
        - SCHEDULE_INFO: Hỏi về lịch chiếu, suất chiếu, giờ chiếu
        - CINEMA_INFO: Hỏi về thông tin rạp chiếu CGV
        - GOODBYE_THANKS: Cảm ơn hoặc chào tạm biệt
        - UNKNOWN: Ý định không xác định rõ
        """
        norm_text = self.normalize_text(text)

        # Kiểm tra Chào hỏi
        greeting_patterns = [
            r"^(xin )?chào", r"^hello", r"^hi(\s|$)", r"^alo", r"^hey",
            r"^chào bạn", r"^chào ad", r"^chào bot", r"^xin chào"
        ]
        is_greeting = any(re.search(pattern, norm_text) for pattern in greeting_patterns)
        if is_greeting:
            # Nếu chỉ là chào hỏi (kể cả có nhắc đến tên rạp cgv như "chào rạp cgv")
            # và không có từ khóa yêu cầu tư vấn cụ thể:
            has_action_request = any(kw in norm_text for kw in [
                "tư vấn", "tu van", "gợi ý", "goi y", "chọn giúp", "muốn xem", "tìm phim", "tim phim", "chiếu lúc mấy giờ"
            ])
            if not has_action_request:
                return "GREETING"

        # Kiểm tra Cảm ơn / Tạm biệt
        thanks_patterns = [
            r"cảm ơn", r"cam on", r"thank", r"tks", r"tạm biệt", r"tam biet",
            r"bye", r"hẹn gặp lại", r"ok cảm ơn"
        ]
        for pattern in thanks_patterns:
            if re.search(pattern, norm_text):
                return "GOODBYE_THANKS"

        # Kiểm tra Hỏi về Bắp nước / Combo Concessions
        combo_patterns = [
            r"combo", r"bắp", r"bap", r"bỏng", r"bong", r"popcorn",
            r"nước ngọt", r"nuoc ngot", r"bắp nước", r"bap nuoc",
            r"đồ ăn", r"do an", r"uống gì", r"uong gi", r"ăn gì", r"an gi",
            r"phô mai", r"caramel", r"sweetbox combo", r"my combo"
        ]
        if any(re.search(p, norm_text) for p in combo_patterns):
            return "CONCESSION_COMBO"

        # Kiểm tra Hỏi về Đặt vé / Giá vé / Loại vé
        ticket_patterns = [
            r"đặt vé", r"dat ve", r"giá vé", r"gia ve", r"mua vé", r"mua ve",
            r"bao nhiêu tiền", r"bao nhieu tien", r"vé học sinh", r"u22",
            r"vé sinh viên", r"vé trẻ em", r"ghế vip", r"ghế đôi", r"sweetbox"
        ]
        if any(re.search(p, norm_text) for p in ticket_patterns):
            return "TICKET_BOOKING"

        # Kiểm tra Hỏi thông tin Suất chiếu / Lịch chiếu
        schedule_patterns = [
            r"suất chiếu", r"suat chieu", r"lịch chiếu", r"lich chieu",
            r"mấy giờ", r"may gio", r"chiếu lúc nào", r"khung giờ", r"giờ chiếu"
        ]
        if any(re.search(p, norm_text) for p in schedule_patterns) and not any(kw in norm_text for kw in ["tư vấn", "gợi ý", "chọn giúp", "phim gì hay"]):
            return "SCHEDULE_INFO"

        # Kiểm tra Hỏi thông tin Rạp chiếu
        cinema_patterns = [
            r"rạp nào", r"địa chỉ rạp", r"có rạp nào", r"ở đâu", r"cụm rạp"
        ]
        if any(re.search(p, norm_text) for p in cinema_patterns) and not any(kw in norm_text for kw in ["tư vấn", "gợi ý", "phim"]):
            return "CINEMA_INFO"

        # Kiểm tra Yêu cầu Gợi ý / Tư vấn chọn phim
        recommend_patterns = [
            r"tư vấn", r"tu van", r"gợi ý", r"goi y", r"chọn giúp", r"chon giup",
            r"phim gì hay", r"xem gì", r"xem gi", r"phim nào", r"tìm phim", r"tim phim",
            r"muốn xem", r"muon xem", r"có phim gì", r"co phim gi", r"đề xuất", r"recommend"
        ]
        if any(re.search(p, norm_text) for p in recommend_patterns):
            return "RECOMMENDATION"

        # Nếu có nhắc đến thể loại, giờ, rạp thì mặc định ý định là cần tư vấn
        extracted_entities = self.extract_entities(text)
        if extracted_entities.get("genres") or extracted_entities.get("cinemas") or extracted_entities.get("user_age") or extracted_entities.get("time"):
            return "RECOMMENDATION"

        return "UNKNOWN"

    def extract_entities(self, text: str) -> Dict[str, Any]:
        """
        Trích xuất các thực thể trong câu (Named Entity Recognition - NER):
        - Thể loại phim (genres)
        - Cụm rạp CGV (cinemas)
        - Định dạng phim (formats)
        - Độ tuổi người xem (user_age)
        - Khung giờ / Thời gian mong muốn (time: raw_text, period, specific_hour)
        """
        norm_text = self.normalize_text(text)
        entities: Dict[str, Any] = {
            "genres": [],
            "cinemas": [],
            "formats": [],
            "user_age": None,
            "time_period": None,      # "morning", "afternoon", "evening", "late_night"
            "specific_hour": None,    # số giờ float ví dụ 19.5 (19h30)
            "time_text": None,
            "group_size": None,       # số người đi cùng (1, 2, 3, 4...)
            "has_children": False,    # có trẻ em/bé đi cùng không
            "popcorn_flavor": []      # phô mai, caramel, ngọt...
        }

        # Trích xuất số người (Group size)
        if any(w in norm_text for w in ["1 mình", "mot minh", "đi lẻ", "1 người", "một người", "đơn"]):
            entities["group_size"] = 1
        elif any(w in norm_text for w in ["2 người", "hai người", "cặp đôi", "2 bạn", "hai bạn", "người yêu", "2 đứa"]):
            entities["group_size"] = 2
        else:
            g_match = re.search(r"(\d{1,2})\s*(?:người|bạn|vé|đứa|khách)", norm_text)
            if g_match:
                try:
                    entities["group_size"] = int(g_match.group(1))
                except ValueError:
                    pass

        # Trích xuất có trẻ em không
        if any(w in norm_text for w in ["trẻ em", "trẻ con", "bé", "con nhỏ", "gia đình", "kid"]):
            entities["has_children"] = True

        # Trích xuất vị bắp
        for flav in ["phô mai", "caramel", "ngọt", "mặn", "truffe", "sô cô la"]:
            if flav in norm_text:
                entities["popcorn_flavor"].append(flav)

        # 1. Trích xuất Thể loại (Genres)
        for standard_genre, keywords in self.GENRE_MAPPING.items():
            for kw in keywords:
                pattern = r"\b" + re.escape(kw) + r"\b"
                if re.search(pattern, norm_text):
                    if standard_genre not in entities["genres"]:
                        entities["genres"].append(standard_genre)
                    break

        # 2. Trích xuất Cụm rạp (Cinemas)
        for standard_cinema, keywords in self.CINEMA_MAPPING.items():
            for kw in keywords:
                pattern = r"\b" + re.escape(kw) + r"\b"
                if re.search(pattern, norm_text):
                    if standard_cinema not in entities["cinemas"]:
                        entities["cinemas"].append(standard_cinema)
                    break

        # 3. Trích xuất Định dạng (Formats: 2D, 3D, IMAX)
        for standard_fmt, keywords in self.FORMAT_MAPPING.items():
            for kw in keywords:
                pattern = r"\b" + re.escape(kw) + r"\b"
                if re.search(pattern, norm_text):
                    if standard_fmt not in entities["formats"]:
                        entities["formats"].append(standard_fmt)
                    break

        # 4. Trích xuất Độ tuổi người xem (User Age)
        # Bắt cả 2 dạng: số trước chữ ("16 tuổi", "16t", "16+") VÀ chữ trước số ("tuổi 16", "độ tuổi 16", "u16")
        age_patterns = [
            r"(?:tuổi|lứa tuổi|độ tuổi|u)\s*(\d{1,2})\b",        # "tuổi 16", "độ tuổi 16", "u16"
            r"\b(\d{1,2})\s*(?:tuổi|t\b|\+|yo\b)",              # "16 tuổi", "16t", "16+"
            r"cho\s*(?:người|bạn|bé|trẻ)?\s*(\d{1,2})\s*tuổi"   # "cho người 16 tuổi"
        ]
        detected_age = None
        for ap in age_patterns:
            m = re.search(ap, norm_text)
            if m:
                try:
                    val = int(m.group(1))
                    if 1 <= val <= 100:
                        detected_age = val
                        break
                except (ValueError, IndexError):
                    pass

        if detected_age is not None:
            entities["user_age"] = detected_age
        elif "trẻ em" in norm_text or "cho bé" in norm_text or "học sinh cấp 1" in norm_text or "mầm non" in norm_text:
            entities["user_age"] = 10
        elif "học sinh cấp 2" in norm_text or "13+" in norm_text:
            entities["user_age"] = 14
        elif "học sinh cấp 3" in norm_text or "16+" in norm_text:
            entities["user_age"] = 16
        elif "người lớn" in norm_text or "18+" in norm_text or "u18" not in norm_text and "u20" in norm_text:
            entities["user_age"] = 20

        # 5. Trích xuất Thời gian (Time / Showtimes)
        # Giờ cụ thể: ví dụ "18h30", "18:30", "20h", "20 giờ", "8h tối"
        hour_match = re.search(r"(\d{1,2})([h:](\d{2})?)(\s*(sáng|chiều|tối|đêm))?", norm_text)
        if hour_match:
            h = int(hour_match.group(1))
            m = int(hour_match.group(3)) if hour_match.group(3) else 0
            period_mod = hour_match.group(5)
            if period_mod in ["tối", "chiều"] and h < 12:
                h += 12
            entities["specific_hour"] = h + (m / 60.0)
            entities["time_text"] = hour_match.group(0)

        # Khung giờ tổng quát (General Periods)
        if "sáng" in norm_text:
            entities["time_period"] = "morning"     # < 12:00
            if not entities["time_text"]:
                entities["time_text"] = "buổi sáng"
        elif "trưa" in norm_text or "chiều" in norm_text:
            entities["time_period"] = "afternoon"   # 12:00 - 17:59
            if not entities["time_text"]:
                entities["time_text"] = "buổi chiều"
        elif "tối" in norm_text:
            entities["time_period"] = "evening"     # 18:00 - 21:59
            if not entities["time_text"]:
                entities["time_text"] = "buổi tối"
        elif "đêm" in norm_text or "khuya" in norm_text:
            entities["time_period"] = "late_night"  # >= 22:00
            if not entities["time_text"]:
                entities["time_text"] = "đêm muộn"

        return entities
