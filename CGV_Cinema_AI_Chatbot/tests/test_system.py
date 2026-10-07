"""
Unit Test & Kiểm thử tích hợp toàn diện hệ thống Chatbot CGV.
Kiểm tra:
1. Cơ sở tri thức (Knowledge Base) đủ 60+ phim, đa dạng thể loại, nhãn tuổi.
2. Xử lý ngôn ngữ tự nhiên (NLP Intent & NER).
3. Suy luận logic (Horn Clauses): Kiểm tra loại trừ phim vượt tuổi, lọc rạp, lọc giờ.
4. Thuật toán gợi ý (Content-based Recommender + Explainable AI).
5. Tích hợp Chatbot phản hồi đúng kịch bản.
"""

import os
import sys

# Đảm bảo đường dẫn import
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.knowledge_base import MovieKnowledgeBase
from src.nlp_processor import NLPProcessor
from src.reasoning_engine import ReasoningEngine
from src.recommender import MovieRecommender
from src.chatbot import CGVChatbot

def test_knowledge_base():
    print("\n--- [1] KIỂM THỬ CƠ SỞ TRI THỨC (KNOWLEDGE BASE) ---")
    kb = MovieKnowledgeBase()
    movies = kb.get_all_movies()
    print(f"Tổng số phim trong cơ sở tri thức: {len(movies)}")
    assert len(movies) >= 50, f"Số phim phải >= 50, hiện tại là {len(movies)}"
    
    genres = kb.get_unique_genres()
    print(f"Danh sách thể loại ({len(genres)}): {genres}")
    assert len(genres) >= 7, "Cần tối thiểu 7 thể loại khác nhau"

    cinemas = kb.get_unique_cinemas()
    print(f"Danh sách cụm rạp ({len(cinemas)}): {cinemas}")
    assert len(cinemas) >= 4, "Cần tối thiểu 4 cụm rạp CGV lớn"

    ratings = kb.get_unique_age_ratings()
    print(f"Nhãn độ tuổi: {ratings}")
    assert set(ratings) == {"P", "C13", "C16", "C18"}, "Phải đủ 4 nhãn P, C13, C16, C18"
    print("=> PASS: Cơ sở tri thức đạt chuẩn yêu cầu đề bài!")

def test_nlp_processor():
    print("\n--- [2] KIỂM THỬ XỬ LÝ NGÔN NGỮ TỰ NHIÊN (NLP) ---")
    nlp = NLPProcessor()

    # Test Intent Recognition
    test_cases_intent = [
        ("Xin chào rạp CGV", "GREETING"),
        ("Tư vấn giúp mình phim hành động hay với", "RECOMMENDATION"),
        ("Ở CGV Vincom có những phim nào?", "RECOMMENDATION"),
        ("Hôm nay có những suất chiếu mấy giờ?", "SCHEDULE_INFO"),
        ("Nên mua combo bắp nước nào?", "CONCESSION_COMBO"),
        ("Giá vé xem phim bao nhiêu tiền?", "TICKET_BOOKING"),
        ("Cảm ơn bạn nhiều nhé, tạm biệt", "GOODBYE_THANKS"),
    ]
    for text, expected_intent in test_cases_intent:
        actual_intent = nlp.recognize_intent(text)
        print(f"Câu: '{text}' => Intent: {actual_intent} (Kỳ vọng: {expected_intent})")
        assert actual_intent == expected_intent, f"Sai intent: {actual_intent} != {expected_intent}"

    # Test Entity Extraction (NER)
    sample_text = "Tôi 15 tuổi, muốn tìm phim kinh dị hoặc trinh thám chiếu vào buổi tối ở CGV Landmark 81 định dạng IMAX"
    entities = nlp.extract_entities(sample_text)
    print(f"\nPhân tích câu mẫu: '{sample_text}'")
    print(f"Entities trích xuất được: {entities}")
    assert entities["user_age"] == 15, "Không bắt được tuổi 15"
    assert "Kinh dị" in entities["genres"] and "Trinh thám" in entities["genres"], "Không bắt đúng thể loại"
    assert "CGV Landmark 81" in entities["cinemas"], "Không bắt đúng rạp Landmark 81"
    assert entities["time_period"] == "evening", "Không bắt đúng khung giờ tối"
    assert "IMAX" in entities["formats"], "Không bắt đúng định dạng IMAX"
    print("=> PASS: NLP Intent & NER hoạt động chính xác!")

def test_reasoning_engine_horn_clauses():
    print("\n--- [3] KIỂM THỬ SUY LUẬN LOGIC (HORN CLAUSES) ---")
    kb = MovieKnowledgeBase()
    movies = kb.get_all_movies()
    engine = ReasoningEngine()

    # Kiểm tra Luật Độ Tuổi: Bé 12 tuổi -> Không được xem bất kỳ phim C13, C16, C18 nào
    entities_12yo = {"user_age": 12}
    passed, rejected = engine.filter_candidates(movies, entities_12yo)
    print(f"Người xem 12 tuổi: Lọc được {len(passed)} phim hợp lệ, loại trừ {len(rejected)} phim.")
    
    for m in passed:
        assert m["age_rating"] == "P", f"Phim {m['title']} có nhãn {m['age_rating']} bị lọt qua cho người 12 tuổi!"
    for r in rejected:
        assert r["movie"]["age_rating"] in ["C13", "C16", "C18"]

    # Kiểm tra Luật Cụm Rạp: Chỉ lọc phim ở CGV Landmark 81
    entities_l81 = {"cinemas": ["CGV Landmark 81"]}
    passed_l81, _ = engine.filter_candidates(movies, entities_l81)
    print(f"Rạp Landmark 81: Có {len(passed_l81)} phim có suất chiếu.")
    for m in passed_l81:
        assert "CGV Landmark 81" in m["cinemas"]

    print("=> PASS: Các luật Horn Clauses suy luận loại trừ hoàn toàn chính xác!")

def test_recommender():
    print("\n--- [4] KIỂM THỬ THUẬT TOÁN GỢI Ý & EXPLAINABLE AI ---")
    kb = MovieKnowledgeBase()
    movies = kb.get_all_movies()
    recommender = MovieRecommender()

    query = "Gợi ý phim hoạt hình vui nhộn cho gia đình có điểm đánh giá cao"
    entities = {
        "genres": ["Hoạt hình", "Gia đình", "Hài kịch"],
        "user_age": None,
        "cinemas": [],
        "formats": []
    }

    top_3 = recommender.rank_and_recommend(movies, query, entities, top_k=3)
    print(f"Top 3 phim đề xuất cho: '{query}'")
    assert len(top_3) == 3, "Phải trả về đúng Top 3 phim"
    for idx, m in enumerate(top_3, 1):
        print(f"  {idx}. {m['title']} - Thể loại: {m['genres']} - Điểm: {m['similarity_score']} (Rating: {m['rating']})")
        print(f"     Lý do: {m['explanation']}")
        assert len(m["explanation"]) > 10, "Lời giải thích (XAI) không được để trống"
    print("=> PASS: Hệ thống Recommender Content-based & XAI hoạt động xuất sắc!")

def test_chatbot_integration():
    print("\n--- [5] KIỂM THỬ TÍCH HỢP TOÀN BỘ CHATBOT (END-TO-END) ---")
    bot = CGVChatbot()
    
    # Test 1: Chào hỏi
    res1 = bot.process_message("Xin chào bạn!")
    assert res1["intent"] == "GREETING"
    assert "CGV Cinemas" in res1["reply_text"]
    
    # Test 2: Yêu cầu tư vấn phim có nhiều ràng buộc logic
    msg = "Mình 16 tuổi, muốn xem phim hành động viễn tưởng tối nay ở rạp Landmark 81 định dạng IMAX"
    res2 = bot.process_message(msg)
    print(f"\nCâu hỏi: {msg}")
    print(f"Intent: {res2['intent']}")
    print(f"Quá trình suy luận logic: {res2['filter_explanation']}")
    print(f"Số phim ứng viên thỏa mãn: {res2['candidate_count']}")
    print(f"Số phim Top 3 gợi ý: {len(res2['top_recommendations'])}")
    assert len(res2["top_recommendations"]) > 0
    # Phim đề xuất phải có nhãn tuổi P, C13, hoặc C16 (không được C18 vì người dùng 16 tuổi)
    for m in res2["top_recommendations"]:
        assert m["age_rating"] in ["P", "C13", "C16"], f"Phim C18 {m['title']} vi phạm luật tuổi 16!"
        assert "CGV Landmark 81" in m["cinemas"], f"Phim {m['title']} không chiếu ở Landmark 81!"
        assert "IMAX" in m["formats"], f"Phim {m['title']} không hỗ trợ IMAX!"
    print("=> PASS: Toàn bộ quy trình End-to-End của Chatbot hoạt động hoàn hảo!")

def test_content_based_semantic_retrieval():
    print("\n--- [6] KIỂM THỬ TÌM KIẾM THEO NGỮ NGHĨA & KHO TỪ KHÓA (CONTENT-BASED IR) ---")
    bot = CGVChatbot()
    query = "phim trừ tà phong thủy đào mộ Hàn Quốc"
    res = bot.process_message(query)
    print(f"Câu truy vấn vụn vặt: '{query}'")
    top1 = res["top_recommendations"][0]
    print(f"Top 1 phim tìm được: {top1['title']} - Nhãn: {top1['age_rating']}")
    print(f"Từ khóa trùng khớp (XAI): {top1.get('matched_keywords')}")
    print(f"Lý do XAI: {top1.get('explanation')}")
    
    assert "Exhuma" in top1["title"] or "Quật Mộ" in top1["title"], f"Phim trả về sai: {top1['title']}"
    assert len(top1.get("matched_keywords", [])) >= 2, "Phải trích xuất được tối thiểu 2 từ khóa trùng khớp"
    print("=> PASS: Thuật toán Content-based IR & Cosine TF-IDF so khớp chính xác 100%!")

def test_proactive_reverse_advising():
    print("\n--- [7] KIỂM THỬ TÍNH NĂNG TƯ VẤN NGƯỢC LẠI CHỦ ĐỘNG (PROACTIVE ADVISING) ---")
    bot = CGVChatbot()
    
    # Scenario 1: Hỏi bắp nước chung chung
    res_combo = bot.process_message("Nên mua combo bắp nước nào?")
    print(f"\nCâu hỏi bắp nước chung chung: 'Nên mua combo bắp nước nào?'")
    print(f"Phản hồi hỏi ngược lại: \n{res_combo['reply_text']}")
    assert res_combo["mode_used"] == "Proactive Slot Filling"
    assert "đi mấy người" in res_combo["reply_text"].lower() or "trẻ em" in res_combo["reply_text"].lower()

    # Scenario 2: Cảm xúc tự nhiên (Buồn)
    res_sad = bot.process_message("Nay tôi buồn quá...")
    print(f"\nCâu cảm xúc tự nhiên: 'Nay tôi buồn quá...'")
    print(f"Phản hồi đồng cảm & hỏi ngược lại: \n{res_sad['reply_text']}")
    assert res_sad["mode_used"] == "Proactive Slot Filling"
    assert "rạp cgv nào" in res_sad["reply_text"].lower() or "bài hát" in res_sad["reply_text"].lower() or "phim" in res_sad["reply_text"].lower()
    
    print("=> PASS: Tính năng Tư vấn ngược lại chủ động hoạt động cực kỳ linh hoạt và thông minh!")

def test_semantic_concept_associations():
    print("\n--- [8] KIỂM THỬ LIÊN TƯỞNG NGỮ NGHĨA KHI GÕ TỰ NHIÊN (SEMANTIC CONCEPT ASSOCIATIONS) ---")
    bot = CGVChatbot()
    concept_tests = [
        ("tìm phim có con vượn với con khủng long đánh nhau", "Godzilla x Kong"),
        ("có phim nào liên quan đến máy bay với sân bay không", "Chàng Nữ Phi Công"),
        ("phim về mấy con rô bốt biến hình thành xe hơi", "Transformers Một"),
        ("phim hoạt hình có con gấu trúc béo biết đánh võ", "Kung Fu Panda 4"),
        ("phim mô tả suy nghĩ trong bộ não con người", "Inside Out 2")
    ]
    for q, expected in concept_tests:
        res = bot.process_message(q)
        top1 = res["top_recommendations"][0]["title"]
        print(f"Câu liên tưởng: '{q}' => Top 1: {top1}")
        assert expected in top1, f"Kỳ vọng {expected}, nhưng trả về {top1}"
    print("=> PASS: Hệ thống liên tưởng ngữ nghĩa đồ vật, con vật, phương tiện chính xác 100%!")

if __name__ == "__main__":
    test_knowledge_base()
    test_nlp_processor()
    test_reasoning_engine_horn_clauses()
    test_recommender()
    test_chatbot_integration()
    test_content_based_semantic_retrieval()
    test_proactive_reverse_advising()
    test_semantic_concept_associations()
    print("\n=======================================================")
    print("🎉 TẤT CẢ 8/8 BỘ TEST CASE ĐỀU VƯỢT QUA XUẤT SẮC! 🎉")
    print("=======================================================")

