"""
Module Bộ Não Chatbot (Chatbot Engine).
Điều phối hoạt động giữa các module:
1. Nhận tin nhắn của người dùng.
2. Gọi NLP Processor để nhận diện ý định (Intent) và thực thể (NER).
3. Sử dụng Reasoning Engine (Horn Clauses) lọc ứng viên hợp lệ.
4. Gọi Movie Recommender xếp hạng Top 3 và sinh lời giải thích (XAI).
5. Sinh câu phản hồi tự nhiên qua Chế độ kép (Dual-Mode):
   - Chế độ 1 (Mặc định): Bộ sinh đối thoại tự nhiên nội bộ (Offline Natural NLG) - xưng hô thân thiện, có duyên.
   - Chế độ 2 (Tùy chọn): Tích hợp Mô hình Ngôn ngữ Lớn (Gemini LLM) khi người dùng/giảng viên cung cấp API Key.
"""

from typing import Dict, Any, List, Optional
from .knowledge_base import MovieKnowledgeBase
from .nlp_processor import NLPProcessor
from .reasoning_engine import ReasoningEngine
from .recommender import MovieRecommender

class CGVChatbot:
    """Chatbot tư vấn chọn phim thông minh cho rạp CGV."""

    def __init__(self, data_path: str = None):
        self.kb = MovieKnowledgeBase(data_path)
        self.nlp = NLPProcessor()
        self.reasoning = ReasoningEngine()
        self.recommender = MovieRecommender()

    def _generate_natural_offline_response(
        self,
        user_message: str,
        entities: Dict[str, Any],
        candidate_count: int,
        top_recs: List[Dict[str, Any]]
    ) -> str:
        """
        Bộ sinh lời thoại tự nhiên nội bộ (Natural NLG):
        Trả lời trực diện, tự nhiên, đi thẳng vào vấn đề theo phong cách người thật tư vấn:
        Ví dụ: "Dạ theo như yêu cầu của bạn thì đó chính là bộ phim Godzilla x Kong: Đế Chế Mới đúng không nhỉ? 🎬"
        """
        if not top_recs:
            return "Dạ hiện tại mình chưa tìm thấy phim phù hợp với từ khóa này. Bạn thử tìm từ khóa khác giúp mình nhé!"

        lines = []
        u_age = entities.get("user_age")
        top1 = top_recs[0]
        top1_title = top1.get("title")
        matched_kws = top1.get("matched_keywords", [])

        # 1. Câu mở đầu trực diện, tự nhiên vào thẳng vấn đề
        if matched_kws:
            kw_str = ", ".join([f"'{k}'" for k in matched_kws])
            lines.append(f"Dạ theo như yêu cầu tìm {kw_str} của bạn thì đó chính là bộ phim **{top1_title}** đúng không nhỉ? 🎬🍿")
        else:
            lines.append(f"Dạ theo như yêu cầu của bạn thì bộ phim phù hợp nhất chính là **{top1_title}** nè! 🎬🍿")

        # 2. Nhắc nhở quy định độ tuổi (ngắn gọn nếu có)
        if u_age is not None:
            age_rating = top1.get("age_rating", "P")
            if u_age < 18 and age_rating in ["C13", "C16"]:
                lines.append(f"*(Phim dán nhãn **{age_rating}**, phù hợp cho độ tuổi {u_age} của bạn nhé)*")

        lines.append("")

        # 3. Thông tin chi tiết phim Top 1
        m_genres = ", ".join(top1.get("genres", []))
        matched_st = ", ".join(top1.get("matched_showtimes", top1.get("showtimes", []))[:3])
        exp = top1.get("explanation", "")
        rating = top1.get("rating")
        age = top1.get("age_rating")

        lines.append(f"🥇 **{top1_title}** (Nhãn: `{age}` | ⭐ **{rating}/10**)")
        lines.append(f"   • *Thể loại*: {m_genres}")
        lines.append(f"   • *Suất chiếu gợi ý*: **{matched_st}**")
        lines.append(f"   • *Điểm đặc sắc*: {exp}")

        # 4. Các phim gợi ý thêm (Top 2, Top 3)
        if len(top_recs) > 1:
            lines.append("\n💡 **Ngoài ra, bạn cũng có thể tham khảo thêm:**")
            badge_emojis = ["🥈", "🥉"]
            for idx, movie in enumerate(top_recs[1:], 0):
                icon = badge_emojis[idx] if idx < len(badge_emojis) else "✨"
                t = movie.get("title")
                a = movie.get("age_rating")
                r = movie.get("rating")
                g = ", ".join(movie.get("genres", []))
                lines.append(f"{icon} **{t}** (Nhãn: `{a}` | ⭐ **{r}/10**) - *{g}*")

        # 5. Hỏi từng câu tự nhiên, từng bước một (Không tiêu đề rườm rà, không danh sách 1-2-3)
        lines.append("")
        if entities.get("group_size") is None and not entities.get("has_children"):
            lines.append("Hôm nay bạn đi 1 mình hay đi mấy người nhỉ, trong đoàn có trẻ con đi cùng không ạ?")
            if age in ["C13", "C16", "C18"]:
                age_num = 13 if age == "C13" else (16 if age == "C16" else 18)
                lines.append(f"*(Bộ phim **{top1_title}** này dán nhãn **{age}** dành cho người từ {age_num} tuổi trở lên đấy ạ. Nếu bạn đi cùng bé nhỏ thì bảo mình để mình giới thiệu bạn sang các phim Hoạt hình / Gia đình nhãn **P** cho bé nhé!)*")
        elif not entities.get("cinemas"):
            lines.append("Bạn muốn xem ở cụm rạp CGV nào (như Landmark 81, Vincom Center, Bà Triệu, Aeon Mall...) nè?")
        elif not entities.get("time_period"):
            lines.append("Bạn định xem vào suất chiếu buổi sáng, chiều hay tối tiện hơn nè?")
        else:
            lines.append("Bạn xem qua thử bộ phim nào hợp gu nhất bảo mình để mình hỗ trợ giữ chỗ nhé ❤️!")

        return "\n".join(lines)

    def _generate_llm_response(
        self,
        user_message: str,
        entities: Dict[str, Any],
        candidate_count: int,
        top_recs: List[Dict[str, Any]],
        api_key: str
    ) -> Optional[str]:
        """
        Gọi mô hình Gemini LLM để sinh lời thoại siêu tự nhiên (nếu có API Key).
        """
        try:
            from google import genai
            client = genai.Client(api_key=api_key)

            movies_summary = []
            for i, m in enumerate(top_recs, 1):
                st_list = ", ".join(m.get("matched_showtimes", m.get("showtimes", []))[:3])
                movies_summary.append(
                    f"{i}. Tên: {m.get('title')} | Nhãn: {m.get('age_rating')} | Rating: {m.get('rating')}/10 | "
                    f"Thể loại: {', '.join(m.get('genres', []))} | Suất chiếu: {st_list} | Lý do: {m.get('explanation')}"
                )

            prompt = (
                "Bạn là chuyên viên tư vấn phim thân thiện, nhiệt tình và am hiểu điện ảnh của cụm rạp CGV Cinemas.\n"
                f"Khách hàng vừa hỏi: '{user_message}'\n\n"
                f"Hệ thống suy luận logic đã phân tích và lọc được:\n"
                f"- Độ tuổi khách: {entities.get('user_age')} (Đã loại trừ toàn bộ phim vượt lứa tuổi theo quy định Cục Điện ảnh)\n"
                f"- Rạp yêu cầu: {entities.get('cinemas')}\n"
                f"- Danh sách Top phim được thuật toán xếp hạng cao nhất:\n"
                + "\n".join(movies_summary)
                + "\n\nHãy viết một câu trả lời thật tự nhiên, duyên dáng, xưng hô 'mình - bạn', giải thích tại sao phim phù hợp với lứa tuổi và sở thích của khách, nêu rõ tên phim, nhãn tuổi, điểm rating và suất chiếu gợi ý. Thêm emoji sinh động phong cách rạp chiếu phim."
            )

            # Danh sách model theo thứ tự ưu tiên
            candidate_models = ['gemini-3.6-flash', 'gemini-2.5-flash']
            
            for mod in candidate_models:
                try:
                    response = client.models.generate_content(
                        model=mod,
                        contents=prompt
                    )
                    if response and response.text:
                        return response.text.strip()
                except Exception as ex:
                    print(f"[Model {mod} notice] {ex}")
                    continue
            return None
        except Exception as e:
            # Nếu có lỗi mạng hoặc API Key không hợp lệ, tự động fallback
            print(f"[LLM Fallback Notice] {e}")
            return None

    def process_message(self, user_message: str, api_key: str = None, last_intent: str = None) -> Dict[str, Any]:
        """
        Xử lý toàn diện một lượt tin nhắn từ người dùng.
        Hỗ trợ Chế độ kép (Dual-Mode): Offline Natural NLG hoặc Gemini LLM.
        Hỗ trợ nhớ ngữ cảnh hội thoại đa lượt (Multi-turn Context).
        """
        all_movies = self.kb.get_all_movies()
        intent = self.nlp.recognize_intent(user_message)
        entities = self.nlp.extract_entities(user_message)

        # Xử lý ngữ cảnh nối dài (Context Awareness):
        # Nếu câu trước bot vừa hỏi tư vấn combo bắp nước và người dùng đang trả lời số người/trẻ con
        if last_intent == "CONCESSION_COMBO":
            if (entities.get("group_size") is not None or entities.get("has_children") or entities.get("popcorn_flavor")) and not entities.get("genres"):
                intent = "CONCESSION_COMBO"

        response = {
            "intent": intent,
            "entities": entities,
            "reply_text": "",
            "top_recommendations": [],
            "candidate_count": len(all_movies),
            "total_movies": len(all_movies),
            "filter_explanation": "",
            "mode_used": "Offline Natural NLG"
        }

        # 0. Kiểm tra tính năng Tư vấn ngược lại (Proactive Slot Filling)
        # Nếu chưa có ngữ cảnh trước đó, kiểm tra xem câu hỏi có thiếu dữ kiện (bắp nước, cảm xúc, tư vấn chung) để hỏi lại
        if not last_intent or last_intent == "GREETING":
            missing_slot = self.nlp.detect_missing_slots(user_message, intent, entities)
            if missing_slot:
                if missing_slot.get("suggested_genres"):
                    for sg in missing_slot["suggested_genres"]:
                        if sg not in entities["genres"]:
                            entities["genres"].append(sg)
                response["reply_text"] = missing_slot["question"]
                response["mode_used"] = "Proactive Slot Filling"
                return response

        # 1. Xử lý Chào hỏi (Greeting)
        if intent == "GREETING":
            response["reply_text"] = (
                "👋 **Dạ chào bạn nha!** Rất vui được đón tiếp bạn tại **CGV Cinemas** 🍿.\n\n"
                "Hôm nay bạn đang có tâm trạng muốn xem thể loại phim gì nào?\n"
                "• 🎬 **Hành động bùng nổ, Viễn tưởng mãn nhãn**\n"
                "• 👻 **Kinh dị rùng rợn giật gân**\n"
                "• 😂 **Hài hước vui nhộn hay Hoạt hình gia đình**\n"
                "• 💕 **Tình cảm lãng mạn sâu lắng**\n\n"
                "*Cứ nhắn cho mình thể loại yêu thích, độ tuổi hoặc rạp CGV gần bạn nhất, mình sẽ tư vấn bộ phim ưng ý nhất ngay nhé!*"
            )
            return response

        # 2. Xử lý Cảm ơn / Tạm biệt (Goodbye / Thanks)
        if intent == "GOODBYE_THANKS":
            response["reply_text"] = (
                "❤️ **Dạ không có chi, rất vui được hỗ trợ bạn nè!**\n\n"
                "Chúc bạn có những phút giây giải trí thật bùng nổ và cảm xúc tại **CGV Cinemas** 🍿🎬.\n"
                "Khi nào cần săn phim hay suất chiếu đẹp cứ nhắn lại cho mình nhé!"
            )
            return response

        # 3. Xử lý Hỏi rạp chiếu (Cinema Info)
        if intent == "CINEMA_INFO":
            cinemas = self.kb.get_unique_cinemas()
            cinemas_str = "\n".join([f"• 📍 **{c}**" for c in cinemas])
            response["reply_text"] = (
                f"Dạ hiện tại hệ thống CGV đang mở cửa đón khách tại các cụm rạp tiêu biểu sau:\n\n{cinemas_str}\n\n"
                "👉 Bạn dự định ghé rạp nào trong số này để mình tra cứu các phim đang chiếu tại đó cho bạn nha!"
            )
            return response

        # 4. Xử lý Hỏi lịch chiếu (Schedule Info)
        if intent == "SCHEDULE_INFO" and not entities["genres"] and not entities["user_age"]:
            response["reply_text"] = (
                "⏰ Các rạp CGV có suất chiếu liên tục từ **09:00 sáng đến 23:50 đêm** hàng ngày.\n"
                "Để mình lọc lịch chiếu chuẩn xác nhất, bạn thích xem vào buổi nào (sáng, chiều hay tối) và ở cụm rạp nào thế nè?"
            )
            return response

        # 5. Xử lý Hỏi về Bắp nước / Combo Concessions (CONCESSION_COMBO)
        if intent == "CONCESSION_COMBO":
            g_size = entities.get("group_size")
            has_kids = entities.get("has_children", False)
            flavors = entities.get("popcorn_flavor", [])
            flavor_str = f" với vị bắp **{', '.join(flavors)}**" if flavors else ""

            offline_combo_text = ""
            if has_kids or (g_size is not None and g_size >= 3):
                offline_combo_text = (
                    f"👨‍👩‍👧‍👦 **Gợi ý Combo hoàn hảo cho gia đình/nhóm có bé nhỏ{flavor_str}**:\n\n"
                    "🍿 **Family / Kid Combo (Khoảng 179.000đ - 199.000đ)**:\n"
                    "• 2 Bắp lớn 69oz (thoải mái chọn vị Ngọt, Phô mai hoặc Caramel thơm lừng)\n"
                    "• 2 Nước ngọt lớn 32oz cho người lớn\n"
                    "• 1 Nước trái cây / sữa / snack giòn rụm cho bé\n"
                    "• 🎁 **Đặc biệt**: Tặng kèm 01 Ly hoạt hình hoặc Đồ chơi nhân vật độc quyền của CGV (Inside Out / Minions / Kungfu Panda) mà các bé siêu mê!\n\n"
                    "👉 Bạn có thể đặt mua combo trực tiếp tại quầy Concession khi đến rạp hoặc mua kèm vé nhé!"
                )
            elif g_size == 1:
                offline_combo_text = (
                    f"🥤 **Gợi ý Combo lý tưởng cho bạn đi xem phim 1 mình{flavor_str}**:\n\n"
                    "🍿 **My Combo (Khoảng 89.000đ - 99.000đ)**:\n"
                    "• 1 Bắp lớn 69oz (bạn có thể chọn vị Ngọt bơ giòn rụm, Phô mai lắc hoặc Caramel béo ngậy)\n"
                    "• 1 Nước ngọt lớn 32oz mát lạnh (Coke / Sprite / Fanta)\n\n"
                    "💡 *Mẹo nhỏ*: Khẩu phần này vừa vặn no nê cho 1 người nhâm nhi suốt 2 tiếng xem phim mà không sợ dư thừa lãng phí nè!"
                )
            elif g_size == 2:
                offline_combo_text = (
                    f"👫 **Gợi ý Combo 'quốc dân' cho 2 người (Cặp đôi / Bạn thân){flavor_str}**:\n\n"
                    "🍿 **CGV Combo (Khoảng 119.000đ - 129.000đ)** - *Lựa chọn bán chạy nhất tại rạp*:\n"
                    "• 1 Bắp lớn 69oz với **khay chia 2 ngăn** (cho phép mix 2 vị cùng lúc, ví dụ: 1 bên Phô mai đậm đà, 1 bên Caramel ngọt lịm chiều lòng cả 2 người!)\n"
                    "• 2 Nước ngọt lớn 32oz tự chọn\n\n"
                    "💡 *Mẹo tiết kiệm*: Nếu muốn trải nghiệm sang xịn hơn, bạn có thể bù thêm 20k để đổi sang bắp Truffle hoặc ly nước phiên bản giới hạn nha!"
                )
            else:
                # Người dùng hỏi chung chung: "nên mua combo nào"
                offline_combo_text = (
                    "🍿 **Dạ tại rạp CGV hiện có 3 Combo bắp nước 'chân ái' được ưa chuộng nhất tùy theo nhu cầu của bạn nè**:\n\n"
                    "🥤 **1. My Combo (~89k - 99k)**: Dành cho **1 người** (gồm 1 Bắp lớn 69oz + 1 Nước ngọt lớn 32oz, vừa đủ no nê trọn buổi chiếu).\n"
                    "👫 **2. CGV Combo (~119k - 129k)**: Dành cho **2 người / Cặp đôi** (gồm 1 Bắp lớn 2 ngăn mix 2 vị Caramel/Phô mai + 2 Nước ngọt lớn - *Combo bán chạy nhất rạp*).\n"
                    "👨‍👩‍👧‍👦 **3. Family / Kid Combo (~179k - 199k)**: Dành cho **Gia đình / Nhóm có trẻ nhỏ** (gồm 2 Bắp lớn + 2 Nước lớn + 1 Nước trái cây/sữa cho bé + 🎁 Tặng kèm ly/đồ chơi hoạt hình CGV độc quyền).\n\n"
                    "---\n"
                    "💬 **Để mình tư vấn chính xác và tiết kiệm nhất cho bạn nhé:**\n"
                    "👉 Hôm nay bạn **đi mấy người** (đi 1 mình, đi 2 người hay đi nhóm đông)?\n"
                    "👉 Nhóm mình **có em bé / trẻ con đi cùng không**?\n"
                    "👉 Và bạn đang thèm bắp vị gì nè: **Ngọt truyền thống, Phô mai mằn mặn hay Caramel ngọt béo**?"
                )

            # Nếu có API key, gọi LLM để đối thoại sinh động hơn
            if api_key and api_key.strip():
                llm_prompt = (
                    f"Khách hàng vừa hỏi về bắp nước / combo CGV: '{user_message}'\n"
                    f"Dưới đây là bảng giá combo bắp nước chuẩn của CGV:\n{offline_combo_text}\n"
                    "Hãy đóng vai chuyên viên tư vấn quầy Concession CGV thân thiện, duyên dáng, trả lời tự nhiên, xưng hô 'mình - bạn'. "
                    "Nếu khách chưa nói rõ đi mấy người hoặc có trẻ con không, hãy nhiệt tình hỏi lại xem khách đi mấy người, có trẻ em không để tư vấn combo chuẩn nhất!"
                )
                try:
                    from google import genai
                    client = genai.Client(api_key=api_key.strip())
                    for mod in ['gemini-3.6-flash', 'gemini-2.5-flash']:
                        try:
                            res_gen = client.models.generate_content(model=mod, contents=llm_prompt)
                            if res_gen and res_gen.text:
                                response["reply_text"] = res_gen.text.strip()
                                response["mode_used"] = "Gemini LLM Hybrid"
                                return response
                        except Exception:
                            continue
                except Exception:
                    pass

            response["reply_text"] = offline_combo_text
            return response

        # 6. Xử lý Hỏi về Đặt vé / Giá vé (TICKET_BOOKING)
        if intent == "TICKET_BOOKING":
            ticket_text = (
                "🎟️ **Bảng Giá Vé & Hướng Dẫn Đặt Vé Tại Rạp CGV**:\n\n"
                "💰 **1. Các mức giá vé tiêu chuẩn**:\n"
                "• **Vé 2D Tiêu chuẩn**: 80.000đ - 95.000đ (Thứ 2 - Thứ 5) | 105.000đ - 125.000đ (Cuối tuần & Ngày lễ).\n"
                "• **Ưu đãi Học sinh - Sinh viên & U22**: Đồng giá **55.000đ - 65.000đ** cho thành viên U22 (áp dụng cả tuần khi xuất trình thẻ HSSV/CCCD) 🎉.\n"
                "• **Vé Trẻ em (dưới 1.3m / dưới 12 tuổi)**: Giảm từ 20% - 30% so với giá vé người lớn.\n"
                "• **Định dạng cao cấp**: 3D (từ 130.000đ) | IMAX Laser (từ 160.000đ - 220.000đ).\n\n"
                "📲 **2. Cách đặt vé nhanh nhất**:\n"
                "• Bạn có thể chọn trực tiếp các **nút suất chiếu màu đỏ** ở Bảng đề xuất bên phải để hệ thống ghi nhận đặt vé mô phỏng.\n"
                "• Hoặc đặt vé trực tuyến qua **Ứng dụng CGV Cinemas / Website cgv.vn** để tích điểm thành viên CGV Member và chọn ghế ngồi đẹp nhất!\n\n"
                "👉 Bạn đang muốn xem phim nào hoặc tại cụm rạp nào để mình kiểm tra giá vé và suất chiếu chính xác cho bạn nha?"
            )
            response["reply_text"] = ticket_text
            return response

        # 7. Xử lý Yêu cầu Tư vấn Phim (RECOMMENDATION)
        # BƯỚC 1: Suy luận logic If-Then (Horn Clauses) lọc ứng viên
        passed_candidates, rejected = self.reasoning.filter_candidates(all_movies, entities)
        response["candidate_count"] = len(passed_candidates)

        # Xây dựng dấu vết suy luận logic (Logic Inference Trace)
        trace_items = []
        if entities["user_age"] is not None:
            trace_items.append(f"Luật độ tuổi: Giữ lại phim hợp lệ cho người {entities['user_age']} tuổi (loại {len(rejected)} phim vi phạm)")
        if entities["cinemas"]:
            trace_items.append(f"Luật rạp: Lọc phim có suất tại {', '.join(entities['cinemas'])}")
        if entities["formats"]:
            trace_items.append(f"Luật định dạng: Lọc phim hỗ trợ {', '.join(entities['formats'])}")
        if entities["time_period"] or entities["specific_hour"] is not None:
            trace_items.append(f"Luật thời gian: Lọc suất chiếu trong khung giờ mong muốn")

        if trace_items:
            response["filter_explanation"] = " ➔ ".join(trace_items) + f" (Còn {len(passed_candidates)}/{len(all_movies)} phim ứng viên)."

        # Nếu không có phim nào thỏa mãn mọi ràng buộc
        if not passed_candidates:
            reason_summary = []
            if entities["user_age"] is not None and entities["user_age"] < 13:
                reason_summary.append(f"độ tuổi dưới 13")
            if entities["cinemas"]:
                reason_summary.append(f"rạp {', '.join(entities['cinemas'])}")
            if entities["time_period"]:
                reason_summary.append(f"khung giờ {entities['time_text']}")

            response["reply_text"] = (
                f"😔 Tiếc quá bạn ơi, hiện tại CGV chưa có bộ phim nào thỏa mãn đồng thời tất cả các điều kiện"
                f"{(' (' + ', '.join(reason_summary) + ')') if reason_summary else ''}.\n\n"
                f"💡 **Mẹo nhỏ từ CGV**: Bạn thử nới lỏng khung giờ xem hoặc chọn thêm cụm rạp CGV khác để mình tìm thêm nhiều phim hay cho bạn nhé!"
            )
            return response

        # BƯỚC 2: Thuật toán xếp hạng đề xuất (Content-Based + Hybrid Ranking)
        top_recs = self.recommender.rank_and_recommend(
            candidate_movies=passed_candidates,
            user_query=user_message,
            entities=entities,
            top_k=3
        )
        response["top_recommendations"] = top_recs

        # BƯỚC 3: Sinh câu phản hồi tự nhiên (Ưu tiên LLM nếu có Key, fallback Natural NLG)
        llm_reply = None
        if api_key and api_key.strip():
            llm_reply = self._generate_llm_response(
                user_message=user_message,
                entities=entities,
                candidate_count=len(passed_candidates),
                top_recs=top_recs,
                api_key=api_key.strip()
            )
            if llm_reply:
                response["mode_used"] = "Gemini LLM Hybrid"

        if not llm_reply:
            response["reply_text"] = self._generate_natural_offline_response(
                user_message=user_message,
                entities=entities,
                candidate_count=len(passed_candidates),
                top_recs=top_recs
            )
            response["mode_used"] = "Offline Natural NLG"
        else:
            response["reply_text"] = llm_reply

        return response
