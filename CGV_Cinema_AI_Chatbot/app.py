"""
Ứng dụng Web Chatbot Tư Vấn Chọn Phim Cho Rạp CGV (Streamlit Application).
Bài tập lớn môn Trí tuệ Nhân tạo (AI Project).
Tích hợp: Hiệu ứng Streaming Text (gõ chữ mượt mà), Giao diện Cinema Glassmorphism và Tương tác giả lập đặt vé.
"""

import streamlit as st
import os
import sys
import time
import re

# Đảm bảo đường dẫn import
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.chatbot import CGVChatbot
from src.knowledge_base import MovieKnowledgeBase

# --- CẤU HÌNH TRANG STREAMLIT ---
st.set_page_config(
    page_title="CGV Cinema AI - Trợ Lý Tư Vấn Phim Thông Minh",
    page_icon="🍿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS CAO CẤP PHONG CÁCH RẠP CHIẾU PHIM CGV ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Montserrat', sans-serif;
    }
    
    /* Header phong cách Cinema Hiện Đại */
    .cgv-hero-header {
        background: linear-gradient(135deg, #8B0000 0%, #E50914 45%, #B20710 80%, #400000 100%);
        padding: 22px 30px;
        border-radius: 18px;
        color: white;
        margin-bottom: 22px;
        box-shadow: 0 12px 30px rgba(229, 9, 20, 0.35);
        display: flex;
        align-items: center;
        justify-content: space-between;
        border: 1px solid rgba(255, 255, 255, 0.15);
    }
    .cgv-hero-header h1 {
        font-size: 26px;
        font-weight: 800;
        margin: 0;
        letter-spacing: 0.8px;
        text-shadow: 0 2px 4px rgba(0,0,0,0.4);
    }
    .cgv-hero-header p {
        margin: 6px 0 0 0;
        font-size: 13.5px;
        opacity: 0.92;
    }
    
    /* Badge độ tuổi */
    .badge-age {
        display: inline-block;
        padding: 3px 9px;
        border-radius: 6px;
        font-weight: 800;
        font-size: 11.5px;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    .badge-age-P { background: #28a745; color: white; }
    .badge-age-C13 { background: #007bff; color: white; }
    .badge-age-C16 { background: #fd7e14; color: white; }
    .badge-age-C18 { background: #dc3545; color: white; }
    
    /* Movie Card phong cách Glassmorphism Cinema */
    .movie-card-container {
        background: #18181f;
        border-radius: 16px;
        border: 1px solid #2d2d38;
        overflow: hidden;
        margin-bottom: 18px;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.45);
        transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease;
    }
    .movie-card-container:hover {
        transform: translateY(-3px);
        border-color: #E71A0F;
        box-shadow: 0 12px 28px rgba(229, 9, 20, 0.3);
    }
    .movie-card-body {
        padding: 16px;
    }
    .movie-card-title {
        font-size: 17px;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 6px;
        line-height: 1.3;
    }
    .movie-meta-row {
        font-size: 12.5px;
        color: #a5a5b5;
        margin-bottom: 8px;
    }
    .movie-explanation-box {
        background: rgba(229, 9, 20, 0.08);
        border-left: 3px solid #E71A0F;
        padding: 10px 12px;
        border-radius: 0 8px 8px 0;
        font-size: 12.5px;
        color: #dedede;
        line-height: 1.45;
        margin-top: 10px;
    }
    
    /* Chip suất chiếu & định dạng */
    .chip-format {
        display: inline-block;
        background: #282835;
        color: #e0e0e0;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 11.5px;
        font-weight: 600;
        margin-right: 4px;
        border: 1px solid #3c3c4d;
    }
    .chip-time {
        display: inline-block;
        background: #E71A0F;
        color: #ffffff;
        padding: 3px 9px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 700;
        margin: 2px 4px 2px 0;
        box-shadow: 0 2px 6px rgba(229, 9, 20, 0.3);
    }
    
    /* Khung suy luận logic Horn Clauses */
    .inference-trace-box {
        background: #141b2b;
        border: 1px solid #20355a;
        border-radius: 8px;
        padding: 9px 13px;
        font-size: 12.5px;
        color: #9ec5fe;
        margin-bottom: 12px;
        line-height: 1.4;
    }
    
    /* Nút prompt gợi ý */
    .quick-prompt-btn {
        margin-bottom: 6px;
    }
</style>
""", unsafe_allow_html=True)

# --- KHỞI TẠO ĐỐI TƯỢNG VÀ CACHE ---
@st.cache_resource
def get_bot():
    return CGVChatbot()

@st.cache_resource
def get_kb():
    return MovieKnowledgeBase()

bot = get_bot()
kb = get_kb()

# Khởi tạo session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "👋 **Xin chào bạn!** Tôi là trợ lý ảo AI thông minh của **CGV Cinemas** 🍿.\n\n"
                "Tôi có thể hỗ trợ bạn tìm bộ phim lý tưởng nhất cho buổi xem hôm nay:\n"
                "- 🎯 **Gợi ý phim** theo thể loại yêu thích (Hành động, Kinh dị, Hài hước, Tình cảm...)\n"
                "- 👶 **Lọc phim đúng độ tuổi** an toàn (P, C13, C16, C18)\n"
                "- 📍 **Lọc theo cụm rạp & suất chiếu** (Vincom, Landmark 81, Aeon Mall, Bà Triệu...)\n"
                "- 🎬 **Định dạng phòng chiếu** (2D, 3D, IMAX)\n\n"
                "*Bạn muốn xem phim thể loại gì hoặc cần tư vấn thế nào? Hãy nhắn cho tôi nhé!*"
            ),
            "inference": None
        }
    ]

if "last_recommendations" not in st.session_state:
    all_m = kb.get_all_movies()
    top_initial = sorted(all_m, key=lambda x: x.get("rating", 0), reverse=True)[:3]
    for m in top_initial:
        m["explanation"] = f"Bộ phim bom tấn chiếu rạp với điểm đánh giá cực cao ⭐ {m.get('rating')}/10, đang rất được yêu thích tại các cụm rạp CGV!"
        m["similarity_score"] = 0.95
    st.session_state.last_recommendations = top_initial

# --- SIDEBAR: CGV VIP CONCIERGE & DỊCH VỤ NHANH ---
with st.sidebar:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #8B0000 0%, #E50914 50%, #B20710 100%); padding: 18px 15px; border-radius: 16px; text-align: center; color: white; box-shadow: 0 6px 20px rgba(229, 9, 20, 0.35); margin-bottom: 22px; border: 1px solid rgba(255, 255, 255, 0.15);">
        <div style="font-size: 24px; font-weight: 900; letter-spacing: 1.5px; text-shadow: 0 2px 4px rgba(0,0,0,0.5);">🍿 CGV VIP AI</div>
        <div style="font-size: 11px; opacity: 0.95; letter-spacing: 1px; margin-top: 4px; font-weight: 600;">TRỢ LÝ ĐIỆN ẢNH THÔNG MINH</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🍿 Dịch Vụ CGV Nhanh")
    sb_p1 = st.button("🥤 Tư vấn Bắp Nước & Combo", use_container_width=True)
    sb_p2 = st.button("🎟️ Tra cứu Giá Vé & Ưu đãi U22", use_container_width=True)
    sb_p3 = st.button("📍 Danh sách Cụm Rạp CGV", use_container_width=True)
    sb_p4 = st.button("🔥 Top Phim Hot Đang Chiếu", use_container_width=True)

    st.markdown("---")
    st.markdown("### 📊 Thống Kê Rạp")
    total_m = len(kb.get_all_movies())
    c1, c2 = st.columns(2)
    c1.metric("Tổng phim", f"{total_m}")
    c2.metric("Cụm rạp", f"{len(kb.get_unique_cinemas())}")

    st.markdown("---")
    with st.expander("🤖 Cấu hình Bộ Não AI (Dual-Mode)", expanded=False):
        enable_gemini = st.toggle("⚡ Kích hoạt Siêu AI (Gemini LLM)", value=False, help="Bật để sử dụng mô hình ngôn ngữ lớn Gemini đối thoại tự do. Tắt để dùng bộ sinh ngôn ngữ Offline.")
        DEFAULT_GEMINI_KEY = "AQ.Ab8RN6KGHVp7AzdN9at2m-L_idUJyugOCc_7zBbvYO9gGxSSVQ"
        custom_key = st.text_input("Gemini API Key:", value=DEFAULT_GEMINI_KEY, type="password")
        active_key = custom_key.strip() if custom_key.strip() else DEFAULT_GEMINI_KEY
        user_api_key = active_key if enable_gemini else None

        if enable_gemini:
            st.success("⚡ Đang chạy: **Gemini LLM Hybrid**")
        else:
            st.info("🟢 Đang chạy: **Offline Natural NLG**")

    st.markdown("---")
    if st.button("🗑️ Xóa Lịch Sử Trò Chuyện", use_container_width=True):
        st.session_state.messages = [st.session_state.messages[0]]
        st.rerun()

# --- HERO BANNER ---
st.markdown("""
<div class="cgv-hero-header">
    <div>
        <h1>🍿 CGV CINEMAS - SMART AI CHATBOT</h1>
        <p>Hệ thống tư vấn phim chiếu rạp thông minh kết hợp Suy Luận Logic Horn Clauses & Recommender System</p>
    </div>
    <div style="font-size: 42px;">🎬</div>
</div>
""", unsafe_allow_html=True)

# Bố cục 2 cột: Trái (Chat 2 chiều), Phải (Bảng tóm tắt Top 3)
col_chat, col_recs = st.columns([6, 5], gap="large")

# =======================================================
# CỘT TRÁI: GIAO DIỆN CHAT STREAMING
# =======================================================
with col_chat:
    st.subheader("💬 Trò Chuyện Cùng Trợ Lý CGV")

    prompt_clicked = None
    if sb_p1:
        prompt_clicked = "Nên mua combo bắp nước nào?"
    elif sb_p2:
        prompt_clicked = "Giá vé xem phim bao nhiêu tiền?"
    elif sb_p3:
        prompt_clicked = "Cho mình xem danh sách các rạp CGV"
    elif sb_p4:
        prompt_clicked = "Gợi ý phim hot nhất tuần này"

    # Gợi ý câu hỏi mẫu (Quick Prompts)
    st.markdown("<small style='color:#888;'>Gợi ý câu hỏi mẫu nhanh:</small>", unsafe_allow_html=True)
    q1, q2 = st.columns(2)
    if q1.button("👻 Kinh dị tối nay ở Landmark 81", use_container_width=True):
        prompt_clicked = "Tìm phim kinh dị chiếu vào buổi tối ở CGV Landmark 81"
    if q2.button("👨‍👩‍👧‍👦 Phim hoạt hình cho trẻ em", use_container_width=True):
        prompt_clicked = "Tư vấn phim hoạt hình vui vẻ cho bé 10 tuổi"
    
    q3, q4 = st.columns(2)
    if q3.button("💥 Hành động IMAX cho người 18+", use_container_width=True):
        prompt_clicked = "Gợi ý phim hành động bom tấn định dạng IMAX cho người 18 tuổi trở lên"
    if q4.button("🍿 Phim tình cảm ở Vincom", use_container_width=True):
        prompt_clicked = "Có phim tình cảm lãng mạn nào hay chiếu ở CGV Vincom Center không?"

    st.markdown("<hr style='margin: 10px 0 14px 0; border-color: #262633;'>", unsafe_allow_html=True)

    # Khung cuộn tin nhắn
    chat_container = st.container(height=520)

    # 1. Hiển thị toàn bộ lịch sử tin nhắn cũ
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"], avatar="🍿" if msg["role"] == "assistant" else "👤"):
                if msg.get("inference"):
                    st.markdown(f"<div class='inference-trace-box'>🔍 <b>Suy luận Logic (Horn Clauses):</b><br>{msg['inference']}</div>", unsafe_allow_html=True)
                st.markdown(msg["content"])

    # 2. Xử lý tin nhắn mới từ người dùng
    user_input = st.chat_input("Nhập câu hỏi (Ví dụ: 'Tìm phim trinh thám chiếu lúc 19h ở Bà Triệu')...")
    
    if prompt_clicked:
        user_input = prompt_clicked

    if user_input:
        # A. Hiển thị tin nhắn người dùng ngay lập tức
        with chat_container:
            with st.chat_message("user", avatar="👤"):
                st.markdown(user_input)

        # Lưu tin nhắn người dùng vào session_state
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })

        # B. Chatbot xử lý qua các tầng AI (Hỗ trợ Dual-Mode & Nhớ ngữ cảnh hội thoại)
        last_intent = st.session_state.get("last_intent")
        response = bot.process_message(user_input, api_key=user_api_key, last_intent=last_intent)
        st.session_state["last_intent"] = response.get("intent")

        # C. Hiển thị phản hồi của Bot bằng HIỆU ỨNG STREAMING GÕ CHỮ MƯỢT MÀ
        with chat_container:
            with st.chat_message("assistant", avatar="🍿"):
                # Hiển thị thông tin suy luận Horn Clauses
                if response.get("filter_explanation"):
                    st.markdown(f"<div class='inference-trace-box'>🔍 <b>Suy luận Logic (Horn Clauses):</b><br>{response['filter_explanation']}</div>", unsafe_allow_html=True)

                if response.get("mode_used") == "Gemini LLM Hybrid":
                    st.markdown("<small style='color: #4ade80;'>⚡ <i>Phản hồi được đối thoại bởi Google Gemini LLM</i></small><br>", unsafe_allow_html=True)

                # Generator phát từng từ mượt mà (Typewriter streaming effect)
                def stream_text_generator():
                    reply_text = response["reply_text"]
                    tokens = re.split(r"(\s+)", reply_text)
                    for token in tokens:
                        yield token
                        if token.strip():
                            time.sleep(0.012)  # Tốc độ lướt chữ nhịp nhàng

                st.write_stream(stream_text_generator)

        # D. Lưu tin nhắn bot vào session_state
        st.session_state.messages.append({
            "role": "assistant",
            "content": response["reply_text"],
            "inference": response.get("filter_explanation"),
            "mode_used": response.get("mode_used")
        })

        # E. Cập nhật Top 3 đề xuất cho cột bên phải
        if response.get("top_recommendations"):
            st.session_state.last_recommendations = response["top_recommendations"]

# =======================================================
# CỘT PHẢI: BẢNG TÓM TẮT TOP 3 PHIM & TRA CỨU
# =======================================================
with col_recs:
    tab_recs, tab_all = st.tabs(["🌟 Top 3 Phim Gợi Ý Tốt Nhất", "📋 Tra Cứu Toàn Bộ Phim CGV"])

    with tab_recs:
        st.markdown("#### 🏆 Bộ Ba Phim Phù Hợp Nhất Với Bạn")
        
        recs = st.session_state.last_recommendations
        if not recs:
            st.info("Hiện chưa có gợi ý nào. Hãy nhắn tin ở khung chat bên trái để nhận gợi ý tức thì!")
        else:
            for idx, movie in enumerate(recs, 1):
                age_rating = movie.get("age_rating", "P")
                score = movie.get("similarity_score", 0.85)
                match_pct = min(100, int(score * 100))
                
                with st.container():
                    st.markdown(f"""
                    <div class="movie-card-container">
                        <div style="display: flex; flex-direction: row;">
                            <div style="width: 135px; min-width: 135px; height: 215px; overflow: hidden; background: #111;">
                                <img src="{movie.get('poster_url', '')}" style="width: 100%; height: 100%; object-fit: cover;" onerror="this.src='https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?w=400';">
                            </div>
                            <div class="movie-card-body" style="flex: 1;">
                                <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                                    <span class="movie-card-title">#{idx}. {movie.get('title')}</span>
                                    <span class="badge-age badge-age-{age_rating}">{age_rating}</span>
                                </div>
                                <div class="movie-meta-row">
                                    🎭 <b>{', '.join(movie.get('genres', []))}</b> • ⏱️ {movie.get('duration', 120)} phút
                                </div>
                                <div style="margin-bottom: 6px; font-size: 13px;">
                                    <span style="color: #f5c518; font-weight: 700;">⭐ {movie.get('rating', 7.5)}/10</span>
                                    <span style="color: #4ade80; font-weight: 600; margin-left: 8px;">🎯 {match_pct}% Phù hợp</span>
                                </div>
                                <div>
                                    <span style="font-size: 12px; color: #aaa;">Định dạng:</span> 
                                    {' '.join([f"<span class='chip-format'>{fmt}</span>" for fmt in movie.get('formats', [])])}
                                </div>
                                <div style="margin-top: 6px;">
                                    <span style="font-size: 12px; color: #aaa;">Suất chiếu:</span> 
                                    {' '.join([f"<span class='chip-time'>{st_time}</span>" for st_time in movie.get('matched_showtimes', movie.get('showtimes', []))[:4]])}
                                </div>
                                <div class="movie-explanation-box">
                                    💡 <b>Lý do chọn:</b> {movie.get('explanation', '')}
                                </div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Nút tương tác chọn suất chiếu giả lập
                    st_times = movie.get('matched_showtimes', movie.get('showtimes', []))[:3]
                    if st_times:
                        cols_st = st.columns(len(st_times) + 1)
                        cols_st[0].caption("🎟️ Chọn suất:")
                        for i_st, st_t in enumerate(st_times):
                            if cols_st[i_st + 1].button(f"{st_t}", key=f"btn_{movie.get('id')}_{st_t}_{idx}"):
                                st.toast(f"✅ Đã chọn vé xem '{movie.get('title')}' suất {st_t} tại rạp CGV!", icon="🍿")

    with tab_all:
        st.markdown("#### 🎬 Danh Sách Toàn Bộ 65 Phim Tại CGV")
        
        # Bộ lọc tương tác
        all_movies = kb.get_all_movies()
        f_col1, f_col2 = st.columns(2)
        filter_genre = f_col1.selectbox("Lọc theo Thể loại:", ["Tất cả"] + kb.get_unique_genres())
        filter_age = f_col2.selectbox("Lọc theo Độ tuổi:", ["Tất cả", "P", "C13", "C16", "C18"])
        
        filtered = all_movies
        if filter_genre != "Tất cả":
            filtered = [m for m in filtered if filter_genre in m.get("genres", [])]
        if filter_age != "Tất cả":
            filtered = [m for m in filtered if m.get("age_rating") == filter_age]

        st.caption(f"Đang hiển thị **{len(filtered)}/{len(all_movies)}** phim:")
        
        for m in filtered[:10]:
            with st.expander(f"🎬 {m.get('title')} - [{m.get('age_rating')}] - ⭐ {m.get('rating')}/10"):
                st.write(f"**Tên gốc**: {m.get('original_title')}")
                st.write(f"**Thể loại**: {', '.join(m.get('genres', []))}")
                st.write(f"**Đạo diễn**: {m.get('director', 'Đang cập nhật')} | **Diễn viên**: {', '.join(m.get('cast', []))}")
                st.write(f"**Rạp chiếu**: {', '.join(m.get('cinemas', []))}")
                st.write(f"**Suất chiếu**: {', '.join(m.get('showtimes', []))}")
                st.write(f"**Tóm tắt**: {m.get('description')}")
        
        if len(filtered) > 10:
            st.info(f"Và còn {len(filtered) - 10} phim khác trong cơ sở tri thức...")
