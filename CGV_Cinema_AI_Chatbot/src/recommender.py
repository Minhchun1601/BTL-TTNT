"""
Module Thuật Toán Gợi Ý Phim (Movie Recommender System & Explainable AI).
Áp dụng:
1. Content-Based Filtering (Dựa trên thể loại và nội dung tóm tắt phim).
2. Tích hợp điểm đánh giá Rating và độ ưu tiên rạp/suất chiếu (Hybrid Ranking).
3. Cơ chế sinh giải thích trực quan (Explainable AI - XAI).
"""

import math
from typing import List, Dict, Any, Tuple

class MovieRecommender:
    """Hệ thống xếp hạng và khuyến nghị phim rạp CGV."""

    def __init__(self, weight_genre: float = 0.40, weight_content: float = 0.25, weight_rating: float = 0.25, weight_cinema: float = 0.10):
        # Trọng số các thành phần điểm
        self.w_genre = weight_genre
        self.w_content = weight_content
        self.w_rating = weight_rating
        self.w_cinema = weight_cinema

    @staticmethod
    def _compute_genre_similarity(movie_genres: List[str], requested_genres: List[str]) -> float:
        """
        Tính độ tương đồng thể loại theo chỉ số Jaccard Similarity:
        J(A, B) = |A ∩ B| / |A ∪ B|
        Nếu người dùng không chỉ định thể loại, trả về 1.0 để không thiên vị.
        """
        if not requested_genres:
            return 1.0

        set_movie = set(g.lower() for g in movie_genres)
        set_req = set(g.lower() for g in requested_genres)

        intersection = set_movie.intersection(set_req)
        union = set_movie.union(set_req)

        if not union:
            return 0.0
            
        # Thưởng thêm nếu trùng khớp hoàn toàn thể loại yêu cầu
        base_jaccard = len(intersection) / len(union)
        coverage = len(intersection) / len(set_req)
        return 0.5 * base_jaccard + 0.5 * coverage

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Tách từ và tiền xử lý cho tiếng Việt."""
        import re
        tokens = re.findall(r"\w+", text.lower())
        # Loại bỏ từ có 1 ký tự và từ dừng cơ bản
        stop_words = {"và", "của", "là", "thì", "mà", "ở", "với", "cho", "được", "có", "những", "các", "này", "đó", "ra", "vào", "phim", "xem"}
        return [t for t in tokens if len(t) > 1 and t not in stop_words]

    def _extract_matching_keywords(self, user_query: str, movie: Dict[str, Any]) -> List[str]:
        """
        Trích xuất các từ khóa trùng khớp giữa câu truy vấn của người dùng và kho từ khóa (keywords) + mô tả của bộ phim.
        Hỗ trợ so khớp có dấu, KHÔNG DẤU (Unaccented) và gần đúng.
        """
        from src.nlp_processor import NLPProcessor
        norm_query = NLPProcessor.normalize_text(user_query)
        no_acc_query = NLPProcessor.remove_accents(user_query)

        movie_keywords = movie.get("keywords", [])
        matched = []

        # 1. So khớp các cụm từ khóa (keywords)
        for kw in movie_keywords:
            kw_norm = NLPProcessor.normalize_text(kw)
            kw_no_acc = NLPProcessor.remove_accents(kw)
            if (len(kw_norm) > 1 and kw_norm in norm_query) or (len(kw_no_acc) > 2 and kw_no_acc in no_acc_query):
                if kw not in matched:
                    matched.append(kw)

        # 2. So khớp Đạo diễn / Diễn viên trực tiếp
        director = movie.get("director")
        if director:
            dir_norm = NLPProcessor.normalize_text(director)
            dir_no_acc = NLPProcessor.remove_accents(director)
            if (dir_norm and dir_norm in norm_query) or (len(dir_no_acc) > 3 and dir_no_acc in no_acc_query):
                if director not in matched:
                    matched.append(director)

        cast = movie.get("cast", [])
        for actor in cast:
            if actor:
                act_norm = NLPProcessor.normalize_text(actor)
                act_no_acc = NLPProcessor.remove_accents(actor)
                if (act_norm and act_norm in norm_query) or (len(act_no_acc) > 3 and act_no_acc in no_acc_query):
                    if actor not in matched:
                        matched.append(actor)

        # 3. So khớp Tên phim (Title)
        title = movie.get("title", "")
        if title:
            title_no_acc = NLPProcessor.remove_accents(title)
            if len(title_no_acc) > 2 and title_no_acc in no_acc_query:
                if title not in matched:
                    matched.append(title)

        # Loại bỏ các từ đơn nếu nó đã nằm trong một cụm từ dài hơn
        filtered_matched = []
        for kw in matched:
            kw_norm = NLPProcessor.normalize_text(kw)
            is_sub = any(kw_norm != NLPProcessor.normalize_text(other) and (f" {kw_norm} " in f" {NLPProcessor.normalize_text(other)} ") for other in matched)
            if not is_sub:
                filtered_matched.append(kw)

        return filtered_matched[:5]  # Lấy tối đa 5 từ khóa khớp nhất

    def _compute_text_cosine_similarity(self, query: str, movie_doc: str) -> float:
        """
        Tính độ tương đồng Cosine TF-IDF giữa câu hỏi người dùng và nội dung phim.
        Hỗ trợ so khớp cả có dấu lẫn KHÔNG DẤU để xử lý gõ sai / gõ thiếu dấu.
        """
        from src.nlp_processor import NLPProcessor
        query_tokens = self._tokenize(query)
        doc_tokens = self._tokenize(movie_doc)

        no_acc_query_tokens = [NLPProcessor.remove_accents(t) for t in query_tokens]
        no_acc_doc_tokens = [NLPProcessor.remove_accents(t) for t in doc_tokens]

        all_q = query_tokens + no_acc_query_tokens
        all_d = doc_tokens + no_acc_doc_tokens

        if not all_q or not all_d:
            return 0.0

        # Đếm tần suất từ (Term Frequency)
        tf_query: Dict[str, int] = {}
        for t in all_q:
            tf_query[t] = tf_query.get(t, 0) + 1

        tf_doc: Dict[str, int] = {}
        for t in all_d:
            tf_doc[t] = tf_doc.get(t, 0) + 1

        # Tích vô hướng
        dot_product = 0.0
        for t, freq in tf_query.items():
            if t in tf_doc:
                dot_product += freq * tf_doc[t]

        # Độ dài vector (Norm)
        norm_q = math.sqrt(sum(v ** 2 for v in tf_query.values()))
        norm_d = math.sqrt(sum(v ** 2 for v in tf_doc.values()))

        if norm_q == 0 or norm_d == 0:
            return 0.0

        return dot_product / (norm_q * norm_d)

    def _generate_explanation(self, movie: Dict[str, Any], entities: Dict[str, Any], score: float, matched_keywords: List[str] = None) -> str:
        """
        Mô hình Explainable AI (XAI):
        Tự động tổng hợp lý do ngắn gọn, cô đọng cho từng phim.
        """
        reasons = []
        if matched_keywords:
            reasons.append(f"Khớp từ khóa **{', '.join(matched_keywords)}**")

        req_genres = entities.get("genres", [])
        if req_genres:
            matched_g = [g for g in movie.get("genres", []) if g in req_genres]
            if matched_g:
                reasons.append(f"Đúng dòng phim **{', '.join(matched_g)}**")

        # Thêm tóm tắt ngắn từ mô tả nếu chưa có nhiều lý do
        desc = movie.get("description", "")
        if desc and len(reasons) < 2:
            short_desc = desc.split(".")[0] if "." in desc else desc
            if len(short_desc) <= 80:
                reasons.append(f"{short_desc}")

        if not reasons:
            reasons.append(f"Phim hot đánh giá cao ⭐ {movie.get('rating', 7.0)}/10")

        return ". ".join(reasons) + "."

    def rank_and_recommend(
        self,
        candidate_movies: List[Dict[str, Any]],
        user_query: str,
        entities: Dict[str, Any],
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Tính điểm kết hợp (Hybrid Score) cho từng phim và trả về Top K phim tối ưu nhất kèm giải thích.
        """
        if not candidate_movies:
            return []

        req_genres = entities.get("genres", [])
        req_cinemas = entities.get("cinemas", [])

        scored_movies = []

        for movie in candidate_movies:
            # A. Điểm tương đồng thể loại (Jaccard)
            genre_sim = self._compute_genre_similarity(movie.get("genres", []), req_genres)

            # B. Điểm tương đồng nội dung văn bản & từ khóa (Cosine TF-IDF)
            keywords_str = " ".join(movie.get("keywords", []))
            cast_str = " ".join(movie.get("cast", []))
            director_str = movie.get("director", "")
            content_doc = f"{movie.get('title', '')} {keywords_str} {movie.get('description', '')} {cast_str} {director_str} {' '.join(movie.get('genres', []))}"
            content_sim = self._compute_text_cosine_similarity(user_query, content_doc)

            # Trích xuất các từ khóa trùng khớp trực tiếp
            matched_keywords = self._extract_matching_keywords(user_query, movie)
            if matched_keywords:
                # Thưởng điểm cao hơn cho từ khóa cụm (multi-word) hoặc tên đạo diễn/diễn viên (như Trấn Thành)
                cast_l = [c.lower() for c in movie.get("cast", [])]
                dir_l = movie.get("director", "").lower()
                
                phrase_matches = [
                    k for k in matched_keywords 
                    if len(k.split()) >= 2 or k.lower() in cast_l or k.lower() == dir_l
                ]
                bonus = 0.45 * len(phrase_matches) + 0.15 * (len(matched_keywords) - len(phrase_matches))
                content_sim = min(1.0, content_sim + bonus)

            # C. Điểm đánh giá chuẩn hóa về [0, 1]
            rating_score = movie.get("rating", 7.0) / 10.0

            # D. Điểm ưu tiên rạp
            cinema_score = 1.0 if (not req_cinemas or set(movie.get("cinemas", [])).intersection(set(req_cinemas))) else 0.0

            # Nếu có từ khóa nhân vật/đạo diễn/khái niệm đặc biệt, tăng mạnh ưu tiên tương đồng nội dung
            w_c = self.w_content
            w_g = self.w_genre
            keyword_bonus = 0.0
            if matched_keywords:
                w_c = 0.50
                w_g = 0.15
                keyword_bonus = 0.35 + 0.10 * len(matched_keywords)

            # Tổng hợp điểm số đa tiêu chí
            final_score = (
                w_g * genre_sim +
                w_c * content_sim +
                self.w_rating * rating_score +
                self.w_cinema * cinema_score +
                keyword_bonus
            )

            # Tạo lời giải thích minh bạch XAI
            explanation = self._generate_explanation(movie, entities, final_score, matched_keywords=matched_keywords)

            result_item = dict(movie)
            result_item["similarity_score"] = round(final_score, 4)
            result_item["genre_match_score"] = round(genre_sim, 2)
            result_item["matched_keywords"] = matched_keywords
            result_item["explanation"] = explanation

            scored_movies.append(result_item)

        # Sắp xếp giảm dần theo điểm số tổng hợp
        scored_movies.sort(key=lambda x: (x["similarity_score"], x.get("rating", 0.0)), reverse=True)

        return scored_movies[:top_k]
