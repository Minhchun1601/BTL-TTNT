"""
Module Động Cơ Suy Luận Logic (Rule-based Reasoning Engine).
Áp dụng các mệnh đề Horn (Horn Clauses) và hệ luật If-Then để suy luận lọc ứng viên hợp lệ:
1. Luật ràng buộc độ tuổi (Legal Age Restriction Horn Clause).
2. Luật ràng buộc cụm rạp khả dụng (Cinema Availability Rule).
3. Luật ràng buộc khung giờ chiếu (Showtime Feasibility Rule).
4. Luật định dạng phòng chiếu (Format Compatibility Rule).
"""

from typing import List, Dict, Any, Tuple

class ReasoningEngine:
    """Động cơ suy luận luật If-Then (Horn Clauses) phục vụ hệ thống chuyên gia chọn phim."""

    # Bản đồ ngưỡng tuổi tối thiểu tương ứng với nhãn phim CGV
    AGE_LIMITS = {
        "P": 0,       # Phổ biến mọi lứa tuổi
        "C13": 13,    # Cấm người dưới 13 tuổi
        "C16": 16,    # Cấm người dưới 16 tuổi
        "C18": 18     # Cấm người dưới 18 tuổi
    }

    def __init__(self):
        pass

    @classmethod
    def check_age_rule(cls, movie_age_rating: str, user_age: int) -> Tuple[bool, str]:
        """
        Horn Clause 1 (Luật kiểm tra độ tuổi):
        IF (Rating == 'C18' AND UserAge < 18) THEN REJECT
        IF (Rating == 'C16' AND UserAge < 16) THEN REJECT
        IF (Rating == 'C13' AND UserAge < 13) THEN REJECT
        ELSE ACCEPT
        """
        required_age = cls.AGE_LIMITS.get(movie_age_rating, 0)
        if user_age < required_age:
            reason = f"Không thỏa mãn độ tuổi: Phim dán nhãn {movie_age_rating} (yêu cầu từ {required_age} tuổi trở lên), người xem {user_age} tuổi."
            return False, reason
        return True, "Thỏa mãn độ tuổi quy định."

    @staticmethod
    def check_cinema_rule(movie_cinemas: List[str], requested_cinemas: List[str]) -> Tuple[bool, str]:
        """
        Horn Clause 2 (Luật cụm rạp):
        IF (RequestedCinemas is NOT EMPTY AND MovieCinemas ∩ RequestedCinemas == ∅) THEN REJECT
        ELSE ACCEPT
        """
        if not requested_cinemas:
            return True, "Người dùng không giới hạn rạp chiếu cụ thể."
            
        matched = set(movie_cinemas).intersection(set(requested_cinemas))
        if matched:
            return True, f"Có chiếu tại rạp yêu cầu: {', '.join(matched)}."
        else:
            return False, f"Phim không có suất chiếu tại các rạp được chọn: {', '.join(requested_cinemas)}."

    @staticmethod
    def check_format_rule(movie_formats: List[str], requested_formats: List[str]) -> Tuple[bool, str]:
        """
        Horn Clause 3 (Luật định dạng phòng chiếu 2D/3D/IMAX):
        IF (RequestedFormats is NOT EMPTY AND MovieFormats ∩ RequestedFormats == ∅) THEN REJECT
        ELSE ACCEPT
        """
        if not requested_formats:
            return True, "Chấp nhận mọi định dạng phòng chiếu."

        matched = set(movie_formats).intersection(set(requested_formats))
        if matched:
            return True, f"Hỗ trợ định dạng yêu cầu: {', '.join(matched)}."
        else:
            return False, f"Phim không hỗ trợ định dạng {', '.join(requested_formats)}."

    @staticmethod
    def _parse_time_to_minutes(time_str: str) -> int:
        """Chuyển chuỗi giờ '18:30' sang số phút từ 0h."""
        parts = time_str.strip().split(":")
        return int(parts[0]) * 60 + int(parts[1])

    @classmethod
    def check_time_rule(cls, movie_showtimes: List[str], time_period: str = None, specific_hour: float = None) -> Tuple[bool, List[str], str]:
        """
        Horn Clause 4 (Luật khung giờ chiếu):
        IF (Khung giờ mong muốn không có bất kỳ suất chiếu nào thỏa mãn) THEN REJECT
        ELSE ACCEPT kèm danh sách các suất chiếu phù hợp.
        """
        if not time_period and specific_hour is None:
            return True, movie_showtimes, "Không giới hạn khung giờ chiếu."

        valid_showtimes = []

        for st in movie_showtimes:
            minutes = cls._parse_time_to_minutes(st)
            hour_float = minutes / 60.0

            # Trường hợp 1: Có giờ cụ thể (Cho phép lệch ± 2 giờ)
            if specific_hour is not None:
                if abs(hour_float - specific_hour) <= 2.0:
                    valid_showtimes.append(st)
            # Trường hợp 2: Khung giờ trong ngày
            elif time_period == "morning":      # Sáng (< 12:00)
                if hour_float < 12.0:
                    valid_showtimes.append(st)
            elif time_period == "afternoon":    # Chiều (12:00 - 17:59)
                if 12.0 <= hour_float < 18.0:
                    valid_showtimes.append(st)
            elif time_period == "evening":      # Tối (18:00 - 21:59)
                if 18.0 <= hour_float < 22.0:
                    valid_showtimes.append(st)
            elif time_period == "late_night":  # Đêm muộn (>= 22:00)
                if hour_float >= 22.0:
                    valid_showtimes.append(st)

        if valid_showtimes:
            return True, valid_showtimes, f"Tìm thấy {len(valid_showtimes)} suất chiếu phù hợp: {', '.join(valid_showtimes)}."
        else:
            return False, [], "Không có suất chiếu nào phù hợp với khung giờ yêu cầu."

    def filter_candidates(self, movies: List[Dict[str, Any]], entities: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Áp dụng toàn bộ hệ luật logic suy luận (Horn Clauses Filter)
        để lọc ra danh sách các bộ phim ứng viên hợp lệ.
        
        Trả về:
        - passed_candidates: Danh sách phim thỏa mãn mọi ràng buộc cứng.
        - rejected_candidates: Danh sách phim bị loại kèm nguyên nhân logic bị loại.
        """
        user_age = entities.get("user_age")
        cinemas = entities.get("cinemas", [])
        formats = entities.get("formats", [])
        time_period = entities.get("time_period")
        specific_hour = entities.get("specific_hour")

        passed_candidates = []
        rejected_candidates = []

        for movie in movies:
            # 1. Kiểm tra luật độ tuổi
            if user_age is not None:
                is_age_valid, age_msg = self.check_age_rule(movie.get("age_rating", "P"), user_age)
                if not is_age_valid:
                    rejected_candidates.append({"movie": movie, "reason": age_msg})
                    continue

            # 2. Kiểm tra luật rạp chiếu
            if cinemas:
                is_cinema_valid, cinema_msg = self.check_cinema_rule(movie.get("cinemas", []), cinemas)
                if not is_cinema_valid:
                    rejected_candidates.append({"movie": movie, "reason": cinema_msg})
                    continue

            # 3. Kiểm tra luật định dạng phòng chiếu
            if formats:
                is_fmt_valid, fmt_msg = self.check_format_rule(movie.get("formats", []), formats)
                if not is_fmt_valid:
                    rejected_candidates.append({"movie": movie, "reason": fmt_msg})
                    continue

            # 4. Kiểm tra luật khung giờ chiếu
            if time_period or specific_hour is not None:
                is_time_valid, valid_st, time_msg = self.check_time_rule(movie.get("showtimes", []), time_period, specific_hour)
                if not is_time_valid:
                    rejected_candidates.append({"movie": movie, "reason": time_msg})
                    continue
                # Lưu lại các suất chiếu hợp lệ để hiển thị
                movie_copy = dict(movie)
                movie_copy["matched_showtimes"] = valid_st
            else:
                movie_copy = dict(movie)
                movie_copy["matched_showtimes"] = movie.get("showtimes", [])

            passed_candidates.append(movie_copy)

        return passed_candidates, rejected_candidates
