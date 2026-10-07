"""
Module Quản lý Cơ sở Tri thức (Knowledge Base Manager).
Chịu trách nhiệm tải, lưu trữ và cung cấp các hàm truy vấn dữ liệu phim rạp CGV.
"""

import json
import os
from typing import List, Dict, Any, Optional

class MovieKnowledgeBase:
    """Lớp quản lý cơ sở tri thức phim rạp CGV."""
    
    def __init__(self, data_path: Optional[str] = None):
        if data_path is None:
            # Đường dẫn mặc định đến file movies.json
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_path = os.path.join(base_dir, "data", "movies.json")
            
        self.data_path = data_path
        self.movies: List[Dict[str, Any]] = []
        self.load_data()
        
    def load_data(self) -> None:
        """Đọc toàn bộ dữ liệu phim từ file JSON vào bộ nhớ."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Không tìm thấy file cơ sở tri thức tại: {self.data_path}")
            
        with open(self.data_path, "r", encoding="utf-8") as f:
            self.movies = json.load(f)
            
    def get_all_movies(self) -> List[Dict[str, Any]]:
        """Trả về toàn bộ danh sách phim trong cơ sở tri thức."""
        return self.movies
        
    def get_movie_by_id(self, movie_id: str) -> Optional[Dict[str, Any]]:
        """Tìm phim theo mã định danh (ID)."""
        for m in self.movies:
            if m.get("id", "").upper() == movie_id.upper():
                return m
        return None

    def get_unique_genres(self) -> List[str]:
        """Lấy danh sách tất cả các thể loại phim có trong hệ thống."""
        genres_set = set()
        for m in self.movies:
            for g in m.get("genres", []):
                genres_set.add(g)
        return sorted(list(genres_set))

    def get_unique_cinemas(self) -> List[str]:
        """Lấy danh sách các cụm rạp CGV."""
        cinemas_set = set()
        for m in self.movies:
            for c in m.get("cinemas", []):
                cinemas_set.add(c)
        return sorted(list(cinemas_set))

    def get_unique_age_ratings(self) -> List[str]:
        """Lấy danh sách các nhãn phân loại độ tuổi (P, C13, C16, C18)."""
        return ["P", "C13", "C16", "C18"]

    def get_unique_formats(self) -> List[str]:
        """Lấy danh sách các định dạng chiếu (2D, 3D, IMAX)."""
        return ["2D", "3D", "IMAX"]
