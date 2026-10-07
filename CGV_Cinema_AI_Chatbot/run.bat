@echo off
chcp 65001 > nul
title CGV Cinema AI Chatbot - Bai Tap Lon Tri Tue Nhan Tao

echo ========================================================
echo   POP-CORN & CINEMA: CGV SMART AI CHATBOT
echo   BÀI TẬP LỚN MÔN TRÍ TUỆ NHÂN TẠO
echo ========================================================
echo.

echo [1/3] Đang kiểm tra môi trường Python...
python --version > nul 2>&1
if errorlevel 1 (
    echo [LỖI] Không tìm thấy Python! Vui lòng cài đặt Python và tích chọn "Add Python to PATH".
    pause
    exit /b
)

echo [2/3] Đang kiểm tra cơ sở dữ liệu phim (movies.json)...
if not exist "data\movies.json" (
    echo Chưa có dữ liệu phim, đang khởi tạo data\movies.json...
    python data\generate_movies.py
)

echo [3/3] Đang khởi chạy ứng dụng Web Streamlit...
echo Hệ thống sẽ tự động mở trình duyệt tại địa chỉ http://localhost:8501
echo Nhấn Ctrl + C để dừng ứng dụng khi kết thúc.
echo.

streamlit run app.py

pause
