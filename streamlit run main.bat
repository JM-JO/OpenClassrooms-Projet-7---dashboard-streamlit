@echo off
title STREAMLIT
call %USERPROFILE%\anaconda3\Scripts\activate.bat
@echo on
streamlit run main.py
pause

