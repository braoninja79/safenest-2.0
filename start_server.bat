@echo off
start python stream_server.py
timeout /t 3
start "" "index.html"
