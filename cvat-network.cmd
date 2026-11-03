@echo off
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0cvat-network.ps1" %*
