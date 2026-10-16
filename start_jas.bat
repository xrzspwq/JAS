@echo off
rem Double-click launcher: runs start_jas.ps1 even if PowerShell script execution is restricted.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_jas.ps1" %*
