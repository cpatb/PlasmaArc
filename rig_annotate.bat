@echo off
rem Drag a .cine file or a folder of cines onto this file (or run: rig_annotate.bat <path> [--gap-mm 2.4]).
rem Annotations are saved in "%~dp0rig_annotations\".
cd /d "%~dp0"
python -X utf8 rig_annotate.py %*
pause
