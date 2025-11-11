@echo off
powershell -Command "Start-Process powershell -ArgumentList '-NoExit', '-Command', 'cd backend; .\.venv\Scripts\activate; flask run --debug' -WindowStyle Normal"
cd hair-moggin
npm run dev
