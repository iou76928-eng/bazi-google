# 1. 使用官方 Python 輕量版映像檔 (與您的 Render 環境類似)
FROM python:3.9-slim

# 2. 設定容器內的工作目錄
WORKDIR /app

# 3. 複製 requirements.txt 並安裝套件
#    先做這步可以利用 Docker 快取，加速部署
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. 複製所有程式碼到容器內
#    這會把 app_updated.py, 八字.py, bazi_calc_v2.py 全部複製進去
COPY . .

# 5. 設定環境變數 (Cloud Run 會自動注入 PORT，這是備用預設值)
ENV PORT=8080

# 6. 啟動指令 (關鍵！)
#    解釋：使用 gunicorn 啟動 app_updated.py 裡面的 app 物件
#    注意：如果您的主程式檔名不是 app_updated.py，請修改下行冒號前的名稱
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app_updated:app