from flask import Flask, request, render_template_string
import traceback
import os
from datetime import datetime, timedelta
try:
    from zoneinfo import ZoneInfo  # Py3.9+
except Exception:
    ZoneInfo = None  # type: ignore

# ✅ 改用「bazi.py」本地運算，不再走爬蟲
#    兼容中文檔名：優先正常 import，失敗則用 importlib 動態載入
try:
    import bazi as bazi_py  # type: ignore
except Exception:
    import importlib.util
    from pathlib import Path
    _bazi_path = Path(__file__).with_name("bazi.py")
    _spec = importlib.util.spec_from_file_location("bazi_py", _bazi_path)
    if _spec is None or _spec.loader is None:
        raise ImportError(f"無法載入bazi.py：{_bazi_path}")
    bazi_py = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(bazi_py)  # type: ignore

calc_bazi_8char = bazi_py.calc_bazi_8char

from bazi_calc_v2 import WebBaziAnalyzer, ZHI

app = Flask(__name__)


def now_in_taipei() -> datetime:
    """Return a 'now' datetime in Asia/Taipei.

    Render (or other minimal containers) might lack IANA tzdata. We try ZoneInfo
    first and fall back to UTC+8.
    """
    if ZoneInfo is not None:
        try:
            return datetime.now(ZoneInfo("Asia/Taipei"))
        except Exception:
            pass
    return datetime.utcnow() + timedelta(hours=8)

# ==========================================
# 🎨 前端設計：CSS 樣式庫 (米黃禪意風)
# ==========================================
COMMON_CSS = """
    @import url('https://fonts.googleapis.com/css2?family=Noto+Serif+TC:wght@400;700&family=Noto+Sans+TC:wght@300;400;500&display=swap');
    
    :root {
        --bg-color: #fdfbf7; /* 米黃宣紙色 */
        --card-bg: #ffffff;
        --primary-color: #5d4037; /* 深褐 */
        --accent-color: #c0392b; /* 硃砂紅 */
        --text-color: #4a4a4a;
        --shadow: 0 10px 30px rgba(93, 64, 55, 0.1);
        --radius: 12px;
    }

    body { 
        font-family: 'Noto Sans TC', sans-serif; 
        background-color: var(--bg-color); 
        color: var(--text-color);
        margin: 0; padding: 0;
        line-height: 1.6;
        background-image: linear-gradient(to bottom, #fdfbf7 0%, #f5f0e6 100%);
    }
    
    h1, h2, h3 { font-family: 'Noto Serif TC', serif; color: var(--primary-color); }
    
    .container { max-width: 800px; margin: 0 auto; padding: 20px; }
    
    .card { 
        background: var(--card-bg); 
        padding: 2.5rem; 
        border-radius: var(--radius); 
        box-shadow: var(--shadow); 
        margin-bottom: 2rem; 
        border: 1px solid rgba(0,0,0,0.03);
    }

    .btn-primary {
        width: 100%; 
        padding: 1rem; 
        background-color: var(--primary-color); 
        color: white; 
        border: none; 
        border-radius: 8px; 
        font-size: 1.1rem; 
        cursor: pointer; 
        transition: all 0.3s; 
        font-family: 'Noto Serif TC', serif;
        letter-spacing: 2px;
    }
    .btn-primary:hover { background-color: #3e2723; transform: translateY(-2px); box-shadow: 0 5px 15px rgba(0,0,0,0.2); }
    
    /* 讀取動畫 */
    .loading-overlay {
        display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%;
        background: rgba(253, 251, 247, 0.95); z-index: 999;
        text-align: center; padding-top: 30vh;
    }
    .spinner {
        border: 4px solid #f3f3f3; border-top: 4px solid var(--accent-color);
        border-radius: 50%; width: 50px; height: 50px; margin: 0 auto 20px;
        animation: spin 1s linear infinite;
    }
    @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
"""

# ==========================================
# 🏠 首頁 HTML
# ==========================================
INDEX_HTML = f"""
<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>八字日支運勢指南</title>
    <style>
        {COMMON_CSS}
        .hero-section {{ text-align: center; margin-bottom: 2rem; }}
        .hero-title {{ font-size: 2.2rem; margin-bottom: 0.5rem; }}
        .hero-subtitle {{ font-size: 1rem; color: #888; font-weight: 300; letter-spacing: 1px; }}
        
        .form-group {{ margin-bottom: 1.2rem; }}
        label {{ display: block; margin-bottom: 0.5rem; color: var(--primary-color); font-weight: bold; font-size: 0.95rem; }}
        input, select {{ 
            width: 100%; padding: 0.8rem; border: 1px solid #ddd; 
            border-radius: 6px; font-size: 1rem; background: #fafafa;
            box-sizing: border-box;
        }}
        input:focus, select:focus {{ border-color: var(--primary-color); outline: none; }}
        
        .radio-group {{ display: flex; gap: 1.5rem; }}
        
        /* 底部介紹區塊 */
        .intro-section {{ 
            margin-top: 3rem; border-top: 1px solid #e0e0e0; padding-top: 2rem;
            text-align: center; color: #666; font-size: 0.95rem;
        }}
        .intro-title {{ font-size: 1.2rem; color: var(--accent-color); margin-bottom: 1rem; }}
    </style>
    <script>
        function showLoading() {{
            document.getElementById('loading').style.display = 'block';
        }}
    </script>
</head>
<body>
    <div class="container">
        <div class="card">
            <div class="hero-section">
                <h1 class="hero-title">🔮 八字日支運勢指南</h1>
                <p class="hero-subtitle">古老智慧 × 現代運算 · 三秒洞悉今日契機</p>
            </div>
            
            <form action="/analyze" method="POST" onsubmit="showLoading()">
                <div class="form-group">
                    <label>您的姓名 (選填，用於排盤)</label>
                    <input type="text" name="name" placeholder="請輸入姓名">
                </div>
                
                <div class="form-group">
                    <label>性別</label>
                    <div class="radio-group">
                        <label><input type="radio" name="sex" value="1" checked> 男性</label>
                        <label><input type="radio" name="sex" value="0"> 女性</label>
                    </div>
                </div>

                <div class="form-group">
                    <label>出生民國年 (例如：76)</label>
                    <input type="number" name="year" required placeholder="請輸入數字，如 76">
                </div>

                <div style="display: flex; gap: 15px;">
                    <div class="form-group" style="flex:1">
                        <label>出生月</label>
                        <select name="month" required>
                            <script>for(let i=1;i<=12;i++) document.write(`<option value="${{i}}">${{i}} 月</option>`);</script>
                        </select>
                    </div>
                    <div class="form-group" style="flex:1">
                        <label>出生日</label>
                        <select name="day" required>
                            <script>for(let i=1;i<=31;i++) document.write(`<option value="${{i}}">${{i}} 日</option>`);</script>
                        </select>
                    </div>
                </div>

                <div style="display: flex; gap: 15px;">
                    <div class="form-group" style="flex:1">
                        <label>出生時 (0-23)</label>
                        <select name="hour" required>
                            <script>for(let i=0;i<=23;i++) document.write(`<option value="${{i}}">${{i}} 時</option>`);</script>
                        </select>
                    </div>
                    <div class="form-group" style="flex:1">
                        <label>出生分 (選填)</label>
                        <input type="number" name="minute" value="0" min="0" max="59">
                    </div>
                </div>
                
                <input type="hidden" name="year_mode" value="1"> 
                <button type="submit" class="btn-primary">開始運算</button>
            </form>
        </div>

<div class="intro-section">
            <h3 class="intro-title">這不是算命，這是你的決策系統</h3>
            <p style="max-width: 600px; margin: 0 auto; line-height: 1.8;">
                傳統命理給你的是一本寫滿吉凶的「宿命帳本」，<br>
                但我們提供的是一套穿越迷霧的<strong>「決策系統」</strong>。<br><br>
                問題不在於「準不準」，而在於「怎麼用」。<br>
                我們拒絕販賣恐懼，因為恐懼讓人困擾；<br>
                我們販賣的是<strong>「方向」</strong>，因為方向能讓你從容抵達。<br>
                <br>
                <strong>別讓你的人生，只是一場聽天由命的賭局。</strong>
            </p>
            <p style="margin-top: 2rem; font-size: 0.8rem; color:#999;">© 2025 編碼命運. All rights reserved.</p>
        </div>    </div>

    <div id="loading" class="loading-overlay">
        <div class="spinner"></div>
        <h3 style="color:#5d4037;">正在連線命理資料庫...</h3>
        <p style="color:#666;">系統正在提取四柱資訊，約需 10-15 秒</p>
    </div>
</body>
</html>
"""

# ==========================================
# 📊 結果頁 HTML (排版緊湊優化版)
# ==========================================
RESULT_HTML = f"""
<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>分析結果</title>
    <style>
        {COMMON_CSS}
        .header-info {{ text-align: center; border-bottom: 2px solid #eee; padding-bottom: 1.5rem; margin-bottom: 2rem; }}
        .zhi-badge {{ 
            display: inline-block; background: var(--primary-color); color: white; 
            width: 50px; height: 50px; line-height: 50px; text-align: center;
            border-radius: 50%; font-size: 1.5rem; margin: 0 10px; font-weight: bold;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .relation-mark {{ font-size: 1.5rem; color: #ccc; vertical-align: middle; }}
        
        .layer-title {{ 
            font-size: 1.3rem; font-weight: bold; margin-bottom: 1rem; 
            display: flex; align-items: center; color: var(--primary-color);
            border-bottom: 1px solid #eee; padding-bottom: 10px;
        }}
        .layer-title::before {{
            content: ''; display: inline-block; width: 6px; height: 24px;
            background: var(--accent-color); margin-right: 12px; border-radius: 3px;
        }}
        
        /* 緊湊排版設定 */
        .relation-block {{ 
            background: #faf9f6; padding: 1.5rem; margin-bottom: 1.5rem; 
            border-radius: 8px; border-left: 5px solid #ccc; 
        }}
        .rel-name {{ font-size: 1.3rem; font-weight: bold; margin-bottom: 0.5rem; }}
        
        .rel-good {{ border-left-color: #27ae60; }} .rel-good .rel-name {{ color: #27ae60; }}
        .rel-bad {{ border-left-color: #c0392b; }} .rel-bad .rel-name {{ color: #c0392b; }}
        .rel-warn {{ border-left-color: #d35400; }} .rel-warn .rel-name {{ color: #d35400; }}
        .rel-normal {{ border-left-color: #7f8c8d; }} .rel-normal .rel-name {{ color: #7f8c8d; }}

        /* 內容區塊優化：取消 pre-wrap，改用正常流動排版 */
        .content-body {{ 
            color: #555; font-size: 1rem; line-height: 1.6; 
            max-width: 95%; /* 防止文字過寬難以閱讀 */
        }}
        
        /* 一般文字行距 */
        .fmt_text_line {{
            margin-bottom: 0.4rem; /* 讓每一行字不要黏在一起，但也不要太開 */
            display: block;
        }}

        /* 特殊文字格式 */
        .fmt_module {{ 
            color: #8e44ad; font-weight: bold; font-size: 0.9rem; 
            opacity: 0.8; display: block; margin-bottom: 0.2rem; 
        }}
        .fmt_subhead {{ 
            color: var(--primary-color); font-weight: bold; 
            margin-top: 1.2rem; margin-bottom: 0.5rem; /* 標題與內文的距離 */
            display: block; font-size: 1.05rem;
            border-left: 3px solid #ddd; padding-left: 8px; /* 增加小裝飾讓層次分明 */
        }}
        .fmt_highlight {{ 
            background: #fff3e0; color: #d35400; padding: 6px 12px; 
            border-radius: 4px; font-weight: bold; display: inline-block; 
            margin-top: 1rem; 
        }}

        .btn-secondary {{
            display: block; width: 100%; text-align: center; padding: 1rem; 
            background: #a1887f; color: white; text-decoration: none; border-radius: 8px; 
            margin-top: 2rem; font-size: 1.1rem; box-sizing: border-box;
        }}
        .btn-secondary:hover {{ background: #8d6e63; }}

        /* 🏆 人生攻略區塊 */
        .strategy-card {{
            background: linear-gradient(135deg, #2c3e50 0%, #1a1a1a 100%);
            color: #fff;
            padding: 2.5rem;
            border-radius: var(--radius);
            margin-top: 3rem;
            text-align: center;
            box-shadow: 0 15px 40px rgba(0,0,0,0.3);
            position: relative; overflow: hidden;
        }}
        .strategy-title {{ 
            color: #f1c40f; font-size: 1.8rem; margin-bottom: 1rem; 
            border-bottom: 1px solid rgba(255,255,255,0.2); padding-bottom: 1rem; display: inline-block;
        }}
        .strategy-text {{ font-size: 1.1rem; margin-bottom: 2rem; color: #ddd; line-height: 1.8; }}
        .btn-strategy {{
            background: #f1c40f; color: #333; padding: 12px 35px;
            text-decoration: none; border-radius: 50px; font-weight: bold;
            display: inline-block; transition: all 0.3s;
        }}
        .btn-strategy:hover {{ background: #fff; transform: scale(1.05); }}
    </style>
</head>
<body>
    <div class="container">
        <div class="card">
            <div class="header-info">
                <p style="color:#888; font-size:0.9rem; margin-bottom:10px;">命盤分析結果</p>
                <div>
                    <span class="zhi-badge">{{{{ result.branches.user_day }}}}</span>
                    <span class="relation-mark">×</span>
                    <span class="zhi-badge">{{{{ result.branches.today_day }}}}</span>
                </div>
                <div style="margin-top: 10px; font-size: 0.9rem; color: #666;">
                    今日月令：{{{{ result.branches.today_month }}}}
                </div>
            </div>

            <div class="layer-section">
                <div class="layer-title">今日核心運勢</div>
                {{% for item in result.layer1 %}}
                <div class="relation-block rel-{{{{ item.relation_type }}}}">
                    <div class="rel-name">{{{{ item.relation_name }}}}</div>
                    <div class="content-body">
                        {{% for line in item.content.split('\\n') %}}
                            {{% if line.strip() %}} {{% if '【模組' in line %}}<span class="fmt_module">{{{{ line }}}}</span>
                                {{% elif '定義' in line or '建議' in line or '指引' in line or '提醒' in line %}}<span class="fmt_subhead">{{{{ line }}}}</span>
                                {{% elif '👉' in line %}}<span class="fmt_highlight">{{{{ line }}}}</span>
                                {{% else %}}<span class="fmt_text_line">{{{{ line }}}}</span>{{% endif %}}
                            {{% endif %}}
                        {{% endfor %}}
                    </div>
                </div>
                {{% endfor %}}
            </div>

            <div class="layer-section" style="margin-top: 3rem;">
                <div class="layer-title">本月大環境氣場</div>
                {{% for item in result.layer2 %}}
                <div class="relation-block rel-{{{{ item.relation_type }}}}">
                    <div class="rel-name">{{{{ item.relation_name }}}}</div>
                    <div class="content-body">
                        {{% for line in item.content.split('\\n') %}}
                            {{% if line.strip() %}} {{% if '【模組' in line %}}<span class="fmt_module">{{{{ line }}}}</span>
                                {{% elif '定義' in line or '建議' in line or '指引' in line or '提醒' in line %}}<span class="fmt_subhead">{{{{ line }}}}</span>
                                {{% elif '👉' in line %}}<span class="fmt_highlight">{{{{ line }}}}</span>
                                {{% else %}}<span class="fmt_text_line">{{{{ line }}}}</span>{{% endif %}}
                            {{% endif %}}
                        {{% endfor %}}
                    </div>
                </div>
                {{% endfor %}}
            </div>
        </div>

<div class="strategy-card">
            <div style="font-size: 0.9rem; color: rgba(255,255,255,0.6); letter-spacing: 2px; margin-bottom: 5px;">
                PREMIUM SERVICE
            </div>
            <h2 class="strategy-title">2026 專屬人生攻略白皮書</h2>
            
            <p class="strategy-text" style="text-align: left; max-width: 90%; margin: 0 auto 2rem auto;">
                <strong>你的 2026，不該只有「預測」，更該有「對策」。</strong><br>
                我們將你的人生視為一家公司，為你聘請了四位頂級高管進行全面診斷：<br><br>
                
                💰 <strong>財務長 (CFO)：</strong>資金鍊風控，避開庫存破洞，精準配置資產。<br>
                📣 <strong>行銷長 (CMO)：</strong>個人品牌定位，找出你的貴人畫像與變現路徑。<br>
                🧠 <strong>營運長 (COO)：</strong>內在系統體檢，將情緒轉化為可執行的策略。<br>
                📅 <strong>時間管理大師：</strong>365 天流日導航，避開決策雷區。<br>
                <br>
                <span style="color: #f1c40f; font-weight: bold;">
                    這是一份約 30,000 字的年度戰略報告。<br>
                    別再用「運氣不好」來解釋虧損，拿回你的人生主導權。
                </span>
            </p>
            
            <a href="https://maplife01.netlify.app/" class="btn-strategy">
                立即預購 
            </a>
            <p style="font-size: 0.8rem; color: #888; margin-top: 15px;">
                *本產品為全客製化運算，將於「編碼同學會」後陸續發送。
            </p>
        </div>
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET'])
def index():
    return render_template_string(INDEX_HTML)

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.form

        # 1) 使用者輸入（表單是「民國年」）
        roc_year = int(data.get('year'))
        year = roc_year + 1911 if roc_year < 1911 else roc_year
        month = int(data.get('month'))
        day = int(data.get('day'))
        hour = int(data.get('hour'))
        minute = int(data.get('minute') or 0)

        # 2) 計算「使用者八字」
        user_bazi = calc_bazi_8char(year, month, day, hour, minute)

        # 3) 計算「今日八字」（以 Asia/Taipei 為準；若缺 tzdata 則退回 UTC+8）
        now = now_in_taipei()
        today_bazi = calc_bazi_8char(now.year, now.month, now.day, now.hour, now.minute)

        # 4) 抽取地支：日主地支、今日日支、今日月支
        user_day = user_bazi.day[-1]
        today_day = today_bazi.day[-1]
        today_month = today_bazi.month[-1]

        # 5) 依序丟給 bazi_calc_v2
        if not all(b in ZHI for b in [user_day, today_day, today_month]):
            raise ValueError("地支解析異常（請確認八字輸出是否為「天干地支」兩字組合）")

        result = WebBaziAnalyzer.get_analysis_result(user_day, today_day, today_month)

        # debug：保留四柱方便你檢查
        result["debug_info"] = {
            "user_pillars": [user_bazi.year, user_bazi.month, user_bazi.day, user_bazi.hour],
            "today_pillars": [today_bazi.year, today_bazi.month, today_bazi.day, today_bazi.hour],
            "now_local": now.isoformat(timespec="seconds") if "now" in locals() else None,
        }

        return render_template_string(RESULT_HTML, result=result)

    except Exception as e:
        traceback.print_exc()
        return f"""
        <div style="font-family:sans-serif; text-align:center; padding-top:50px;">
            <h1 style="color:#c0392b;">⚠️ 分析發生中斷</h1>
            <p>原因：{str(e)}</p>
            <p>請按上一頁修正輸入後再試一次。</p>
            <a href="/" style="display:inline-block; margin-top:20px; padding:10px 20px; background:#5d4037; color:white; text-decoration:none; border-radius:5px;">回首頁</a>
        </div>
        """, 500

if __name__ == '__main__':
    # 本機測試用：Render 會用 gunicorn 啟動，不會走到這裡
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
