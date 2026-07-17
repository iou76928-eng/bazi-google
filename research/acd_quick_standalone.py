from __future__ import annotations

import itertools
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

NY = "America/New_York"
DATA_URL = (
    "https://raw.githubusercontent.com/TheSnowGuru/"
    "Stocks-Futures-Financial-Time-series-Tick-Bar-Data/main/"
    "commodities/gold/XAUUSD_M5.csv"
)
OUT = Path("artifacts_execution_quick")
OUT.mkdir(exist_ok=True)


@dataclass(frozen=True)
class Config:
    entry_mode: str
    stop_mode: str
    rr: float
    breakeven_r: float | None
    force_exit: str
    cost_points: float = 0.25


@dataclass
class Trade:
    date: str
    entry_time: str
    exit_time: str
    entry: float
    exit: float
    stop: float
    target: float
    risk: float
    net_points: float
    r_net: float
    exit_reason: str


@dataclass
class Day:
    date: str
    frame: pd.DataFrame
    atr: float
    or_high: float
    or_low: float
    or_mid: float
    entry_pos: np.ndarray
    last_1200: int | None
    last_1600: int | None


def metrics(trades):
    ts = list(trades)
    if not ts:
        return dict(trades=0, win_rate=np.nan, profit_factor=np.nan, net_points=0.0, avg_r=np.nan, max_dd_r=np.nan, max_consecutive_losses=0)
    pnl = np.array([t.net_points for t in ts], dtype=float)
    rs = np.array([t.r_net for t in ts], dtype=float)
    gp = pnl[pnl > 0].sum()
    gl = -pnl[pnl < 0].sum()
    equity = np.cumsum(rs)
    peak = np.maximum.accumulate(np.r_[0.0, equity])
    dd = peak[1:] - equity
    max_losing = cur = 0
    for x in pnl:
        if x < 0:
            cur += 1
            max_losing = max(max_losing, cur)
        else:
            cur = 0
    return dict(
        trades=len(ts), win_rate=float((pnl > 0).mean() * 100),
        profit_factor=float(gp / gl) if gl > 0 else np.inf,
        net_points=float(pnl.sum()), avg_r=float(rs.mean()),
        max_dd_r=float(dd.max()) if len(dd) else 0.0,
        max_consecutive_losses=max_losing,
    )


def load_data():
    raw = pd.read_csv(DATA_URL, sep="\t")
    raw.columns = [str(c).strip().lower() for c in raw.columns]
    raw["time"] = pd.to_datetime(raw["time"], utc=True, errors="coerce")
    raw = raw.dropna(subset=["time", "open", "high", "low", "close"]).set_index("time").sort_index()
    raw = raw[~raw.index.duplicated(keep="last")]
    daily = raw[["high", "low", "close"]].resample("1D").agg({"high":"max","low":"min","close":"last"}).dropna()
    prev = daily.close.shift(1)
    tr = pd.concat([daily.high-daily.low,(daily.high-prev).abs(),(daily.low-prev).abs()],axis=1).max(axis=1)
    atr = tr.rolling(14,min_periods=14).mean()
    atr.index = pd.Index([x.date() for x in atr.index])
    intraday = raw[["open","high","low","close","volume"]].copy()
    intraday.index = intraday.index.tz_convert(NY)
    intraday["ema200"] = intraday.close.ewm(span=200,adjust=False).mean()
    return intraday, atr.dropna()


def positions(index, start, end):
    sh,sm=map(int,start.split(":")); eh,em=map(int,end.split(":"))
    mins=index.hour*60+index.minute
    return np.flatnonzero((mins>=sh*60+sm)&(mins<eh*60+em))


def prepare(intraday, atr):
    adates=np.array(list(atr.index),dtype=object); avals=atr.to_numpy(float)
    days=[]
    for d,frame in intraday.groupby(intraday.index.date,sort=True):
        if pd.Timestamp(d).weekday() not in {1,2,3,4}:
            continue
        ai=int(np.searchsorted(adates,d,side="left"))-1
        if ai<0: continue
        da=float(avals[ai])
        frame=frame.sort_index()
        op=positions(frame.index,"09:30","09:45")
        if len(op)<2: continue
        orb=frame.iloc[op]; oh=float(orb.high.max()); ol=float(orb.low.min())
        if not 0.05 <= (oh-ol)/da <= 0.35: continue
        ep=positions(frame.index,"09:45","12:00")
        p12=positions(frame.index,"09:45","12:00"); p16=positions(frame.index,"09:45","16:00")
        days.append(Day(str(d),frame,da,oh,ol,(oh+ol)/2,ep,int(p12[-1]) if len(p12) else None,int(p16[-1]) if len(p16) else None))
    return days


def find_entry(day: Day, level: float, mode: str):
    f=day.frame; ps=day.entry_pos
    if mode=="touch":
        for pos in ps:
            pos=int(pos)
            if pos<=int(ps[0]): continue
            prev=f.iloc[pos-1]; row=f.iloc[pos]
            if prev.close>prev.ema200 and row.high>=level:
                return pos,max(float(level),float(row.open)),float(row.low)
        return None
    if mode in {"close1","close2"}:
        need=1 if mode=="close1" else 2; streak=0
        for pos in ps:
            pos=int(pos); row=f.iloc[pos]
            streak=streak+1 if row.close>level and row.close>row.ema200 else 0
            if streak>=need:
                ep=pos+1
                if ep>=len(f) or ep>int(ps[-1]): return None
                return ep,float(f.iloc[ep].open),float(row.low)
        return None
    if mode=="retest":
        br=None
        for pos in ps:
            pos=int(pos); row=f.iloc[pos]
            if br is None:
                if row.close>level and row.close>row.ema200: br=pos
                continue
            if pos>br+6: return None
            if row.low<=level+day.atr*0.01 and row.close>level and row.close>row.ema200:
                ep=pos+1
                if ep>=len(f) or ep>int(ps[-1]): return None
                return ep,float(f.iloc[ep].open),float(row.low)
        return None
    raise ValueError(mode)


def simulate(day: Day, cfg: Config, ep: int, entry: float, signal_low: float):
    stop = day.or_low if cfg.stop_mode=="far" else day.or_mid if cfg.stop_mode=="mid" else min(signal_low,entry-1e-9)
    risk=entry-stop
    if risk<=0 or not np.isfinite(risk): return None
    target=entry+risk*cfg.rr
    last=day.last_1200 if cfg.force_exit=="12:00" else day.last_1600
    if last is None or ep>last: return None
    active=stop; armed=False; xp=float(day.frame.iloc[last].close); xpos=last; reason="TIME"
    for pos in range(ep,last+1):
        row=day.frame.iloc[pos]
        if row.low<=active:
            xp=active; xpos=pos; reason="BE" if active>=entry else "STOP"; break
        if row.high>=target:
            xp=target; xpos=pos; reason="TP"; break
        if cfg.breakeven_r is not None and not armed and row.high>=entry+risk*cfg.breakeven_r:
            armed=True
        elif armed:
            active=max(active,entry)
    net=xp-entry-cfg.cost_points
    return Trade(day.date,day.frame.index[ep].isoformat(),day.frame.index[xpos].isoformat(),entry,xp,stop,target,risk,net,net/risk,reason)


def backtest(days,cfg):
    out=[]
    for d in days:
        found=find_entry(d,d.or_high+d.atr*0.05,cfg.entry_mode)
        if found is None: continue
        t=simulate(d,cfg,*found)
        if t is not None: out.append(t)
    return out


def period(trades,y0=None,y1=None):
    return metrics([t for t in trades if (y0 is None or int(t.date[:4])>=y0) and (y1 is None or int(t.date[:4])<=y1)])


def main():
    intraday,atr=load_data(); days=prepare(intraday,atr)
    configs=[Config(em,sm,rr,be,fx) for em,sm,rr,be,fx in itertools.product(
        ["touch","close1","close2","retest"],["far","mid","signal_low"],[0.75,1.0,1.25,1.5],[None,0.75,1.0],["12:00","16:00"])]
    rows=[]
    for cfg in configs:
        ts=backtest(days,cfg); row=asdict(cfg)
        row.update({f"disc_{k}":v for k,v in period(ts,None,2021).items()})
        row.update({f"y2022_{k}":v for k,v in period(ts,2022,2022).items()})
        row.update({f"y2023_{k}":v for k,v in period(ts,2023,2023).items()})
        row.update({f"all_{k}":v for k,v in metrics(ts).items()}); rows.append(row)
    df=pd.DataFrame(rows); df["disc_score"]=df.disc_avg_r.clip(-1,2)*np.sqrt(df.disc_trades.clip(lower=0))-0.025*df.disc_max_dd_r
    df.to_csv(OUT/"quick_grid.csv",index=False)
    ranked=df[(df.disc_trades>=40)&np.isfinite(df.disc_profit_factor)&(df.disc_profit_factor>1)&(df.disc_avg_r>0)].sort_values(["disc_score","disc_profit_factor"],ascending=False)
    robust=ranked[(ranked.y2022_trades>=15)&(ranked.y2023_trades>=12)&(ranked.y2022_profit_factor>1)&(ranked.y2023_profit_factor>1)&(ranked.y2022_avg_r>0)&(ranked.y2023_avg_r>0)]
    robust.to_csv(OUT/"quick_validation_positive.csv",index=False)
    cols=["entry_mode","stop_mode","rr","breakeven_r","force_exit","disc_trades","disc_profit_factor","disc_avg_r","disc_max_dd_r","y2022_trades","y2022_profit_factor","y2022_avg_r","y2023_trades","y2023_profit_factor","y2023_avg_r","all_trades","all_win_rate","all_profit_factor","all_avg_r","all_max_dd_r"]
    text=["# Corrected ACD Execution Test","","No look-ahead: touch orders use the previous completed bar and activate on the next bar; gap-through fills use the open.","","## Top discovery-ranked","",ranked.head(15)[cols].to_markdown(index=False,floatfmt=".3f"),"","## Positive in 2022 and 2023","",f"Count: {len(robust)}","",robust.head(30)[cols].to_markdown(index=False,floatfmt=".3f") if len(robust) else "None."]
    (OUT/"quick_report.md").write_text("\n".join(text),encoding="utf-8")
    print((OUT/"quick_report.md").read_text(encoding="utf-8"))


if __name__=="__main__": main()
