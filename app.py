"""
IB Toolkit — Flask Backend
===========================
pip install flask yfinance pandas numpy
python app.py
"""

from flask import Flask, jsonify, request, send_from_directory
import yfinance as yf
import numpy as np
import os

app = Flask(__name__, static_folder="static")

# ── Serve frontend ──────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory("static", "index.html")

# ── Helpers ─────────────────────────────────────────────────
def sf(v):
    try:
        f = float(v)
        return None if (f != f) else f   # NaN check
    except:
        return None

def est_growth(hist, yrs=5):
    d = [f for f in hist if f and f == f]
    if len(d) < 2: return 0.08
    d = d[-(yrs+1):]
    s, e, n = d[0], d[-1], len(d)-1
    if s <= 0 or e <= 0:
        pos = [f for f in d if f > 0]
        return ((pos[-1]/pos[0])**(1/(len(pos)-1))-1) if len(pos)>=2 else 0.08
    return max(min((e/s)**(1/n)-1, 0.40), -0.15)

def run_dcf_calc(bf, gr, wacc, tgr, yrs, nd, sh):
    yl = list(range(1, yrs+1))
    pf = [bf*(1+gr)**y for y in yl]
    df = [f/(1+wacc)**y for y,f in zip(yl,pf)]
    tv = pf[-1]*(1+tgr)/(wacc-tgr)
    dtv = tv/(1+wacc)**yrs
    pv = sum(df); ev = pv+dtv; eq = ev-nd
    return {
        "years": yl, "proj_fcf": pf, "disc_fcf": df,
        "pv_fcf": pv, "terminal_val": tv, "disc_terminal": dtv,
        "enterprise_value": ev, "equity_value": eq,
        "intrinsic_per_share": eq/sh if sh else 0
    }

# ── DCF endpoint ─────────────────────────────────────────────
@app.route("/api/dcf")
def api_dcf():
    ticker = request.args.get("ticker","AAPL").upper()
    wacc   = float(request.args.get("wacc", 0.10))
    tgr    = float(request.args.get("tgr",  0.025))
    yrs    = int(request.args.get("yrs",    5))
    mos    = float(request.args.get("mos",  0.20))
    override_gr = request.args.get("gr")

    try:
        tk   = yf.Ticker(ticker)
        info = tk.info
        if not info or not info.get("regularMarketPrice"):
            return jsonify({"error": f"No data found for '{ticker}'"}), 404

        cashflow = tk.cashflow
        balance  = tk.balance_sheet

        fcf_history, fcf_years = [], []
        for ro, rc in [("Operating Cash Flow","Capital Expenditure"),
                        ("Total Cash From Operating Activities","Capital Expenditures")]:
            try:
                raw = (cashflow.loc[ro] + cashflow.loc[rc]).dropna()
                fcf_history = raw.values[::-1].tolist()
                fcf_years   = [str(d.year) for d in raw.index[::-1]]
                break
            except: continue

        try:    debt = float(balance.loc["Total Debt"].iloc[0])
        except: debt = float(info.get("totalDebt",0) or 0)
        try:    cash = float(balance.loc["Cash And Cash Equivalents"].iloc[0])
        except: cash = float(info.get("totalCash",0) or 0)

        shares = info.get("sharesOutstanding") or info.get("impliedSharesOutstanding") or 1
        net_debt = debt - cash

        if not fcf_history:
            return jsonify({"error": "No FCF data available for this ticker"}), 404

        bf = fcf_history[-1]
        if bf <= 0:
            pos = [f for f in fcf_history if f > 0]
            bf = float(np.mean(pos)) if pos else 0
        if bf <= 0:
            return jsonify({"error": "No positive FCF — cannot run DCF"}), 400
        if wacc <= tgr:
            return jsonify({"error": "WACC must be greater than terminal growth rate"}), 400

        gr = float(override_gr) if override_gr else est_growth(fcf_history)
        res = run_dcf_calc(bf, gr, wacc, tgr, yrs, net_debt, shares)

        # Sensitivity matrix
        wr = [max(wacc-0.02,0.01), max(wacc-0.01,0.01), wacc, wacc+0.01, wacc+0.02]
        tr = [t for t in [0.010,0.015,0.020,0.025,0.030,0.035,0.040] if t < wacc]
        matrix = []
        for t in tr:
            row = []
            for w in wr:
                if w > t:
                    r2 = run_dcf_calc(bf,gr,w,t,yrs,net_debt,shares)
                    row.append(round(r2["intrinsic_per_share"],2))
                else:
                    row.append(None)
            matrix.append(row)

        return jsonify({
            "ticker": ticker,
            "name": info.get("longName", ticker),
            "sector": info.get("sector","N/A"),
            "industry": info.get("industry","N/A"),
            "current_price": sf(info.get("regularMarketPrice") or info.get("currentPrice")),
            "market_cap": sf(info.get("marketCap")),
            "pe_ratio": sf(info.get("trailingPE")),
            "ev_ebitda": sf(info.get("enterpriseToEbitda")),
            "beta": sf(info.get("beta")),
            "analyst_target": sf(info.get("targetMeanPrice")),
            "description": info.get("longBusinessSummary","")[:500],
            "fcf_history": fcf_history,
            "fcf_years": fcf_years,
            "growth_rate": round(gr, 4),
            "net_debt": net_debt,
            "intrinsic": round(res["intrinsic_per_share"], 2),
            "mos_price": round(res["intrinsic_per_share"]*(1-mos), 2),
            "upside": round((res["intrinsic_per_share"] - (sf(info.get("regularMarketPrice")) or 1)) / (sf(info.get("regularMarketPrice")) or 1) * 100, 1),
            "proj_fcf": [round(f,0) for f in res["proj_fcf"]],
            "disc_fcf": [round(f,0) for f in res["disc_fcf"]],
            "pv_fcf": round(res["pv_fcf"],0),
            "disc_terminal": round(res["disc_terminal"],0),
            "enterprise_value": round(res["enterprise_value"],0),
            "equity_value": round(res["equity_value"],0),
            "years": res["years"],
            "sensitivity_matrix": matrix,
            "sensitivity_wacc": [round(w*100,1) for w in wr],
            "sensitivity_tgr":  [round(t*100,1) for t in tr],
            "mos": mos,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── Comps endpoint ────────────────────────────────────────────
@app.route("/api/comps")
def api_comps():
    tickers_raw = request.args.get("tickers","AAPL,MSFT,GOOGL")
    tickers = [t.strip().upper() for t in tickers_raw.split(",")]
    rows = []
    for t in tickers:
        try:
            info = yf.Ticker(t).info
            mc = sf(info.get("marketCap")); ev = sf(info.get("enterpriseValue"))
            eb = sf(info.get("ebitda")); rv = sf(info.get("totalRevenue"))
            pe = sf(info.get("trailingPE")); fp = sf(info.get("forwardPE"))
            db = sf(info.get("totalDebt")); fc = sf(info.get("freeCashflow"))
            pr = sf(info.get("regularMarketPrice") or info.get("currentPrice"))
            rows.append({
                "ticker": t,
                "company": info.get("longName", t)[:28],
                "price": pr,
                "market_cap": mc/1e9 if mc else None,
                "ev_ebitda": ev/eb if ev and eb and eb>0 else None,
                "ev_revenue": ev/rv if ev and rv and rv>0 else None,
                "pe": pe, "pe_fwd": fp,
                "debt_ebitda": db/eb if db and eb and eb>0 else None,
                "fcf_yield": fc/mc if fc and mc and mc>0 else None,
            })
        except: continue

    # Medians
    def med(key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return round(float(np.median(vals)), 2) if vals else None

    return jsonify({
        "rows": rows,
        "medians": {k: med(k) for k in ["ev_ebitda","ev_revenue","pe","pe_fwd","debt_ebitda","fcf_yield"]}
    })

# ── 3-Statement endpoint ──────────────────────────────────────
@app.route("/api/statements")
def api_statements():
    ticker    = request.args.get("ticker","AAPL").upper()
    rev_gr    = float(request.args.get("rev_gr",   0.08))
    op_margin = float(request.args.get("op_margin",0.20))
    tax_rate  = float(request.args.get("tax_rate", 0.21))
    capex_pct = float(request.args.get("capex_pct",0.05))
    da_pct    = float(request.args.get("da_pct",   0.04))
    nwc_pct   = float(request.args.get("nwc_pct",  0.01))

    try:
        tk   = yf.Ticker(ticker)
        info = tk.info
        inc  = tk.income_stmt
        bal  = tk.balance_sheet
        cf   = tk.cashflow

        def extract(df, rows):
            try:
                df2 = df.copy()
                df2.columns = [str(c.year) for c in df2.columns]
                df2 = (df2/1e9).round(2)
                result = {}
                for r in rows:
                    if r in df2.index:
                        result[r] = {yr: (None if (v!=v) else round(float(v),2))
                                     for yr, v in df2.loc[r].items()}
                return result
            except: return {}

        inc_rows = ["Total Revenue","Gross Profit","Operating Income","Net Income","EBITDA"]
        cf_rows  = ["Operating Cash Flow","Capital Expenditure","Total Cash From Operating Activities","Capital Expenditures"]
        bal_rows = ["Total Assets","Total Liabilities Net Minority Interest","Stockholders Equity","Total Debt","Cash And Cash Equivalents"]

        # Base revenue for forecast
        br = None
        for rn in ["Total Revenue","Revenue"]:
            try: br = float(inc.loc[rn].iloc[0]); break
            except: continue

        forecast = []
        if br and br==br:
            r = br
            for i in range(5):
                r *= (1+rev_gr); oi = r*op_margin; eb = oi+r*da_pct
                ni = oi*(1-tax_rate); fc = ni+r*da_pct-r*capex_pct-r*nwc_pct
                forecast.append({
                    "year": f"Y+{i+1}",
                    "revenue": round(r/1e9,2), "op_income": round(oi/1e9,2),
                    "ebitda": round(eb/1e9,2), "net_income": round(ni/1e9,2),
                    "fcf": round(fc/1e9,2)
                })

        # Sensitivity
        rr = [x/100 for x in np.arange(max(rev_gr*100-6,1), rev_gr*100+8, 2)]
        mr = [x/100 for x in np.arange(max(op_margin*100-8,1), op_margin*100+10, 2)]
        sens, sens_x, sens_y = [], [f"{x*100:.1f}%" for x in rr], [f"{x*100:.1f}%" for x in mr]
        for mg in mr:
            row = []
            for rg in rr:
                rv = br if br else 0
                for _ in range(5): rv *= (1+rg)
                row.append(round((rv*mg*(1-tax_rate)+rv*da_pct-rv*capex_pct-rv*nwc_pct)/1e9, 2))
            sens.append(row)

        return jsonify({
            "name": info.get("longName", ticker),
            "income": extract(inc, inc_rows),
            "cashflow": extract(cf, cf_rows),
            "balance": extract(bal, bal_rows),
            "forecast": forecast,
            "sensitivity": sens,
            "sensitivity_x": sens_x,
            "sensitivity_y": sens_y,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── Screener endpoint ─────────────────────────────────────────
@app.route("/api/screener")
def api_screener():
    max_de  = float(request.args.get("max_de",  3.0))
    min_fy  = float(request.args.get("min_fy",  0.05))
    max_pe  = float(request.args.get("max_pe",  25.0))
    min_cap = float(request.args.get("min_cap", 10.0))
    n       = int(request.args.get("n", 50))

    SP500 = ["AAPL","MSFT","GOOGL","AMZN","NVDA","META","BRK-B","LLY","AVGO","JPM",
             "TSLA","UNH","XOM","V","MA","JNJ","PG","HD","COST","MRK",
             "ABBV","CVX","KO","PEP","ADBE","WMT","CRM","BAC","TMO","ORCL",
             "MCD","CSCO","ACN","ABT","NKE","LIN","DHR","NEE","PM","IBM",
             "RTX","QCOM","T","LOW","UPS","GE","CAT","SPGI","MS","BLK",
             "INTU","ISRG","AMGN","SYK","GS","AXP","DE","MDLZ","ADI","REGN",
             "PLD","CI","TJX","MMC","VRTX","CB","HUM","BSX","NOW","ZTS",
             "C","MO","GILD","EOG","COP","SLB","USB","WFC","PNC","TGT",
             "F","GM","BA","MMM","DIS","NFLX","PYPL","INTC","AMD","TXN"]

    results = []
    for t in SP500[:n]:
        try:
            info = yf.Ticker(t).info
            mc = sf(info.get("marketCap"))
            if not mc or mc < min_cap*1e9: continue
            eb = sf(info.get("ebitda")); db = sf(info.get("totalDebt"))
            fc = sf(info.get("freeCashflow")); pe = sf(info.get("trailingPE"))
            ev = sf(info.get("enterpriseValue"))
            pr = sf(info.get("regularMarketPrice") or info.get("currentPrice"))
            de = db/eb if db and eb and eb>0 else None
            fy = fc/mc if fc and mc and mc>0 else None
            ee = ev/eb if ev and eb and eb>0 else None
            if fy is None or de is None: continue
            if de > max_de: continue
            if fy < min_fy: continue
            if pe and pe > max_pe: continue
            results.append({
                "ticker": t, "company": info.get("longName",t)[:30],
                "sector": info.get("sector","N/A"), "price": pr,
                "market_cap": round(mc/1e9,1), "pe": pe,
                "ev_ebitda": round(ee,1) if ee else None,
                "debt_ebitda": round(de,2), "fcf_yield": round(fy*100,1),
            })
        except: continue

    results.sort(key=lambda x: x["fcf_yield"], reverse=True)
    return jsonify({"results": results, "count": len(results)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
