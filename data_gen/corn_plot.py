import os
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
from matplotlib.patches import Patch

# ── Konfiguration ─────────────────────────────────────────────────
TICKER = "ZC=F"
START = "1970-01-01"
END = "2026-06-12"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "stock_prices")
CACHE_FILE = os.path.join(DATA_DIR, f"{TICKER}_{START}_{END}.csv")

# ── Daten laden (aus Cache oder von Yahoo Finance) ─────────────────
os.makedirs(DATA_DIR, exist_ok=True)

if os.path.exists(CACHE_FILE):
    print(f"Lade Corn-Kursdaten aus lokalem Cache: {CACHE_FILE}")
    df = pd.read_csv(CACHE_FILE, index_col=0, parse_dates=True)
else:
    print("Lade Corn-Kursdaten von Yahoo Finance...")
    df = yf.download(TICKER, start=START, end=END, progress=False)
    df = df[["Close"]].copy()
    df.columns = ["Kurs"]
    df.index = df.index.tz_localize(None)
    df.to_csv(CACHE_FILE)
    print(f"Daten gespeichert unter: {CACHE_FILE}")

# Jahresschlusskurse
jahres = df.resample("YE").last()
jahres.index = jahres.index.year

# ── Tagesweise Konsolenausgabe ───────────────────────────────────
print("\n" + "=" * 65)
print(f"{f'Corn (ZC=F) Tageskurse {df.index[0].year}–{df.index[-1].year}':^65}")
print(f"{'(split-bereinigt, via Yahoo Finance, in USD/bushel)':^65}")
print("=" * 65)
print(f"{'Datum':<14} {'Kurs (USD)':>12}   {'Veränd. (%)':<14} {'Veränd. (USD)'}")
print("-" * 65)

prev = None
for datum, row in df.iterrows():
    kurs = row["Kurs"]
    if prev is None:
        pct_str, abs_str = "      —", "       —"
    else:
        pct = (kurs - prev) / prev * 100
        absd = kurs - prev
        pct_str = f"  +{pct:6.2f}%" if pct >= 0 else f"  {pct:7.2f}%"
        abs_str = f"  +${absd:7.2f}" if absd >= 0 else f"  -${abs(absd):7.2f}"
    print(f"{str(datum.date()):<14} ${kurs:>10.2f}   {pct_str:<14} {abs_str}")
    prev = kurs

print("-" * 65)
gesamt = (df["Kurs"].iloc[-1] - df["Kurs"].iloc[0]) / df["Kurs"].iloc[0] * 100
print(f"{'Gesamtrendite:':<35} +{gesamt:.0f}%")
print(f"{'Aus $1.000 wurden:':<35} ${1000 * df['Kurs'].iloc[-1] / df['Kurs'].iloc[0]:,.0f}")
print("=" * 65)

# ── Jahresübersicht ──────────────────────────────────────────────
kurse_list = jahres["Kurs"].tolist()
jahre_list = jahres.index.tolist()

print("\n" + "=" * 55)
print(f"{'Jahresübersicht':^55}")
print("=" * 55)
print(f"{'Jahr':<8} {'Kurs (USD)':>12}   {'Veränd. (%)':<14} {'Trend'}")
print("-" * 55)
for i, (jahr, kurs) in enumerate(zip(jahre_list, kurse_list)):
    if i == 0:
        pct_str, trend = "      —", ""
    else:
        pct = (kurs - kurse_list[i - 1]) / kurse_list[i - 1] * 100
        pct_str = f"  +{pct:6.2f}%" if pct >= 0 else f"  {pct:7.2f}%"
        trend = "▲" * min(int(abs(pct) / 15) + 1, 5) if pct >= 0 else "▼"
    hinweis = " ← aktuell" if i == len(jahre_list) - 1 else ""
    print(f"{jahr:<8} ${kurs:>10.2f}   {pct_str:<14} {trend}{hinweis}")
print("=" * 55)

# ── Grafik ───────────────────────────────────────────────────────
fig, axes = plt.subplots(
    3, 1, figsize=(14, 10),
    gridspec_kw={"height_ratios": [3, 1, 1]},
    facecolor="#0d1117"
)
ax1, ax2, ax3 = axes
fig.suptitle(
    f"Corn (ZC=F) Futures Price {df.index[0].year}–{df.index[-1].year}  |  Quelle: Yahoo Finance",
    color="white", fontsize=15, fontweight="bold", y=0.99
)

# — ax1: Tageskurs-Linie —
ax1.set_facecolor("#0d1117")
ax1.plot(df.index, df["Kurs"], color="#6aa84f", linewidth=1.0, zorder=3)
ax1.fill_between(df.index, df["Kurs"], alpha=0.12, color="#6aa84f", zorder=2)

# Jahrespunkte einfärben
for i, (jahr, kurs) in enumerate(zip(jahre_list, kurse_list)):
    if i == 0:
        farbe = "#6aa84f"
    else:
        farbe = "#3fb950" if kurs >= kurse_list[i - 1] else "#f85149"
    datum = df[df.index.year == jahr].index[-1]
    ax1.scatter(datum, kurs, color=farbe, s=55, zorder=4,
                edgecolors="white", linewidths=0.5)

# Ereignis-Annotationen (Beispiele)
ereignisse = {
    2012: "US\nDrought",
    2014: "South\nAmerica\nYield",
    2020: "COVID\nDemand",
    2022: "Russia/\nUkraine",
    2023: "Weather\nRisk",
}
for jahr, label in ereignisse.items():
    if jahr in jahres.index:
        kurs = jahres.loc[jahr, "Kurs"]
        datum = df[df.index.year == jahr].index[-1]
        ax1.annotate(
            label, xy=(datum, kurs), xytext=(datum, kurs + max(6, kurs * 0.08)),
            fontsize=8, color="#8b949e", ha="center",
            arrowprops=dict(arrowstyle="-", color="#444c56", lw=0.8),
        )

ax1.set_ylabel("Kurs in USD/bushel", color="#8b949e", fontsize=11)
ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.0f"))
ax1.xaxis.set_major_locator(mdates.YearLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax1.tick_params(colors="#8b949e", rotation=45)
for spine in ax1.spines.values():
    spine.set_edgecolor("#30363d")
ax1.grid(axis="y", color="#21262d", linewidth=0.8)
ax1.set_axisbelow(True)

# — ax2: tägliche Veränderung in % —
ax2.set_facecolor("#0d1117")
tages_pct = df["Kurs"].pct_change() * 100
farben_tag = ["#3fb950" if v >= 0 else "#f85149" for v in tages_pct]
ax2.bar(df.index, tages_pct, color=farben_tag, width=1.5, zorder=3)
ax2.axhline(0, color="#444c56", linewidth=0.8)
ax2.set_ylabel("Tages-\nveränd. (%)", color="#8b949e", fontsize=9)
ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
ax2.set_ylim(tages_pct.quantile(0.005), tages_pct.quantile(0.995))
ax2.xaxis.set_major_locator(mdates.YearLocator())
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax2.tick_params(colors="#8b949e", rotation=45)
for spine in ax2.spines.values():
    spine.set_edgecolor("#30363d")
ax2.grid(axis="y", color="#21262d", linewidth=0.8)
ax2.set_axisbelow(True)

# — ax3: jährliche Veränderung —
ax3.set_facecolor("#0d1117")
farben_j, werte_j, pos_j = [], [], []
for i in range(1, len(jahre_list)):
    pct = (kurse_list[i] - kurse_list[i - 1]) / kurse_list[i - 1] * 100
    datum = df[df.index.year == jahre_list[i]].index[-1]
    pos_j.append(datum)
    werte_j.append(pct)
    farben_j.append("#3fb950" if pct >= 0 else "#f85149")

ax3.bar(pos_j, werte_j, color=farben_j, width=200, zorder=3)
ax3.axhline(0, color="#444c56", linewidth=0.8)
ax3.set_ylabel("Jahres-\nveränd. (%)", color="#8b949e", fontsize=9)
ax3.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
ax3.xaxis.set_major_locator(mdates.YearLocator())
ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax3.tick_params(colors="#8b949e", rotation=45)
for spine in ax3.spines.values():
    spine.set_edgecolor("#30363d")
ax3.grid(axis="y", color="#21262d", linewidth=0.8)
ax3.set_axisbelow(True)

# Legende
ax1.legend(
    handles=[
        Patch(color="#3fb950", label="Positives Jahr"),
        Patch(color="#f85149", label="Negatives Jahr"),
        Patch(color="#6aa84f", label="Tageskurs"),
    ],
    loc="upper left", facecolor="#161b22",
    edgecolor="#30363d", labelcolor="white", fontsize=9
)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()
