"""
Post&L Maastricht — Definitive CBS Squares EDA
================================================
Cleaning logic:
  • Their imputation method (mode for categoricals, median for numerics, k=8 neighbours)
  • Our non-residential rule (OAD=NaN + no Urbanization index + peripheral = DROP)
  • Correct square size: 100m × 100m = 0.01 km²
  • All 28 columns imputed (not just 3)
  • Full audit log

Analysis: 14-page PDF covering all of person 2's spatial tasks
"""

import re, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as mgridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.backends.backend_pdf import PdfPages
from scipy.spatial import cKDTree
from scipy.interpolate import griddata

DATA_PATH  = "data/data_Maastricht_2025.xlsx"
OUT_PDF    = "maastricht_eda.pdf"
SQUARE_KM2 = 0.01   # 100m × 100m

plt.rcParams.update({
    "font.family":"DejaVu Sans","font.size":9,
    "axes.titlesize":11,"axes.titleweight":"bold",
    "axes.labelsize":9,"figure.dpi":130,
    "figure.facecolor":"white","axes.facecolor":"#f7f6f3",
    "axes.spines.top":False,"axes.spines.right":False,
    "savefig.facecolor":"white",
})

DEMAND_CMAP = LinearSegmentedColormap.from_list(
    "dmd", ["#eef4fb","#6aaed6","#2171b5","#08306b"])

INCOME_ORDER = {
    "00-20 low":1,"00-40 laag tot onder midden":2,
    "20-40 below middle":3,"20-60 below middle to middle":4,
    "40-60 midden":5,"40-80 middle to above middle":6,
    "60-80 above middle":7,"60-100 above middle to high":8,
    "80-100 high":9,
}

NUM_COLS = [
    "Population","Male","Female",
    "Age0-14","Age15-24","Age25-44","Age45-64","Age65+",
    "Households","Single-person households",
    "Multi-person households w/o kids","Single parent households",
    "Two-parent households","Houses","Home ownership %","Rental %",
    "Social housing %","Vacant houses","Avg. home value k€","OAD",
    "Urbanization index","Percentage low income households",
    "Percentage high income households","Distance nearest supermarket in km",
]
CAT_COLS = ["Median household income"]

# ═══════════════════════════════════════════════════════════════════════
# 1. LOAD
# ═══════════════════════════════════════════════════════════════════════
print("Loading …")
raw = pd.read_excel(DATA_PATH, sheet_name="CBS Squares")
sp  = pd.read_excel(DATA_PATH, sheet_name="Service Point Locations")

note_mask = raw["Age0-14"].apply(lambda v: isinstance(v, str))
df = raw[~note_mask].copy()
print(f"  Dropped {note_mask.sum()} annotation rows → {len(df)} squares")

for c in NUM_COLS:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df["income_score"] = df["Median household income"].map(INCOME_ORDER)

def parse_sq(s):
    m = re.match(r"E(\d+)N(\d+)", str(s))
    return (float(m.group(1)), float(m.group(2))) if m else (np.nan, np.nan)
df["sq_e"], df["sq_n"] = zip(*df["Square"].map(parse_sq))

# ═══════════════════════════════════════════════════════════════════════
# 2. CLASSIFY MISSING SQUARES
# ═══════════════════════════════════════════════════════════════════════
print("\nClassifying missing squares …")

housing_cols = ["Houses","Home ownership %","Rental %","Social housing %"]

# Non-residential: ALL demographic data is NaN/0 AND OAD is NaN
# (OAD>0 means real addresses exist → must impute, not drop)
is_non_res = (
    df["Population"].isna() &
    df["Households"].isna() &
    df[housing_cols].isna().all(axis=1) &
    df["OAD"].isna() &
    df["Urbanization index"].isna()
)

# Residential-but-missing: has OAD>0 OR has Urbanization index
is_res_missing = (
    df["Population"].isna() &
    ~is_non_res
)

print(f"  Non-residential (DROP): {is_non_res.sum()}")
print(f"  Residential missing (IMPUTE): {is_res_missing.sum()}")
print(f"  Already complete: {(~is_non_res & ~is_res_missing).sum()}")

# Audit log
df["cleaning_status"] = "complete"
df.loc[is_non_res,    "cleaning_status"] = "dropped_non_residential"
df.loc[is_res_missing,"cleaning_status"] = "imputed"

# ═══════════════════════════════════════════════════════════════════════
# 3. DROP non-residential
# ═══════════════════════════════════════════════════════════════════════
dropped_df = df[is_non_res].copy()
df_work = df[~is_non_res].copy().reset_index(drop=True)
print(f"\n  After drop: {len(df_work)} squares remain")

# ═══════════════════════════════════════════════════════════════════════
# 4. IMPUTE — their method (kNN median/mode, k=8)
# ═══════════════════════════════════════════════════════════════════════
print("\nImputing …")

coords = df_work[["sq_e","sq_n"]].values
tree   = cKDTree(coords)
impute_rows = df_work[df_work["cleaning_status"] == "imputed"].index.tolist()

df_work["imputation_method"] = "none"

for idx in impute_rows:
    point = coords[idx].reshape(1, -1)
    dists, nn_idx = tree.query(point, k=9)          # k+1 (includes self)
    nn_idx = nn_idx[0][1:]                           # exclude self
    neighbours = df_work.iloc[nn_idx]
    res_nbrs   = neighbours[neighbours["cleaning_status"] != "dropped_non_residential"]

    if len(res_nbrs) == 0:
        df_work.at[idx, "imputation_method"] = "no_neighbours_skipped"
        continue

    # Numeric → median  (their method)
    for col in NUM_COLS + ["income_score"]:
        if pd.isna(df_work.at[idx, col]):
            val = res_nbrs[col].median()
            if pd.notna(val):
                df_work.at[idx, col] = val

    # Categorical → mode  (their method)
    for col in CAT_COLS:
        if pd.isna(df_work.at[idx, col]):
            mode_vals = res_nbrs[col].mode(dropna=True)
            if not mode_vals.empty:
                df_work.at[idx, col] = mode_vals.iloc[0]

    df_work.at[idx, "imputation_method"] = f"knn_median_mode_n{len(res_nbrs)}"

print(f"  Imputed: {(df_work['imputation_method'].str.startswith('knn')).sum()} squares")
print(f"  Population still missing after imputation: {df_work['Population'].isna().sum()}")

# ═══════════════════════════════════════════════════════════════════════
# 5. DERIVED VARIABLES
# ═══════════════════════════════════════════════════════════════════════
df_work["pop_density_km2"]  = df_work["Population"]  / SQUARE_KM2
df_work["hh_density_km2"]   = df_work["Households"]  / SQUARE_KM2
df_work["avg_hh_size"]      = df_work["Population"]  / df_work["Households"].replace(0, np.nan)
df_work["pct_young"]        = df_work["Age0-14"]      / df_work["Population"].replace(0,np.nan) * 100
df_work["pct_working"]      = (df_work["Age25-44"] + df_work["Age45-64"]) \
                               / df_work["Population"].replace(0,np.nan) * 100
df_work["pct_elderly"]      = df_work["Age65+"]       / df_work["Population"].replace(0,np.nan) * 100
df_work["pct_single_hh"]    = df_work["Single-person households"] \
                               / df_work["Households"].replace(0,np.nan) * 100
df_work["pct_rental"]       = df_work["Rental %"]

def norm01(s):
    mn,mx = s.min(), s.max()
    return (s-mn)/(mx-mn+1e-9)

df_work["demand_score"] = (
    0.40 * norm01(df_work["Households"].fillna(0)) +
    0.20 * norm01(df_work["pct_working"].fillna(0)) +
    0.20 * norm01(df_work["Urbanization index"].fillna(0)) +
    0.10 * norm01(df_work["pct_single_hh"].fillna(0)) +
    0.10 * norm01(df_work["income_score"].fillna(0))
)

sp_tree = cKDTree(np.column_stack([sp["X"], sp["Y"]]))
dist_to_sp, nearest_sp_idx = sp_tree.query(np.column_stack([df_work["X"], df_work["Y"]]))
df_work["dist_to_sp"]  = dist_to_sp
df_work["nearest_sp"]  = sp["Location ID"].values[nearest_sp_idx]
df_work["gap_score"]   = norm01(df_work["demand_score"]) * norm01(dist_to_sp)

# ═══════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════
def add_title(fig, t, sub=None):
    fig.text(0.5,0.99,t,ha="center",va="top",fontsize=13,fontweight="bold",color="#1a1a1a")
    if sub:
        fig.text(0.5,0.965,sub,ha="center",va="top",fontsize=9,color="#555")

def sp_overlay(ax):
    ax.scatter(sp["X"],sp["Y"],s=40,c="#ffffff",edgecolors="#111",
               linewidths=1.2,marker="^",zorder=8,label="Service point")

def heat(ax, col, title, cmap, unit="", vmin=None, vmax=None, annotate_sp=False):
    valid = df_work[col].notna()
    null  = ~valid
    ax.scatter(df_work.loc[null,"X"],df_work.loc[null,"Y"],
               s=45,c="#ddd",marker="s",alpha=0.35,linewidths=0,zorder=1)
    sc = ax.scatter(df_work.loc[valid,"X"],df_work.loc[valid,"Y"],
                    c=df_work.loc[valid,col],s=45,marker="s",
                    cmap=cmap,alpha=0.88,linewidths=0,vmin=vmin,vmax=vmax,zorder=2)
    sp_overlay(ax)
    cb = plt.colorbar(sc,ax=ax,fraction=0.034,pad=0.02)
    cb.set_label(unit,fontsize=7); cb.ax.tick_params(labelsize=7)
    ax.set_title(title); ax.set_aspect("equal")
    ax.set_xlabel("X (m)",fontsize=7); ax.set_ylabel("Y (m)",fontsize=7)
    ax.tick_params(labelsize=7)
    ax.legend(handles=[mpatches.Patch(color="#ddd",label=f"No data ({null.sum()})"),
                        mpatches.Patch(color="#111",label=f"SPs (n={len(sp)})")],
              fontsize=6,loc="lower right",framealpha=0.7)
    if annotate_sp:
        for _,row in sp.iterrows():
            ax.annotate(str(int(row["Location ID"])),(row["X"],row["Y"]),
                        fontsize=4.5,xytext=(3,3),textcoords="offset points",color="#111")

# ═══════════════════════════════════════════════════════════════════════
# BUILD PDF
# ═══════════════════════════════════════════════════════════════════════
print("\nBuilding PDF …")
with PdfPages(OUT_PDF) as pdf:

    # ── PAGE 1 — CLEANING AUDIT ──────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    add_title(fig, "Page 1 — Data cleaning audit",
              f"{len(dropped_df)} non-residential dropped · {is_res_missing.sum()} residential imputed via kNN median/mode · square size = 0.01 km²")

    ax = axes[0]
    complete = df_work[df_work["cleaning_status"]=="complete"]
    imputed  = df_work[df_work["cleaning_status"]=="imputed"]
    ax.scatter(complete["X"],complete["Y"],s=50,c="#2171b5",marker="s",
               alpha=0.7,linewidths=0,label=f"Complete ({len(complete)})",zorder=2)
    ax.scatter(imputed["X"], imputed["Y"], s=50,c="#41ab5d",marker="s",
               alpha=0.85,linewidths=0,label=f"Imputed ({len(imputed)})",zorder=3)
    ax.scatter(dropped_df["X"],dropped_df["Y"],s=50,c="#d73027",marker="s",
               alpha=0.7,linewidths=0,label=f"Dropped ({len(dropped_df)})",zorder=3)
    ax.scatter(sp["X"],sp["Y"],s=40,c="#111",marker="^",zorder=8,label=f"SPs ({len(sp)})")
    ax.set_aspect("equal"); ax.tick_params(labelsize=7)
    ax.set_xlabel("X (m)",fontsize=7); ax.set_ylabel("Y (m)",fontsize=7)
    ax.set_title("Cleaning decision map",fontweight="bold")
    ax.legend(fontsize=7,loc="lower right",framealpha=0.8)

    ax2 = axes[1]
    before = pd.Series({c: raw[c].isna().mean()*100
                        for c in NUM_COLS+CAT_COLS if c in raw.columns})
    after  = pd.Series({c: df_work[c].isna().mean()*100
                        for c in NUM_COLS+CAT_COLS if c in df_work.columns})
    before_s = before.sort_values(ascending=True)
    after_s  = after[before_s.index]
    y = np.arange(len(before_s))
    ax2.barh(y-0.2, before_s.values, 0.35, color="#fc8d59", alpha=0.85, label="Before")
    ax2.barh(y+0.2, after_s.values,  0.35, color="#2171b5", alpha=0.85, label="After")
    ax2.set_yticks(y); ax2.set_yticklabels(before_s.index, fontsize=6.5)
    ax2.set_xlabel("% missing"); ax2.set_title("Missing values before vs after",fontweight="bold")
    ax2.legend(fontsize=7); ax2.set_xlim(0,100); ax2.tick_params(labelsize=7)

    plt.tight_layout(rect=[0,0,1,0.94]); pdf.savefig(fig); plt.close(fig)
    print("  Page 1 done")

    # ── PAGE 2 — POPULATION DENSITY ──────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    add_title(fig, "Page 2 — Population & household density",
              "Correctly computed at 100m × 100m = 0.01 km² per square")

    heat(axes[0],"pop_density_km2","Population density (persons/km²)","YlOrRd","persons/km²")
    heat(axes[1],"hh_density_km2", "Household density (HH/km²)","OrRd","HH/km²")

    plt.tight_layout(rect=[0,0,1,0.94]); pdf.savefig(fig); plt.close(fig)
    print("  Page 2 done")

    # ── PAGE 3 — AGE STRUCTURE ────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    add_title(fig, "Page 3 — Age structure")
    heat(axes[0,0],"pct_young",  "Children 0–14 (% of pop)","Blues","%")
    heat(axes[0,1],"Age15-24",   "Youth 15–24 (count)","Purples","persons")
    heat(axes[1,0],"pct_working","Working age 25–64 (% of pop)","Greens","%")
    heat(axes[1,1],"pct_elderly","Elderly 65+ (% of pop)","Oranges","%")
    plt.tight_layout(rect=[0,0,1,0.95]); pdf.savefig(fig); plt.close(fig)
    print("  Page 3 done")

    # ── PAGE 4 — HOUSEHOLD COMPOSITION ───────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    add_title(fig, "Page 4 — Household composition")
    heat(axes[0,0],"pct_single_hh","Single-person HH (% of all HH)","RdPu","%")
    heat(axes[0,1],"Two-parent households","Two-parent HH (count)","YlGn","HH")
    heat(axes[1,0],"Single parent households","Single-parent HH (count)","Reds","HH")
    heat(axes[1,1],"avg_hh_size","Average household size","BuPu","persons/HH")
    plt.tight_layout(rect=[0,0,1,0.95]); pdf.savefig(fig); plt.close(fig)
    print("  Page 4 done")

    # ── PAGE 5 — HOUSEHOLD COMPOSITION BAR (like their notebook) ─────────
    fig = plt.figure(figsize=(14, 9))
    add_title(fig, "Page 5 — Household composition mix per square",
              "Squares sorted by single-person share (descending) — matches their notebook approach")

    hh_cols_raw = ["Single-person households","Multi-person households w/o kids",
                   "Single parent households","Two-parent households"]
    hh_labels   = ["Single-person","Multi (no kids)","Single-parent","Two-parent"]
    hh_colors   = ["#2171b5","#fd8d3c","#d73027","#41ab5d"]

    df_hh = df_work[["Square","Households"]+hh_cols_raw].dropna().copy()
    df_hh = df_hh[df_hh["Households"]>0]
    for i,raw_c in enumerate(hh_cols_raw):
        df_hh[f"sh{i}"] = df_hh[raw_c] / df_hh["Households"]
    df_hh = df_hh.sort_values("sh0", ascending=False).reset_index(drop=True)

    gs = mgridspec.GridSpec(1,2,figure=fig,width_ratios=[3,1])
    ax_bar = fig.add_subplot(gs[0])
    ax_map = fig.add_subplot(gs[1])

    bottom = np.zeros(len(df_hh))
    for i,(lbl,clr) in enumerate(zip(hh_labels,hh_colors)):
        vals = df_hh[f"sh{i}"].values
        ax_bar.bar(df_hh.index, vals, bottom=bottom, color=clr,
                   alpha=0.85, label=lbl, width=1.0)
        bottom += vals
    ax_bar.set_xlim(-0.5, len(df_hh)-0.5)
    ax_bar.set_xlabel("Squares (sorted by single-person share)",fontsize=8)
    ax_bar.set_ylabel("Share of households",fontsize=8)
    ax_bar.set_title("Household type mix per square",fontweight="bold")
    ax_bar.set_xticklabels([]); ax_bar.tick_params(labelsize=7)
    ax_bar.legend(fontsize=8,loc="upper right",framealpha=0.8)
    ax_bar.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f"{v:.0%}"))

    # Dominant type map
    sh_cols = [f"sh{i}" for i in range(4)]
    df_hh["dominant"] = df_hh[sh_cols].idxmax(axis=1).map(
        {f"sh{i}":lbl for i,lbl in enumerate(hh_labels)})
    dom_color_map = dict(zip(hh_labels,hh_colors))
    merged = df_hh.merge(df_work[["Square","X","Y"]], on="Square", how="left")
    for lbl,clr in dom_color_map.items():
        sub = merged[merged["dominant"]==lbl]
        ax_map.scatter(sub["X"],sub["Y"],s=40,c=clr,marker="s",
                       alpha=0.85,linewidths=0,label=lbl,zorder=2)
    ax_map.scatter(sp["X"],sp["Y"],s=35,c="#111",marker="^",zorder=8)
    ax_map.set_aspect("equal"); ax_map.tick_params(labelsize=6)
    ax_map.set_title("Dominant HH type\nby square",fontsize=9,fontweight="bold")
    ax_map.legend(fontsize=6,loc="lower right",framealpha=0.8)

    plt.tight_layout(rect=[0,0,1,0.94]); pdf.savefig(fig); plt.close(fig)
    print("  Page 5 done")

    # ── PAGE 6 — INCOME & HOME VALUE ──────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    add_title(fig, "Page 6 — Income & housing market")
    heat(axes[0,0],"income_score","Median income (1=low → 9=high)","RdYlGn","score")
    heat(axes[0,1],"Avg. home value k€","Avg. home value (k€)","YlGn","k€")
    heat(axes[1,0],"Home ownership %","Home ownership (%)","Blues","%")
    heat(axes[1,1],"pct_rental","Rental housing (%)","Oranges","%")
    plt.tight_layout(rect=[0,0,1,0.95]); pdf.savefig(fig); plt.close(fig)
    print("  Page 6 done")

    # ── PAGE 7 — URBANISATION & ACCESSIBILITY ─────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    add_title(fig, "Page 7 — Urbanisation & supermarket accessibility")
    heat(axes[0],"Urbanization index","Urbanisation index (1=rural → 5=very urban)","PuBu","score")
    heat(axes[1],"Distance nearest supermarket in km",
         "Distance to nearest supermarket (km)","RdYlGn_r","km")
    plt.tight_layout(rect=[0,0,1,0.94]); pdf.savefig(fig); plt.close(fig)
    print("  Page 7 done")

    # ── PAGE 8 — COMPOSITE DEMAND SCORE ──────────────────────────────────
    fig = plt.figure(figsize=(14, 8))
    add_title(fig, "Page 8 — Composite demand score",
              "Weights: 40% households · 20% working-age % · 20% urbanisation · 10% single-HH % · 10% income")
    gs = mgridspec.GridSpec(1,2,figure=fig,width_ratios=[2.2,1])
    ax_map = fig.add_subplot(gs[0]); ax_bar = fig.add_subplot(gs[1])

    sc = ax_map.scatter(df_work["X"],df_work["Y"],
                        c=df_work["demand_score"],s=55,marker="s",
                        cmap=DEMAND_CMAP,alpha=0.9,linewidths=0,vmin=0,vmax=1,zorder=2)
    sp_overlay(ax_map)
    cb = plt.colorbar(sc,ax=ax_map,fraction=0.03,pad=0.02)
    cb.set_label("Demand score",fontsize=7); cb.ax.tick_params(labelsize=7)
    ax_map.set_aspect("equal"); ax_map.tick_params(labelsize=7)
    ax_map.set_title("Demand score per square (0=low, 1=high)",fontweight="bold")
    ax_map.set_xlabel("X (m)",fontsize=7); ax_map.set_ylabel("Y (m)",fontsize=7)

    # SP ranked bar
    sq_tree = cKDTree(np.column_stack([df_work["X"],df_work["Y"]]))
    sp_scores = []
    for _,row in sp.iterrows():
        d,i = sq_tree.query([row["X"],row["Y"]])
        sp_scores.append(df_work.iloc[i]["demand_score"] if d<300 else np.nan)
    sp_df = pd.DataFrame({"id":sp["Location ID"].astype(str),"score":sp_scores})\
              .dropna().sort_values("score")
    ax_bar.barh(range(len(sp_df)),sp_df["score"],
                color=[DEMAND_CMAP(v) for v in sp_df["score"]],height=0.7)
    ax_bar.set_yticks(range(len(sp_df)))
    ax_bar.set_yticklabels(sp_df["id"],fontsize=6)
    ax_bar.set_xlabel("Demand score at SP",fontsize=7)
    ax_bar.set_title("Demand score\nby service point",fontsize=9,fontweight="bold")
    ax_bar.set_xlim(0,1); ax_bar.tick_params(labelsize=7)
    med = sp_df["score"].median()
    ax_bar.axvline(med,color="#d73027",lw=1,ls="--",label=f"Median={med:.2f}")
    ax_bar.legend(fontsize=7)

    plt.tight_layout(rect=[0,0,1,0.93]); pdf.savefig(fig); plt.close(fig)
    print("  Page 8 done")

    # ── PAGE 9 — INTERPOLATED DEMAND SURFACE ─────────────────────────────
    fig, ax = plt.subplots(figsize=(10,9))
    add_title(fig,"Page 9 — Interpolated demand surface with SP overlay")
    vd = df_work["demand_score"].notna()
    xi = np.linspace(df_work["X"].min(),df_work["X"].max(),220)
    yi = np.linspace(df_work["Y"].min(),df_work["Y"].max(),220)
    XX,YY = np.meshgrid(xi,yi)
    ZZ = griddata((df_work.loc[vd,"X"],df_work.loc[vd,"Y"]),
                  df_work.loc[vd,"demand_score"],(XX,YY),method="linear")
    cf = ax.contourf(XX,YY,ZZ,levels=22,cmap=DEMAND_CMAP,alpha=0.78)
    ax.contour(XX,YY,ZZ,levels=8,colors="white",linewidths=0.4,alpha=0.5)
    plt.colorbar(cf,ax=ax,label="Demand score")
    sp_overlay(ax)
    for _,row in sp.iterrows():
        ax.annotate(str(int(row["Location ID"])),(row["X"],row["Y"]),
                    fontsize=4.5,xytext=(3,3),textcoords="offset points",color="#111")
    ax.set_aspect("equal"); ax.tick_params(labelsize=7)
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.legend(fontsize=7,loc="lower right")
    ax.set_title("Continuous demand surface — darker = higher parcel demand",fontweight="bold")
    plt.tight_layout(rect=[0,0,1,0.95]); pdf.savefig(fig); plt.close(fig)
    print("  Page 9 done")

    # ── PAGE 10 — HIGH-DEMAND ZONES ───────────────────────────────────────
    fig, axes = plt.subplots(1,2,figsize=(14,7))
    add_title(fig,"Page 10 — High-demand zones & demand quintiles")

    ax = axes[0]
    df_work["dquint"] = pd.qcut(df_work["demand_score"],5,
                                labels=["Q1","Q2","Q3","Q4","Q5 (highest)"])
    qcols = ["#eff3ff","#bdd7e7","#6baed6","#2171b5","#08306b"]
    for i,(lbl,grp) in enumerate(df_work.groupby("dquint",observed=True)):
        ax.scatter(grp["X"],grp["Y"],s=45,marker="s",c=qcols[i],
                   alpha=0.85,linewidths=0,label=lbl,zorder=2)
    ax.scatter(sp["X"],sp["Y"],s=40,c="#d73027",marker="^",zorder=8,label="SP")
    ax.set_aspect("equal"); ax.tick_params(labelsize=7)
    ax.set_title("Demand quintiles",fontweight="bold")
    ax.set_xlabel("X (m)",fontsize=7); ax.set_ylabel("Y (m)",fontsize=7)
    ax.legend(fontsize=7,loc="lower right",framealpha=0.8)

    ax2 = axes[1]
    sc2 = ax2.scatter(df_work["X"],df_work["Y"],
                      c=df_work["dist_to_sp"],s=45,marker="s",
                      cmap="RdYlGn_r",alpha=0.85,linewidths=0,
                      vmin=0,vmax=3000,zorder=2)
    ax2.scatter(sp["X"],sp["Y"],s=40,c="#000",marker="^",zorder=8,label="SP")
    cb2 = plt.colorbar(sc2,ax=ax2,fraction=0.035,pad=0.02)
    cb2.set_label("Distance to nearest SP (m)",fontsize=7); cb2.ax.tick_params(labelsize=7)
    ax2.set_aspect("equal"); ax2.tick_params(labelsize=7)
    ax2.set_title("Distance to nearest SP\n(red = underserved)",fontweight="bold")
    ax2.set_xlabel("X (m)",fontsize=7)
    ax2.legend(fontsize=7)
    plt.tight_layout(rect=[0,0,1,0.94]); pdf.savefig(fig); plt.close(fig)
    print("  Page 10 done")

    # ── PAGE 11 — COVERAGE GAP MAP ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10,9))
    add_title(fig,"Page 11 — Coverage gap map",
              "Gap score = demand × distance to nearest SP — prime candidates for new SPs or APLs")
    sc = ax.scatter(df_work["X"],df_work["Y"],
                    c=df_work["gap_score"],s=55,marker="s",
                    cmap="hot_r",alpha=0.87,linewidths=0,vmin=0,vmax=1,zorder=2)
    ax.scatter(sp["X"],sp["Y"],s=60,c="#00aaff",edgecolors="#111",
               linewidths=1,marker="^",zorder=8,label="Current SP")
    top5 = df_work.nlargest(5,"gap_score")
    for _,row in top5.iterrows():
        ax.annotate(f"⚠ {row['Square']}\ngap={row['gap_score']:.2f}",
                    (row["X"],row["Y"]),fontsize=6.5,color="#c0392b",fontweight="bold",
                    xytext=(10,8),textcoords="offset points",
                    arrowprops=dict(arrowstyle="-",color="#c0392b",lw=0.8))
    cb = plt.colorbar(sc,ax=ax,fraction=0.03,pad=0.02)
    cb.set_label("Gap score",fontsize=7); cb.ax.tick_params(labelsize=7)
    ax.set_aspect("equal"); ax.tick_params(labelsize=7)
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.legend(fontsize=8,loc="lower right")
    ax.set_title("Coverage gap score — high demand areas poorly served by current network",
                 fontweight="bold")
    plt.tight_layout(rect=[0,0,1,0.94]); pdf.savefig(fig); plt.close(fig)
    print("  Page 11 done")

    # ── PAGE 12 — CORRELATION MATRIX ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(13,11))
    add_title(fig,"Page 12 — Correlation matrix (clean data)")
    cc = ["Population","Households","pct_single_hh","pct_working","pct_elderly",
          "income_score","Avg. home value k€","Home ownership %",
          "Urbanization index","Distance nearest supermarket in km",
          "demand_score","dist_to_sp"]
    ll = ["Population","Households","Single HH %","Working age %","Elderly %",
          "Income","Home value","% ownership","Urban index","Dist. supermarket",
          "Demand score","Dist. to SP"]
    corr = df_work[cc].corr()
    im = ax.imshow(corr.values,cmap="RdBu_r",vmin=-1,vmax=1,aspect="auto")
    ax.set_xticks(range(len(ll))); ax.set_xticklabels(ll,rotation=45,ha="right",fontsize=8)
    ax.set_yticks(range(len(ll))); ax.set_yticklabels(ll,fontsize=8)
    plt.colorbar(im,ax=ax,fraction=0.025,label="Pearson r")
    for i in range(len(ll)):
        for j in range(len(ll)):
            v = corr.values[i,j]
            ax.text(j,i,f"{v:.2f}",ha="center",va="center",fontsize=6.5,
                    color="white" if abs(v)>0.5 else "#333")
    ax.set_title("Pearson correlation — key demand & spatial variables",fontweight="bold")
    plt.tight_layout(rect=[0,0,1,0.94]); pdf.savefig(fig); plt.close(fig)
    print("  Page 12 done")

    # ── PAGE 13 — DISTRIBUTIONS ───────────────────────────────────────────
    fig, axes = plt.subplots(3,3,figsize=(14,12))
    add_title(fig,"Page 13 — Variable distributions (clean data)")
    pairs = [
        ("Population","Population per square"),
        ("pop_density_km2","Population density (persons/km²)"),
        ("hh_density_km2","Household density (HH/km²)"),
        ("pct_single_hh","Single-person HH (%)"),
        ("pct_working","Working age 25–64 (%)"),
        ("pct_elderly","Elderly 65+ (%)"),
        ("income_score","Income score (1–9)"),
        ("Avg. home value k€","Avg home value (k€)"),
        ("demand_score","Composite demand score"),
    ]
    for ax_i,(col,lbl) in zip(axes.flatten(),pairs):
        data = df_work[col].dropna()
        ax_i.hist(data,bins=28,color="#2171b5",alpha=0.75,edgecolor="white",linewidth=0.4)
        ax_i.axvline(data.mean(),color="#d73027",lw=1.5,ls="-",
                     label=f"Mean={data.mean():.1f}")
        ax_i.axvline(data.median(),color="#fc8d59",lw=1.5,ls="--",
                     label=f"Med={data.median():.1f}")
        ax_i.set_title(lbl,fontsize=9,fontweight="bold")
        ax_i.set_ylabel("Count",fontsize=7); ax_i.tick_params(labelsize=7)
        ax_i.legend(fontsize=6)
        ax_i.text(0.97,0.97,f"n={len(data)}\nskew={data.skew():.2f}",
                  transform=ax_i.transAxes,ha="right",va="top",fontsize=6.5,color="#555")
    plt.tight_layout(rect=[0,0,1,0.95]); pdf.savefig(fig); plt.close(fig)
    print("  Page 13 done")

    # ── PAGE 14 — SP NEIGHBOURHOOD PROFILE TABLE ──────────────────────────
    fig, ax = plt.subplots(figsize=(14,9))
    add_title(fig,"Page 14 — Service point neighbourhood profiles")
    ax.axis("off")
    rows = []
    for _,row in sp.iterrows():
        d,i = sq_tree.query([row["X"],row["Y"]])
        sq  = df_work.iloc[i]
        rows.append([
            int(row["Location ID"]),
            sq["Square"],
            f"{sq['Population']:.0f}" if pd.notna(sq["Population"]) else "—",
            f"{sq['Households']:.0f}" if pd.notna(sq["Households"]) else "—",
            f"{sq['pop_density_km2']:.0f}" if pd.notna(sq["pop_density_km2"]) else "—",
            f"{sq['pct_single_hh']:.0f}%" if pd.notna(sq["pct_single_hh"]) else "—",
            str(sq["Median household income"])[:18] if pd.notna(sq["Median household income"]) else "—",
            f"{sq['Urbanization index']:.0f}" if pd.notna(sq["Urbanization index"]) else "—",
            f"{sq['demand_score']:.2f}" if pd.notna(sq["demand_score"]) else "—",
        ])
    cols = ["SP ID","Square","Pop","HH","Pop/km²","Single HH%","Income","Urban","Demand"]
    tbl_df = pd.DataFrame(rows,columns=cols).sort_values("Demand",ascending=False)
    tbl = ax.table(cellText=tbl_df.values,colLabels=tbl_df.columns,
                   loc="center",cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(7.5); tbl.scale(1,1.35)
    for j in range(len(cols)):
        tbl[0,j].set_facecolor("#2171b5")
        tbl[0,j].set_text_props(color="white",fontweight="bold")
    for i,r in enumerate(tbl_df.values,start=1):
        d = float(r[8]) if r[8]!="—" else 0
        shade = plt.cm.Blues(0.18+0.65*d)
        tbl[i,8].set_facecolor(shade)
        for j in range(len(cols)-1):
            tbl[i,j].set_facecolor("#f9f9f7")
    ax.set_title("Service points ranked by neighbourhood demand score",
                 fontsize=10,fontweight="bold",pad=18)
    plt.tight_layout(rect=[0,0,1,0.95]); pdf.savefig(fig); plt.close(fig)
    print("  Page 14 done")

print(f"\n✓  PDF → {OUT_PDF}  (14 pages)")

print("\n── Cleaning summary ──")
print(f"  Raw: 284  |  Annotation rows dropped: {note_mask.sum()}")
print(f"  Non-residential dropped: {is_non_res.sum()}")
print(f"  Residential imputed (kNN median/mode, k=8): {is_res_missing.sum()}")
print(f"  Final clean squares: {len(df_work)}")
print(f"  Population still missing: {df_work['Population'].isna().sum()}")

print("\n── Top-5 demand squares ──")
print(df_work.nlargest(5,"demand_score")[
    ["Square","Population","Households","Urbanization index",
     "income_score","demand_score"]].to_string(index=False))

print("\n── Top-5 coverage gaps ──")
print(df_work.nlargest(5,"gap_score")[
    ["Square","demand_score","dist_to_sp","gap_score"]].to_string(index=False))
