import math, random, time, warnings, copy, os
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pulp
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra as sp_dijkstra
from scipy.spatial import cKDTree
import statsmodels.api as sm

DATA_PATH     = "data/data_Maastricht_2025.xlsx"
K_MIN         = 5
K_MAX         = 40
APL_MAX       = 20
SA_RESTARTS        = 3        
SA_TIME_LIMIT      = 60      
SA_ALPHA           = 0.99997  
SA_REHEAT_INTERVAL = 45.0     
RANDOM_SEED        = 42

MOVE_WEIGHTS = {
    "swap_sp_apl": 0.25,
    "add_apl":     0.20,
    "remove_apl":  0.15,
    "perturb_apl": 0.15,
    "add_sp":      0.125,
    "remove_sp":   0.125,
}
MAX_ACCESS_KM = 3.0
# map scaling to maastricht's RD coordinates for interpretability
RD_SCALE_X, RD_OFFSET_X = 0.06949351, 172244.5245
RD_SCALE_Y, RD_OFFSET_Y = 0.10824467, 312384.3812

FIXED_COST_SP  = 50_000
FIXED_COST_APL = 20_000
DELIVERY_RATE  = 1.50
STORAGE_RATE   = 0.10
DWELL_PICKUP   = 2
DWELL_FAILED   = 3
DWELL_APL      = 1
K_APL_CAP         = 200
K_SP_CAP          = 1400  # max parcels in storage at an SP on a peak day
BOUNCE_PER_APL    = 0.02
BOUNCE_CITY       = 0.01
SP_AVG_REDIRECT_KM = 0.8   # typical distance to next-nearest SP when overflow redirected
APL_UPLIFT_AVG = 0.13
KNN_K          = 8
BASE_FAILED    = 0.05
RATE_MIN       = 0.02
RATE_MAX       = 0.10
PEAK_MULTIPLIER = 1.555

INCOME_MAP = {
    "00-20 low": 10, "00-40 laag tot onder midden": 20,
    "20-40 below middle": 30, "20-60 below middle to middle": 40,
    "40-60 midden": 50, "40-80 middle to above middle": 60,
    "60-80 above middle": 70, "60-100 above middle to high": 80,
    "80-100 high": 90,
}

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# MODEL 1 — DEMAND MODEL

def build_demand_model():
    def _clean(df):
        return df[[c for c in df.columns
                   if not str(c).startswith("Unnamed")]].copy()

    sp_df    = _clean(pd.read_excel(DATA_PATH, sheet_name="Service Point Locations"))
    act_df   = _clean(pd.read_excel(DATA_PATH, sheet_name="Daily Activity"))
    cbs_raw  = _clean(pd.read_excel(DATA_PATH, sheet_name="CBS Squares"))
    nodes_df = _clean(pd.read_excel(DATA_PATH, sheet_name="Nodes"))
    edges_df = _clean(pd.read_excel(DATA_PATH, sheet_name="Edges"))
    cbs_raw.columns = [c.strip() for c in cbs_raw.columns]

    daily_tot  = act_df.groupby("Date")[["Deliveries","Pickups"]].sum().sum(axis=1)
    OPER_DAYS  = int((daily_tot > 0).sum())
    TOTAL_ANN  = int(act_df["Deliveries"].sum() + act_df["Pickups"].sum())
    print(f"\n[2] Operating days: {OPER_DAYS}  |  Annual parcels: {TOTAL_ANN:,}")

    sp_annual = (act_df.groupby("Location ID")[["Pickups","Deliveries"]]
                 .sum().reset_index())
    sp_annual["total"]       = sp_annual["Pickups"] + sp_annual["Deliveries"]
    sp_annual["pickup_rate"] = sp_annual["Pickups"] / sp_annual["total"]
    CLIP_MIN = float(sp_annual["pickup_rate"].min())
    CLIP_MAX = float(sp_annual["pickup_rate"].max())
    print(f"  Pickup rate clip: [{CLIP_MIN:.3f}, {CLIP_MAX:.3f}]")

    cbs   = cbs_raw.copy()
    hcols = [c for c in ["Houses","Home ownership %","Rental %","Social housing %"]
             if c in cbs.columns]
    non_res = (
        cbs["Population"].fillna(0).eq(0) &
        cbs["Households"].fillna(0).eq(0) &
        (cbs[hcols].fillna(0).sum(axis=1).eq(0) if hcols else True)
    )
    res = cbs[~non_res].copy().reset_index(drop=True)
    print(f"  Non-residential removed: {non_res.sum()}  |  Residential: {len(res)}")

    xmed = float(res["X"].median()); ymed = float(res["Y"].median())
    xtol = (res["X"].max()-res["X"].min())*0.25
    ytol = (res["Y"].max()-res["Y"].min())*0.25
    in_centre      = (res["X"].between(xmed-xtol,xmed+xtol) &
                      res["Y"].between(ymed-ytol,ymed+ytol))
    key_cols       = [c for c in ["Population","Households","Avg. home value k€"]
                      if c in res.columns]
    centre_missing = in_centre & res[key_cols].isna().any(axis=1)

    num_cols = [c for c in [
        "Population","Male","Female","Age0-14","Age15-24","Age25-44",
        "Age45-64","Age65+","Households","Single-person households",
        "Multi-person households w/o kids","Single parent households",
        "Two-parent households","Houses","Home ownership %","Rental %",
        "Social housing %","Vacant houses","Avg. home value k€","OAD",
        "Urbanization index","Percentage low income households",
        "Percentage high income households",
        "Distance nearest supermarket in km",
    ] if c in res.columns]

    kd = cKDTree(res[["X","Y"]].values)
    for i in res.index[centre_missing]:
        xy = res.loc[i,["X","Y"]].values.astype(float)
        if np.isnan(xy).any(): continue
        _, nn = kd.query([xy], k=KNN_K+1)
        nbrs  = res.iloc[nn[0][1:]]
        for c in num_cols:
            if pd.isna(res.at[i,c]):
                v = nbrs[c].median()
                if not pd.isna(v): res.at[i,c] = v
        if "Median household income" in res.columns \
                and pd.isna(res.at[i,"Median household income"]):
            m = nbrs["Median household income"].mode(dropna=True)
            if not m.empty: res.at[i,"Median household income"] = m.iloc[0]

    for c in num_cols:
        res[c] = res[c].fillna(res[c].median())
    if "Median household income" in res.columns:
        mv = res["Median household income"].mode(dropna=True)
        if not mv.empty:
            res["Median household income"] = (res["Median household income"]
                                               .fillna(mv.iloc[0]))
    res["income_score"] = res["Median household income"].map(INCOME_MAP)
    res["income_score"] = res["income_score"].fillna(float(res["income_score"].median()))

    v1 = (edges_df["V1"]-1).values.astype(np.int32)
    v2 = (edges_df["V2"]-1).values.astype(np.int32)
    wt = edges_df["DIST"].values.astype(np.float64)
    n  = int(max(edges_df["V1"].max(), edges_df["V2"].max())) + 1  # fix off-by-one

    oneway = edges_df["ONE_WAY"].values.astype(bool)

    # Two-way streets: add both directions
    v1_tw = v1[~oneway]; v2_tw = v2[~oneway]; wt_tw = wt[~oneway]
    # One-way streets: add only the direction in the data
    v1_ow = v1[oneway];  v2_ow = v2[oneway];  wt_ow = wt[oneway]

    rows = np.concatenate([v1_tw, v2_tw, v1_ow])
    cols = np.concatenate([v2_tw, v1_tw, v2_ow])
    vals = np.concatenate([wt_tw, wt_tw, wt_ow])

    graph = csr_matrix((vals, (rows, cols)), shape=(n, n))
    node_xy   = nodes_df[["X","Y"]].values
    node_ids0 = (nodes_df["NODE ID"]-1).values.astype(int)
    kdt       = cKDTree(node_xy)
    _, idx = kdt.query(res[["X","Y"]].values);  res["road_node"] = node_ids0[idx]
    sp_df = sp_df.copy()
    _, idx = kdt.query(sp_df[["X","Y"]].values); sp_df["road_node"] = node_ids0[idx]

    cbs_nodes = res["road_node"].values; sp_nodes = sp_df["road_node"].values
    all_src   = np.unique(np.concatenate([cbs_nodes, sp_nodes]))
    D_full    = sp_dijkstra(graph, directed=True, indices=all_src)
    D_full    = np.where(np.isinf(D_full), 1e8, D_full)
    s2r       = {int(nd):i for i,nd in enumerate(all_src)}
    cbs_rows  = np.array([s2r[int(nd)] for nd in cbs_nodes])
    dist_mat  = D_full[np.ix_(cbs_rows, sp_nodes)] / 1000.0
    D_cand_km = D_full[np.ix_(cbs_rows, cbs_nodes)] / 1000.0

    nearest_idx = dist_mat.argmin(axis=1)
    hh = res["Households"].fillna(0).astype(float)
    res["demand_hh"] = (hh / hh.sum()) * TOTAL_ANN
    sp_loc_ids = sp_df["Location ID"].values
    res["nearest_sp_loc_id"] = sp_loc_ids[nearest_idx]
    obs_vol = sp_annual.set_index("Location ID")["total"]
    demand_cal = res["demand_hh"].values.copy()
    for j in range(len(sp_df)):
        mask = nearest_idx == j
        if not mask.any(): continue
        loc_id = sp_loc_ids[j]
        if loc_id not in obs_vol.index: continue
        if demand_cal[mask].sum() <= 0: continue
        scale = float(obs_vol[loc_id]) / float(demand_cal[mask].sum())
        demand_cal[mask] *= scale
    res["demand_annual"] = demand_cal
    print(f"  Calibrated total: {demand_cal.sum():,.0f}  "
          f"scale range [{(demand_cal/res['demand_hh'].values).min():.3f}x, "
          f"{(demand_cal/res['demand_hh'].values).max():.3f}x]")

    res["dist_to_sp"] = dist_mat[np.arange(len(res)), nearest_idx]
    sp_feats = res.groupby("nearest_sp_loc_id").agg(
        dist_to_sp=("dist_to_sp", "mean"),
        income_score=("income_score", "mean")
    ).reset_index()
    sp_feats = sp_feats.merge(sp_annual[["Location ID","pickup_rate"]],
                            left_on="nearest_sp_loc_id", right_on="Location ID",
                            how="inner")
    sp_feats = sp_feats.dropna(subset=["pickup_rate","dist_to_sp","income_score"])
    y_sq = sp_feats["pickup_rate"].values
    X_sq = sm.add_constant(sp_feats[["dist_to_sp","income_score"]].values)
    ols  = sm.OLS(y_sq, X_sq).fit()
    INTERCEPT, BETA_DIST, BETA_INC = ols.params
    print(f"  SP-level OLS  (n={len(sp_feats)}, R²={ols.rsquared:.3f})")

    res["single_pct"]  = (res["Single-person households"].astype(float) /
                          res["Households"].replace(0,np.nan).astype(float)).fillna(0.0)
    res["working_pct"] = (res["Age25-44"].astype(float) /
                          res["Population"].replace(0,np.nan).astype(float)).fillna(0.0)
    w           = demand_cal
    avg_single  = float(np.average(res["single_pct"],  weights=w))
    avg_working = float(np.average(res["working_pct"], weights=w))
    risk        = res["single_pct"] + res["working_pct"]
    hi, lo      = int(risk.argmax()), int(risk.argmin())
    A = np.array([[res.at[hi,"single_pct"]-avg_single, res.at[hi,"working_pct"]-avg_working],
                  [res.at[lo,"single_pct"]-avg_single, res.at[lo,"working_pct"]-avg_working]])
    b = np.array([RATE_MAX-BASE_FAILED, RATE_MIN-BASE_FAILED])
    try:    alpha_s, beta_w = np.linalg.solve(A, b)
    except: alpha_s, beta_w = np.linalg.lstsq(A, b, rcond=None)[0]
    res["failed_rate"] = np.clip(
        BASE_FAILED + alpha_s*(res["single_pct"]-avg_single) +
        beta_w*(res["working_pct"]-avg_working), RATE_MIN, RATE_MAX)
    print(f"  Failed rate: [{res['failed_rate'].min():.2%}, {res['failed_rate'].max():.2%}]  "
          f"city mean: {np.average(res['failed_rate'],weights=w):.2%}")

    return (res, sp_df, INTERCEPT, BETA_DIST, BETA_INC,
            CLIP_MIN, CLIP_MAX, OPER_DAYS, avg_single, avg_working,
            D_cand_km)

def compute_cost(sp_idx, res, intercept, beta_dist, beta_inc, clip_min, clip_max,
                 avg_single, avg_working, D_cand_km, oper_days, apl_idx=None,
                 fixed_cost_sp=None, fixed_cost_apl=None, delivery_rate=None,
                 apl_uplift_avg=None, peak_multiplier=None, failed_rate_override=None):
    """
    All cost parameters default to module-level globals.
    Pass overrides for sensitivity sweeps.
    """
    _fsp   = fixed_cost_sp    if fixed_cost_sp    is not None else FIXED_COST_SP
    _fapl  = fixed_cost_apl   if fixed_cost_apl   is not None else FIXED_COST_APL
    _dr    = delivery_rate    if delivery_rate     is not None else DELIVERY_RATE
    _upl   = apl_uplift_avg   if apl_uplift_avg   is not None else APL_UPLIFT_AVG
    _peak  = peak_multiplier  if peak_multiplier   is not None else PEAK_MULTIPLIER

    demand       = res["demand_annual"].values
    income       = res["income_score"].values
    failed_rate  = (failed_rate_override if failed_rate_override is not None
                    else res["failed_rate"].values)
    single_frac  = res["single_pct"].values
    working_frac = res["working_pct"].values

    n_sq  = len(demand)
    n_sp  = len(sp_idx)
    n_apl = len(apl_idx) if apl_idx is not None else 0

    all_idx  = np.concatenate([sp_idx, apl_idx]) if n_apl > 0 else np.array(sp_idx)
    dist_all = D_cand_km[:, all_idx]
    nearest  = dist_all.argmin(axis=1)
    d_km     = dist_all[np.arange(n_sq), nearest]
    is_apl   = nearest >= n_sp

    base_rate = np.clip(intercept + beta_dist*d_km + beta_inc*income, clip_min, clip_max)
    if n_apl > 0:
        uplift = _upl * (0.60*single_frac/max(avg_single,1e-9) +
                         0.40*working_frac/max(avg_working,1e-9))
        uplift      = np.clip(uplift, 0.03, 0.25)
        apl_rate    = np.clip(intercept + beta_dist*d_km + beta_inc*income + uplift, 0.0, 1.0)
        pickup_prob = np.where(is_apl, apl_rate, base_rate)
    else:
        pickup_prob = base_rate

    annual_pickups = demand * pickup_prob
    annual_del     = demand * (1 - pickup_prob)
    annual_failed  = annual_del * failed_rate
    annual_success = annual_del - annual_failed
    dwell          = np.where(is_apl, DWELL_APL, DWELL_PICKUP)

    fixed    = n_sp*_fsp + n_apl*_fapl
    storage  = (annual_pickups*dwell + annual_failed*DWELL_FAILED) * STORAGE_RATE
    delivery = annual_success*d_km*_dr + annual_failed*d_km*_dr*2

    # Percentile-based bounce check (v3-3), consistent denominator
    apl_stored_frac  = pickup_prob
    apl_mask_any     = is_apl
    total_stored_ann = float((demand[apl_mask_any]*apl_stored_frac[apl_mask_any]).sum())
    max_bounce_apl   = 0.0; total_overflow = 0.0; n_over = 0
    for j in range(n_sp, n_sp+n_apl):
        mask = nearest == j
        if not mask.any(): continue
        mu_mean = float((demand[mask]*apl_stored_frac[mask]).sum()) / max(oper_days, 1)
        if mu_mean <= 0: continue
        mu_peak = mu_mean * _peak
        ov = max(0.0, mu_peak - K_APL_CAP); br = ov / mu_peak if mu_peak > 0 else 0
        if br > BOUNCE_PER_APL: n_over += 1
        max_bounce_apl = max(max_bounce_apl, br)
        total_overflow += ov * oper_days
    city_bounce  = total_overflow / max(total_stored_ann * _peak, 1)
    apl_feasible = (city_bounce <= BOUNCE_CITY) and (n_over == 0)

    # SP overflow cost: parcels exceeding K_SP_CAP on a peak day are redirected
    # to the next nearest SP, charged at delivery rate for the extra trip.
    sp_overflow_cost = 0.0
    sp_max_load      = 0.0
    for j in range(n_sp):
        mask = nearest == j
        if not mask.any(): continue
        avg_store = (annual_pickups[mask].sum()*DWELL_PICKUP +
                     annual_failed[mask].sum()*DWELL_FAILED) / max(oper_days, 1)
        peak_store = avg_store * _peak
        overflow_parcels = max(0.0, peak_store - K_SP_CAP)
        sp_overflow_cost += overflow_parcels * SP_AVG_REDIRECT_KM * _dr * oper_days
        sp_max_load = max(sp_max_load, peak_store)
    sp_feasible    = True   # informational only — no longer a hard constraint
    sp_max_bounce  = 0.0
    sp_bounce_rate = None

    # Accessibility: flag squares beyond MAX_ACCESS_KM
    access_violations = int((d_km > MAX_ACCESS_KM).sum())
    demand_underserved = float(demand[d_km > MAX_ACCESS_KM].sum())

    return {
        "total": fixed+storage.sum()+delivery.sum()+sp_overflow_cost,
        "fixed": float(fixed), "storage": float(storage.sum()),
        "delivery": float(delivery.sum()),
        "sp_overflow_cost": sp_overflow_cost,
        "sp_max_load": sp_max_load,
        "pickup_rate": float((demand*pickup_prob).sum()/demand.sum()),
        "avg_dist_km": float((demand*d_km).sum()/demand.sum()),
        "max_dist_km": float(d_km.max()),
        "access_violations": access_violations,
        "demand_underserved": demand_underserved,
        "apl_feasible": apl_feasible,
        "apl_max_bounce": max_bounce_apl, "apl_bounce_rate": city_bounce,
        "sp_feasible": sp_feasible, "sp_max_bounce": sp_max_bounce,
        "sp_bounce_rate": sp_bounce_rate,
        "_demand": demand, "_pickup_prob": pickup_prob,
        "_annual_pickups": annual_pickups, "_annual_failed": annual_failed,
        "_annual_success": annual_success, "_d_km": d_km,
        "_nearest": nearest, "_is_apl": is_apl,
    }

def per_facility_breakdown(sp_xy, apl_xy, result, oper_days, sp_idx=None, D_cand_km=None):
    rows = []
    n_sp = len(sp_xy); n_apl = len(apl_xy) if apl_xy is not None else 0
    if n_apl > 0 and sp_idx is not None and D_cand_km is not None:
        d_to_sp = D_cand_km[:, sp_idx].min(axis=1)
    else:
        d_to_sp = result["_d_km"]
    for j in range(n_sp + n_apl):
        mask = result["_nearest"] == j
        if not mask.any(): continue
        is_apl_loc = j >= n_sp
        xy    = apl_xy[j-n_sp] if is_apl_loc else sp_xy[j]
        dem   = result["_demand"][mask]
        pkp   = result["_annual_pickups"][mask].sum()
        fail  = result["_annual_failed"][mask].sum()
        d     = result["_d_km"][mask]
        dw    = DWELL_APL if is_apl_loc else DWELL_PICKUP
        sc    = (pkp*dw + fail*DWELL_FAILED) * STORAGE_RATE
        succ    = result["_annual_success"][mask]
        fail_sq = result["_annual_failed"][mask]
        dc      = float((succ * d * DELIVERY_RATE).sum() + (fail_sq * d * DELIVERY_RATE * 2).sum())
        fc    = FIXED_COST_APL if is_apl_loc else FIXED_COST_SP
        rows.append({
            "Facility": f"APL {j-n_sp+1}" if is_apl_loc else f"SP {j+1}",
            "Type": "APL" if is_apl_loc else "SP",
            "X": round(xy[0]), "Y": round(xy[1]),
            "CBS_squares":    int(mask.sum()),
            "annual_demand":  round(dem.sum()),
            "daily_demand":   round(dem.sum()/oper_days, 1),
            "annual_pickups": round(pkp),
            "annual_failed":  round(fail),
            "pickup_%":       round(pkp/dem.sum()*100,1) if dem.sum()>0 else 0,
            "avg_dist_km":    round(float(np.average(d,weights=dem)),3),
            "max_dist_km":    round(float(d.max()),3),
            "storage_cost":   round(sc),
            "delivery_cost":  round(dc),
            "fixed_cost":     fc,
            "total_cost":     round(sc+dc+fc),
        })
    return (pd.DataFrame(rows)
            .sort_values(["Type","annual_demand"], ascending=[True,False])
            .reset_index(drop=True))

# MODEL 3a — p-MEDIAN MILP

def solve_pmedian(k, D_cand_km, demand):
    """
    p-median MILP.  Prefer assignments within MAX_ACCESS_KM: x[i][j] is
    only created when D_cand_km[i,j] <= MAX_ACCESS_KM (plus each square's nearest
    candidate as a fallback).  For small k this hard cap can be INFEASIBLE
    (k facilities cannot keep every square within MAX_ACCESS_KM), which used to
    return None and crash the cost step / silently drop small k from the resize
    sweep.  We now detect that and automatically relax to full assignment so a
    valid p-median solution is always returned; accessibility is still reported
    (access_violations) and penalised inside the simulated annealing.
    """
    n_sq, n = D_cand_km.shape
    w         = demand / demand.sum()
    nearest_j = D_cand_km.argmin(axis=1)

    def _build(enforce_access):
        prob = pulp.LpProblem(f"p{k}", pulp.LpMinimize)
        y = [pulp.LpVariable(f"y{j}", cat="Binary") for j in range(n)]
        x = {}
        for i in range(n_sq):
            if enforce_access:
                allowed = [j for j in range(n) if D_cand_km[i, j] <= MAX_ACCESS_KM]
                if not allowed:
                    allowed = [int(nearest_j[i])]   # fallback: always allow nearest
            else:
                allowed = range(n)                  # full assignment (always feasible)
            for j in allowed:
                x[(i, j)] = pulp.LpVariable(f"x{i}_{j}", cat="Binary")
        prob += pulp.lpSum(w[i]*D_cand_km[i][j]*x[(i, j)] for (i, j) in x)
        prob += pulp.lpSum(y) == k
        for i in range(n_sq):
            row = [(i, j) for j in range(n) if (i, j) in x]
            prob += pulp.lpSum(x[p] for p in row) == 1
            for (ii, j) in row:
                prob += x[(ii, j)] <= y[j]
        return prob, y

    t0 = time.time()
    prob, y = _build(enforce_access=True)
    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=120))
    if prob.status != 1:                            # infeasible/unsolved under cap
        prob, y = _build(enforce_access=False)      # relax and resolve
        prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=120))
    t = time.time() - t0
    if prob.status != 1:
        return None, None, t
    sel = [j for j in range(n) if pulp.value(y[j]) > 0.5]
    return np.array(sel), pulp.value(prob.objective), t

# MODEL 3b — SIMULATED ANNEALING

def simulated_annealing(init_sp_idx, cand_xy, res,
                        intercept, beta_dist, beta_inc, clip_min, clip_max,
                        avg_single, avg_working, baseline_cost, D_cand_km, oper_days,
                        cost_kwargs=None, sp_allowed_idx=None):
    """
    Runs SA_RESTARTS independent SA runs and returns the best feasible
    solution found across all of them.  Restart 0 uses init_sp_idx (warm start
    from S2); subsequent restarts use random subsets of the allowed sites, giving diversity without losing the
    quality of the warm start.

    Accessibility penalty: squares beyond MAX_ACCESS_KM contribute a
    penalty proportional to the excess distance, scaled by demand, so the SA
    naturally avoids stranding customers.
    """
    if cost_kwargs is None: cost_kwargs = {}
    n_cand  = len(cand_xy)
    demand  = res["demand_annual"].values

    sp_allowed_xy = cand_xy[sp_allowed_idx] if sp_allowed_idx is not None else cand_xy

    _cand_tree = cKDTree(cand_xy)

    def _cost(sp_idx, apl_idx):
        apl_arg = apl_idx if len(apl_idx) > 0 else None
        r = compute_cost(sp_idx, res, intercept, beta_dist, beta_inc,
                         clip_min, clip_max, avg_single, avg_working,
                         D_cand_km, oper_days, apl_arg, **cost_kwargs)
        pen = 0.0
        if len(apl_idx) > 0 and not r["apl_feasible"]:
            if r["apl_max_bounce"] > BOUNCE_PER_APL:
                pen = max(pen, baseline_cost*(r["apl_max_bounce"]-BOUNCE_PER_APL)/BOUNCE_PER_APL)
            if r["apl_bounce_rate"] > BOUNCE_CITY:
                pen = max(pen, baseline_cost*(r["apl_bounce_rate"]-BOUNCE_CITY)/BOUNCE_CITY)
        d_km = r["_d_km"]
        excess = np.maximum(0.0, d_km - MAX_ACCESS_KM)
        pen += float((demand * excess * DELIVERY_RATE).sum())
        return r["total"]+pen, r["total"], r

    def _run_one(start_sp_idx, seed):
            """Single SA run. Uses coordinate-based moves (perturb/snap) on top of
            v5's index-based cost function. Positions are tracked as XY arrays;
            indices are resolved on-the-fly for compute_cost calls."""
            random.seed(seed); np.random.seed(seed)

            bb_x = cand_xy[:, 0].max() - cand_xy[:, 0].min()
            bb_y = cand_xy[:, 1].max() - cand_xy[:, 1].min()
            ps   = 0.10   # perturb scale (10% of bounding box)
            ALL_MOVES = list(MOVE_WEIGHTS.keys())

            def _pick_move(available):
                w = [MOVE_WEIGHTS[m] for m in available]
                return random.choices(available, weights=w, k=1)[0]

            def _nearest_idx(xy_arr):
                """Snap XY positions to nearest CBS centroid indices."""
                if len(xy_arr) == 0:
                    return np.array([], dtype=int)
                _, idx = _cand_tree.query(xy_arr)
                return idx

            def _cost_xy(sp_xy, apl_xy):
                sp_idx_  = _nearest_idx(sp_xy)
                apl_idx_ = _nearest_idx(apl_xy) if len(apl_xy) > 0 else None
                r = compute_cost(sp_idx_, res, intercept, beta_dist, beta_inc,
                                clip_min, clip_max, avg_single, avg_working,
                                D_cand_km, oper_days, apl_idx_, **cost_kwargs)
                pen = 0.0
                if len(apl_xy) > 0 and not r["apl_feasible"]:
                    if r["apl_max_bounce"] > BOUNCE_PER_APL:
                        pen = max(pen, baseline_cost*(r["apl_max_bounce"]-BOUNCE_PER_APL)/BOUNCE_PER_APL)
                    if r["apl_bounce_rate"] > BOUNCE_CITY:
                        pen = max(pen, baseline_cost*(r["apl_bounce_rate"]-BOUNCE_CITY)/BOUNCE_CITY)
                d_km   = r["_d_km"]
                excess = np.maximum(0.0, d_km - MAX_ACCESS_KM)
                pen   += float((demand * excess * DELIVERY_RATE).sum())
                return r["total"] + pen, r["total"], r

            # Initialise from index warm-start → convert to XY
            cur_sp  = cand_xy[np.array(start_sp_idx)].copy().astype(float)
            cur_apl = np.empty((0, 2), dtype=float)
            cur_pen, cur_true, _ = _cost_xy(cur_sp, cur_apl)

            b_sp = cur_sp.copy(); b_apl = cur_apl.copy(); b_cost = cur_true

            T_start = 0.05 * baseline_cost
            T_end   = 1e-5 * baseline_cost
            T = T_start

            op_attempts = {m: 0 for m in ALL_MOVES}
            op_accepted = {m: 0 for m in ALL_MOVES}
            it = 0; n_accept = 0
            history = [(0.0, b_cost)]
            t0 = time.time(); last_print_t = t0; last_improve_t = t0

            print(f"\n  {'Time':>6}  {'T':>10}  {'Best cost':>13}  "
                f"{'Acc%':>6}  {'SPs':>4}  {'APLs':>5}  Move acc% per operator")
            print(f"  {'─'*90}")

            while time.time() - t0 < SA_TIME_LIMIT:
                it   += 1
                n_sp  = len(cur_sp)
                n_apl = len(cur_apl)

                available = []
                can_sp_to_apl = n_sp > K_MIN and n_apl < APL_MAX
                can_apl_to_sp = False
                if can_sp_to_apl:
                    available.append("swap_sp_apl")
                if n_apl < APL_MAX: available.append("add_apl")
                if n_apl > 0:       available.extend(["remove_apl", "perturb_apl"])
                if n_sp  < len(sp_allowed_xy): available.append("add_sp")  # reopen existing site
                if n_sp  > K_MIN:   available.append("remove_sp")

                move    = _pick_move(available)
                new_sp  = cur_sp.copy()
                new_apl = cur_apl.copy()

                if move == "perturb_sp":
                    i = random.randrange(n_sp)
                    new_sp[i, 0] += random.uniform(-ps, ps) * bb_x
                    new_sp[i, 1] += random.uniform(-ps, ps) * bb_y

                elif move == "snap_sp":
                    i = random.randrange(n_sp)
                    new_sp[i] = cand_xy[random.randrange(n_cand)]

                elif move == "swap_sp_apl":
                    if can_sp_to_apl and can_apl_to_sp:
                        direction = random.choice(["sp_to_apl", "apl_to_sp"])
                    elif can_sp_to_apl:
                        direction = "sp_to_apl"
                    else:
                        direction = "apl_to_sp"
                    if direction == "sp_to_apl":
                        i       = random.randrange(n_sp)
                        loc     = new_sp[i].copy()
                        new_sp  = np.delete(new_sp, i, axis=0)
                        new_apl = np.vstack([cur_apl, loc]) if n_apl > 0 else loc.reshape(1, 2)
                    else:
                        i       = random.randrange(n_apl)
                        loc     = new_apl[i].copy()
                        new_apl = np.delete(new_apl, i, axis=0)
                        new_sp  = np.vstack([cur_sp, loc])

                elif move == "add_apl":
                    loc     = cand_xy[random.randrange(n_cand)]
                    new_apl = np.vstack([cur_apl, loc]) if n_apl > 0 else loc.reshape(1, 2)

                elif move == "remove_apl":
                    new_apl = np.delete(cur_apl, random.randrange(n_apl), axis=0)

                elif move == "perturb_apl":
                    i = random.randrange(n_apl)
                    new_apl = cur_apl.copy()
                    new_apl[i, 0] += random.uniform(-ps, ps) * bb_x
                    new_apl[i, 1] += random.uniform(-ps, ps) * bb_y

                elif move == "add_sp":
                    # reopen an EXISTING site that is currently closed
                    open_keys = {tuple(p) for p in cur_sp}
                    closed    = [p for p in sp_allowed_xy if tuple(p) not in open_keys]
                    if closed:
                        new_sp = np.vstack([cur_sp, closed[random.randrange(len(closed))]])

                elif move == "remove_sp":
                    if random.random() < 0.70:
                        sp_tree = cKDTree(cur_sp)
                        _, asgn = sp_tree.query(cand_xy)
                        load    = np.bincount(asgn, weights=demand, minlength=n_sp)
                        idx     = int(np.argmin(load))
                    else:
                        idx = random.randrange(n_sp)
                    new_sp = np.delete(cur_sp, idx, axis=0)

                op_attempts[move] += 1
                new_pen, new_true, new_res = _cost_xy(new_sp, new_apl)
                delta = new_pen - cur_pen

                if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-12)):
                    cur_sp, cur_apl, cur_pen = new_sp, new_apl, new_pen
                    op_accepted[move] += 1; n_accept += 1
                    feasible = (new_res["apl_feasible"] if len(new_apl) > 0 else True) \
                               and new_res.get("sp_feasible", True)
                    if feasible and new_true < b_cost:
                        b_sp, b_apl, b_cost = cur_sp.copy(), cur_apl.copy(), new_true
                        last_improve_t = time.time()

                T = max(T * SA_ALPHA, T_end)

                now = time.time()
                if now - last_improve_t >= SA_REHEAT_INTERVAL:
                    reheat_rate    = random.random() ** 2
                    T              = T_start * reheat_rate
                    last_improve_t = now
                    print(f"  {now-t0:5.0f}s  REHEAT → T={T:,.0f}  (rate={reheat_rate:.2f})")

                if now - last_print_t >= 30.0:
                    elapsed = now - t0
                    op_stat = "  ".join(
                        f"{m[:7]}:{100*op_accepted[m]/op_attempts[m]:.0f}%"
                        if op_attempts[m] > 0 else f"{m[:7]}:--"
                        for m in ALL_MOVES
                    )
                    print(f"  {elapsed:5.0f}s  {T:>10,.0f}  €{b_cost:>11,.0f}  "
                        f"{n_accept/it*100:>5.1f}%  {len(b_sp):>4}  {len(b_apl):>5}  " + op_stat)
                    history.append((round(elapsed, 1), b_cost))
                    op_attempts = {m: 0 for m in ALL_MOVES}
                    op_accepted = {m: 0 for m in ALL_MOVES}
                    last_print_t = now

            elapsed = time.time() - t0
            history.append((round(elapsed, 1), b_cost))
            print(f"\n  SA complete: {it:,} iterations in {elapsed:.0f}s  ({it/elapsed:.0f} it/s)")
            # Convert final XY positions back to nearest-centroid indices for v5 downstream
            return list(_nearest_idx(b_sp)), list(_nearest_idx(b_apl)), b_cost, history

    # Multi-restart loop
    overall_best_sp = None; overall_best_apl = None
    overall_best_cost = np.inf; overall_history = []

    for restart in range(SA_RESTARTS):
        if restart == 0:
            start = list(init_sp_idx)  
        else:
            # Random start: a random subset of EXISTING sites (no new SPs)
            rng  = np.random.default_rng(RANDOM_SEED + restart)
            pool = sp_allowed_idx if sp_allowed_idx is not None else np.arange(n_cand)
            start = list(rng.choice(pool, size=len(init_sp_idx), replace=False))
        seed = RANDOM_SEED + restart
        print(f"\n  SA restart {restart+1}/{SA_RESTARTS} "
              f"({'warm start (S2)' if restart==0 else 'random start'})")
        t0 = time.time()
        b_sp, b_apl, b_cost, hist = _run_one(start, seed)
        print(f"  Restart {restart+1} finished in {time.time()-t0:.0f}s  "
              f"best=€{b_cost:,.0f}  SPs={len(b_sp)}  APLs={len(b_apl)}")
        if b_cost < overall_best_cost:
            overall_best_sp   = b_sp
            overall_best_apl  = b_apl
            overall_best_cost = b_cost
            overall_history   = hist

    # Restore global random state to RANDOM_SEED for reproducibility
    random.seed(RANDOM_SEED); np.random.seed(RANDOM_SEED)
    return (np.array(overall_best_sp, dtype=int),
            np.array(overall_best_apl, dtype=int),
            overall_best_cost, overall_history)

# MAP GENERATION
def generate_maps(res, cand_xy, s1_idx, scenarios_map):
    """
    Generate one interactive Folium map per scenario saved to maps/.
    scenarios_map: list of (label, sp_idx, apl_idx_or_None, result_dict)
    """
    try:
        import folium
        from pyproj import Transformer
    except ImportError:
        print("  [maps] folium / pyproj not installed — skipping map generation.")
        print("         pip install folium pyproj")
        return

    os.makedirs("maps", exist_ok=True)
    t = Transformer.from_crs("EPSG:28992", "EPSG:4326", always_xy=True)

    def to_latlon(x_local, y_local):
        # local → RD New (affine) → WGS84.  Scale matters: an offset alone is wrong.
        rd_x = RD_SCALE_X * np.atleast_1d(x_local) + RD_OFFSET_X
        rd_y = RD_SCALE_Y * np.atleast_1d(y_local) + RD_OFFSET_Y
        lon, lat = t.transform(rd_x, rd_y)
        return lat, lon

    # Pre-compute all CBS square centroids in WGS84
    sq_lats, sq_lons = to_latlon(res["X"].values, res["Y"].values)
    demand_vals = res["demand_annual"].values
    demand_max  = demand_vals.max()

    base_lats, base_lons = to_latlon(cand_xy[s1_idx, 0], cand_xy[s1_idx, 1])

    lat_min, lat_max = float(np.min(sq_lats)), float(np.max(sq_lats))
    lon_min, lon_max = float(np.min(sq_lons)), float(np.max(sq_lons))
    pad_lat = (lat_max - lat_min) * 0.05 or 0.01
    pad_lon = (lon_max - lon_min) * 0.05 or 0.01
    bounds = [[lat_min - pad_lat, lon_min - pad_lon],
              [lat_max + pad_lat, lon_max + pad_lon]]
    center_lat = (lat_min + lat_max) / 2
    center_lon = (lon_min + lon_max) / 2

    for label, sp_idx, apl_idx, result in scenarios_map:
        slug = label.split()[0].lower().replace("—","").strip()
        fname = f"maps/map_{slug}.html"

        # Full-window map; fit_bounds (added below) sets the zoom to the data.
        m = folium.Map(location=[center_lat, center_lon],
                       tiles="CartoDB positron",
                       width="100%", height="100%", control_scale=True)
        m.fit_bounds(bounds)

        # Layer: CBS squares (grey circles, sized by demand)
        sq_group = folium.FeatureGroup(name="CBS squares (demand)", show=True)
        d_km = result["_d_km"]
        for i, (slat, slon, dval, dk) in enumerate(
                zip(sq_lats, sq_lons, demand_vals, d_km)):
            radius = max(3, 12 * (dval / demand_max) ** 0.5)
            over_access = dk > MAX_ACCESS_KM
            folium.CircleMarker(
                location=[slat, slon],
                radius=radius,
                color="#E24B4A" if over_access else "#888780",
                fill=True,
                fill_color="#E24B4A" if over_access else "#B4B2A9",
                fill_opacity=0.5,
                weight=0.5,
                tooltip=(f"Daily demand: {dval/311:.1f}  "
                         f"Dist to SP: {dk:.2f} km"
                         + ("  ⚠ >3km" if over_access else ""))
            ).add_to(sq_group)
        sq_group.add_to(m)

        # Layer: baseline SPs (blue, reference)
        base_group = folium.FeatureGroup(name="Baseline SPs (S1, reference)", show=True)
        for blat, blon in zip(base_lats, base_lons):
            folium.CircleMarker(
                location=[blat, blon], radius=7,
                color="#185FA5", fill=True, fill_color="#378ADD",
                fill_opacity=0.6, weight=1.5,
                tooltip="Baseline SP (S1)"
            ).add_to(base_group)
        base_group.add_to(m)

        # Layer: scenario SPs (green)
        sp_lats, sp_lons = to_latlon(cand_xy[sp_idx, 0], cand_xy[sp_idx, 1])
        sp_group = folium.FeatureGroup(name=f"Scenario SPs ({label})", show=True)
        for slat, slon in zip(sp_lats, sp_lons):
            folium.Marker(
                location=[slat, slon],
                icon=folium.Icon(color="green", icon="star", prefix="fa"),
                tooltip=f"SP — {label}"
            ).add_to(sp_group)
        sp_group.add_to(m)

        # Layer: APLs (orange, S4 only)
        if apl_idx is not None and len(apl_idx) > 0:
            apl_lats, apl_lons = to_latlon(cand_xy[apl_idx, 0], cand_xy[apl_idx, 1])
            apl_group = folium.FeatureGroup(name="APLs", show=True)
            for alat, alon in zip(apl_lats, apl_lons):
                folium.Marker(
                    location=[alat, alon],
                    icon=folium.Icon(color="orange", icon="cube", prefix="fa"),
                    tooltip="APL"
                ).add_to(apl_group)
            apl_group.add_to(m)

        # Legend (inline HTML)
        viol = result.get("access_violations", 0)
        legend_html = f"""
        <div style="position:fixed;bottom:30px;left:30px;z-index:9999;
                    background:white;padding:12px 16px;border-radius:8px;
                    box-shadow:0 2px 8px rgba(0,0,0,.25);font-size:13px;
                    font-family:sans-serif;min-width:230px">
          <b style="font-size:14px">{label}</b><br>
          <span style="color:#555">SPs: {len(sp_idx)} &nbsp;|&nbsp;
          APLs: {len(apl_idx) if apl_idx is not None else 0}</span><br>
          <hr style="margin:6px 0">
          Total cost: <b>€{result['total']:,.0f}</b><br>
          Avg dist:   {result['avg_dist_km']:.3f} km<br>
          Pickup rate: {result['pickup_rate']:.1%}<br>
          Access violations: <b style="color:{'red' if viol>0 else 'green'}">{viol}</b>
          squares &gt;{MAX_ACCESS_KM}km<br>
          <hr style="margin:6px 0;border-top:1px solid #ddd">
          <span style="color:#888780">● CBS squares</span>&nbsp;
          <span style="color:#E24B4A">● &gt;{MAX_ACCESS_KM}km</span><br>
          <span style="color:#185FA5">● Baseline SPs</span>&nbsp;
          <span style="color:#27500A">★ Scenario SPs</span>
        </div>"""
        m.get_root().html.add_child(folium.Element(legend_html))
        folium.LayerControl(collapsed=False).add_to(m)
        m.save(fname)
        print(f"  Saved → {fname}")

def run_sensitivity(res, D_cand_km, demand, cand_xy, intercept, beta_dist, beta_inc,
                    clip_min, clip_max, avg_single, avg_working, oper_days,
                    s1_idx, baseline_cost, s3_idx_base):
    """
    One-at-a-time sensitivity sweep ±30% over six parameters.
    For each parameter value: re-run S1, S2 (fixed cost changes only revalue
    scenarios; MILP placement doesn't change for non-distance params), S3 sweep,
    and record optimal k and all scenario costs.
    """
    print("\n\n" + "=" * 65)
    print("  SENSITIVITY ANALYSIS  ")
    print("=" * 65)

    # Base parameter values
    base_params = {
        "fixed_cost_sp":   FIXED_COST_SP,
        "delivery_rate":   DELIVERY_RATE,
        "apl_uplift_avg":  APL_UPLIFT_AVG,
        "base_failed_rate": BASE_FAILED,
        "fixed_cost_apl":  FIXED_COST_APL,
        "peak_multiplier": PEAK_MULTIPLIER,
    }
    param_labels = {
        "fixed_cost_sp":    "SP fixed cost (€/yr)",
        "delivery_rate":    "Delivery rate (€/km)",
        "apl_uplift_avg":   "APL pickup uplift (pp)",
        "base_failed_rate": "Failed delivery rate",
        "fixed_cost_apl":   "APL fixed cost (€/yr)",
        "peak_multiplier":  "Peak demand multiplier",
    }
    steps = [-0.30, -0.15, 0.0, +0.15, +0.30]
    all_rows = []

    # Pre-compute S2 optimal positions (they only depend on distances, not costs)
    s2_idx, _, _ = solve_pmedian(35, D_cand_km, demand)

    def _cost_with(sp_idx, apl_idx=None, **kwargs):
        return compute_cost(sp_idx, res, intercept, beta_dist, beta_inc,
                            clip_min, clip_max, avg_single, avg_working,
                            D_cand_km, oper_days, apl_idx, **kwargs)

    def _get_failed_override(base_fr_val, res_df):
        """Rebuild per-square failed rates with a different city-mean anchor."""
        single = res_df["single_pct"].values
        working = res_df["working_pct"].values
        w = res_df["demand_annual"].values
        avg_s = float(np.average(single, weights=w))
        avg_w = float(np.average(working, weights=w))
        risk  = single + working
        hi, lo = int(risk.argmax()), int(risk.argmin())
        A = np.array([[single[hi]-avg_s, working[hi]-avg_w],
                      [single[lo]-avg_s, working[lo]-avg_w]])
        b = np.array([RATE_MAX-base_fr_val, RATE_MIN-base_fr_val])
        try:    a_s, b_w = np.linalg.solve(A, b)
        except: a_s, b_w = np.linalg.lstsq(A, b, rcond=None)[0]
        return np.clip(base_fr_val + a_s*(single-avg_s) + b_w*(working-avg_w),
                       RATE_MIN, RATE_MAX)

    for param_key, base_val in base_params.items():
        label = param_labels[param_key]
        print(f"  Sweeping: {label}")
        print(f"  {'Step':>6}  {'Value':>12}  {'S1 cost':>12}  {'S2 cost':>12}  "
              f"{'S3 opt-k':>8}  {'S3 cost':>12}")
        print(f"  {'─'*70}")

        for step in steps:
            pval = base_val * (1 + step)
            kwargs = {}
            failed_override = None

            if param_key == "fixed_cost_sp":
                kwargs["fixed_cost_sp"] = pval
            elif param_key == "delivery_rate":
                kwargs["delivery_rate"] = pval
            elif param_key == "apl_uplift_avg":
                kwargs["apl_uplift_avg"] = pval
            elif param_key == "base_failed_rate":
                failed_override = _get_failed_override(pval, res)
                kwargs["failed_rate_override"] = failed_override
            elif param_key == "fixed_cost_apl":
                kwargs["fixed_cost_apl"] = pval
            elif param_key == "peak_multiplier":
                kwargs["peak_multiplier"] = pval

            # S1: re-evaluate cost at current locations
            r1 = _cost_with(s1_idx, **kwargs)
            # S2: re-evaluate cost at optimal MILP locations (positions unchanged)
            r2 = _cost_with(s2_idx, **kwargs)

            # S3: sweep k to find new optimal (MILP positions fixed if only cost params change;
            # for delivery_rate the MILP objective changes too so we re-solve)
            best_k3, best_cost3, best_idx3 = None, np.inf, None
            for k in range(K_MIN, K_MAX+1):
                if param_key == "delivery_rate":
                    # Re-solve MILP with updated cost weighting — but MILP minimises
                    # weighted distance which doesn't change with delivery_rate;
                    # only the true cost changes. Re-use positions from base sweep.
                    idx_k = s3_idx_base.get(k)
                    if idx_k is None: continue
                else:
                    idx_k = s3_idx_base.get(k)
                    if idx_k is None: continue
                r_k = _cost_with(idx_k, **kwargs)
                if r_k["total"] < best_cost3:
                    best_cost3 = r_k["total"]
                    best_k3    = k
                    best_idx3  = idx_k

            tag = " ◄ base" if step == 0.0 else ""
            fmt_val = (f"€{pval:>9,.0f}" if "cost" in param_key or "rate" in param_key and "failed" not in param_key
                       else f"{pval:>11.4f}")
            print(f"  {step:>+6.0%}  {fmt_val}  "
                  f"€{r1['total']:>10,.0f}  €{r2['total']:>10,.0f}  "
                  f"{best_k3 if best_k3 else '?':>8}  €{best_cost3:>10,.0f}{tag}")

            all_rows.append({
                "parameter":     label,
                "step_%":        round(step*100, 0),
                "param_value":   round(pval, 6),
                "s1_total_cost": round(r1["total"]),
                "s2_total_cost": round(r2["total"]),
                "s3_optimal_k":  best_k3,
                "s3_total_cost": round(best_cost3) if best_cost3 < np.inf else None,
            })
        print()

    sa_df = pd.DataFrame(all_rows)
    sa_df.to_csv("sensitivity_analysis.csv", index=False)
    print(f"  Saved → sensitivity_analysis.csv")

    print(f"\n  {'─'*65}")
    print(f"  TORNADO SUMMARY — impact on S3 total cost (±30% swing)")
    print(f"  {'─'*65}")
    print(f"  {'Parameter':<35}  {'Low (−30%)':>12}  {'High (+30%)':>12}  {'Swing':>10}")
    print(f"  {'─'*65}")

    tornado_rows = []
    for label in param_labels.values():
        sub = sa_df[sa_df["parameter"] == label]
        low_row  = sub[sub["step_%"] == -30]["s3_total_cost"]
        high_row = sub[sub["step_%"] == +30]["s3_total_cost"]
        if low_row.empty or high_row.empty: continue
        low_c  = float(low_row.iloc[0])
        high_c = float(high_row.iloc[0])
        swing  = abs(high_c - low_c)
        tornado_rows.append((label, low_c, high_c, swing))
    tornado_rows.sort(key=lambda x: x[3], reverse=True)
    for label, low_c, high_c, swing in tornado_rows:
        print(f"  {label:<35}  €{low_c:>10,.0f}  €{high_c:>10,.0f}  €{swing:>8,.0f}")
    print(f"  {'─'*65}")

    return sa_df

# PRINT HELPERS

def print_cost(label, r, baseline_cost, n_sp, n_apl=0):
    saving = baseline_cost - r["total"]
    chg    = (r["total"]-baseline_cost)/baseline_cost*100
    print(f"\n{'─'*65}")
    print(f"  {label}")
    print(f"{'─'*65}")
    print(f"  SPs: {n_sp}   APLs: {n_apl}")
    print(f"  Total cost   : €{r['total']:,.0f}  ({chg:+.1f}% vs baseline, saving €{saving:,.0f}/yr)")
    print(f"  Fixed        : €{r['fixed']:,.0f}  ({r['fixed']/r['total']:.1%})")
    print(f"  Storage      : €{r['storage']:,.0f}  ({r['storage']/r['total']:.1%})")
    print(f"  Delivery     : €{r['delivery']:,.0f}  ({r['delivery']/r['total']:.1%})")
    print(f"  Pickup rate  : {r['pickup_rate']:.1%}")
    print(f"  Avg dist     : {r['avg_dist_km']:.3f} km")
    print(f"  Max dist     : {r['max_dist_km']:.3f} km")
    # Accessibility report
    viol = r.get("access_violations", 0)
    unserved = r.get("demand_underserved", 0.0)
    tick_acc = "✓" if viol == 0 else "✗"
    print(f"  Access (>{MAX_ACCESS_KM}km): {viol} squares  "
          f"({unserved:,.0f} annual parcels)  {tick_acc}")
    overflow = r.get("sp_overflow_cost", 0.0)
    max_load = r.get("sp_max_load", 0.0)
    print(f"  SP overflow cost : €{overflow:,.0f}  (peak load on busiest SP: {max_load:.0f} parcels/day)")
    if n_apl > 0:
        tick = lambda v, lim: "✓" if v <= lim else "✗"
        print(f"  APL per-unit : {r['apl_max_bounce']:.2%} (≤{BOUNCE_PER_APL:.0%}) "
              f"{tick(r['apl_max_bounce'],BOUNCE_PER_APL)}  [peak-day]")
        print(f"  APL city     : {r['apl_bounce_rate']:.2%} (≤{BOUNCE_CITY:.0%}) "
              f"{tick(r['apl_bounce_rate'],BOUNCE_CITY)}")

def print_breakdown(bd):
    print(f"\n  {'Facility':<8}  {'Ann.demand':>11}  {'Daily':>7}  "
          f"{'Pickup%':>8}  {'Avg dist':>9}  {'Max dist':>9}  {'Total cost':>11}")
    print(f"  {'─'*80}")
    for _, row in bd.iterrows():
        print(f"  {row['Facility']:<8}  {row['annual_demand']:>11,.0f}  "
              f"{row['daily_demand']:>7.1f}  {row['pickup_%']:>7.1f}%  "
              f"{row['avg_dist_km']:>8.3f}km  {row['max_dist_km']:>8.3f}km  "
              f"€{row['total_cost']:>9,.0f}")
    print(f"  {'─'*80}")
    print(f"  {'TOTAL':<8}  {bd['annual_demand'].sum():>11,.0f}  "
          f"{'':>7}  {'':>8}  {'':>9}  {'':>9}  €{bd['total_cost'].sum():>9,.0f}")

# IMPLIED CAPACITY ANALYSIS
def compute_implied_capacity(scenarios_cap, oper_days):
    """
    For each facility in each scenario, compute the daily storage load implied
    by the optimised demand assignment, at both average and peak (×PEAK_MULTIPLIER)
    conditions.

    Storage load = (annual_pickups × dwell_pickup + annual_failed × dwell_failed)
                   / oper_days   → average daily parcels sitting at the facility.
    Peak load    = average × PEAK_MULTIPLIER.

    This answers the brief's capacity question without needing an externally
    given SP capacity: it tells Post&L how large each facility needs to be.

    Parameters
    scenarios_cap : list of (label, result_dict, n_sp, n_apl)
    oper_days     : int
    """
    print("\n" + "=" * 65)
    print("  IMPLIED FACILITY CAPACITY  ")
    print("=" * 65)
    print(f"  Peak multiplier : ×{PEAK_MULTIPLIER}  (observed city-wide p95/mean ratio)")
    print(f"  Dwell — SP pickup : {DWELL_PICKUP} days  |  "
          f"Failed delivery : {DWELL_FAILED} days  |  APL : {DWELL_APL} day")
    print(f"  Interpretation: 'implied capacity' = parcels in storage on that day\n")

    all_rows = []

    for label, result, n_sp, n_apl in scenarios_cap:
        n_fac    = n_sp + n_apl
        nearest  = result["_nearest"]
        pkp_arr  = result["_annual_pickups"]
        fail_arr = result["_annual_failed"]

        print(f"  {'─'*65}")
        print(f"  {label}")
        print(f"  {'─'*65}")
        print(f"  {'Facility':<10}  {'Type':<4}  "
              f"{'Avg daily storage':>18}  {'Peak daily storage':>18}  {'Flag':>6}")
        print(f"  {'─'*65}")

        for j in range(n_fac):
            mask = nearest == j
            if not mask.any():
                continue
            is_apl = j >= n_sp
            fac_label = f"APL {j-n_sp+1}" if is_apl else f"SP {j+1}"
            fac_type  = "APL" if is_apl else "SP"
            dwell     = DWELL_APL if is_apl else DWELL_PICKUP

            annual_stored = (float(pkp_arr[mask].sum()) * dwell
                             + float(fail_arr[mask].sum()) * DWELL_FAILED)
            avg_daily  = annual_stored / max(oper_days, 1)
            peak_daily = avg_daily * PEAK_MULTIPLIER

            if is_apl:
                flag = "✗ OVER" if peak_daily > K_APL_CAP else "✓"
            else:
                flag = "⚠ HIGH" if peak_daily > 900 else "✓"

            sp_over = bool(peak_daily > 900) if not is_apl else None

            print(f"  {fac_label:<10}  {fac_type:<4}  "
                  f"{avg_daily:>17.1f}  {peak_daily:>17.1f}  {flag:>6}")

            all_rows.append({
                "scenario":          label,
                "facility":          fac_label,
                "type":              fac_type,
                "avg_daily_storage": round(avg_daily, 1),
                "peak_daily_storage": round(peak_daily, 1),
                "apl_capacity":      K_APL_CAP if is_apl else None,
                "apl_over_capacity": bool(peak_daily > K_APL_CAP) if is_apl else None,
                "sp_flag_high":      sp_over,
            })

        # Scenario summary line
        sp_rows  = [r for r in all_rows if r["scenario"] == label and r["type"] == "SP"]
        apl_rows = [r for r in all_rows if r["scenario"] == label and r["type"] == "APL"]
        if sp_rows:
            print(f"\n  SP  avg storage — mean: "
                  f"{sum(r['avg_daily_storage'] for r in sp_rows)/len(sp_rows):.1f}  "
                  f"max: {max(r['avg_daily_storage'] for r in sp_rows):.1f}  "
                  f"| peak — mean: "
                  f"{sum(r['peak_daily_storage'] for r in sp_rows)/len(sp_rows):.1f}  "
                  f"max: {max(r['peak_daily_storage'] for r in sp_rows):.1f}")
        if apl_rows:
            print(f"  APL avg storage — mean: "
                  f"{sum(r['avg_daily_storage'] for r in apl_rows)/len(apl_rows):.1f}  "
                  f"max: {max(r['avg_daily_storage'] for r in apl_rows):.1f}  "
                  f"| peak — mean: "
                  f"{sum(r['peak_daily_storage'] for r in apl_rows)/len(apl_rows):.1f}  "
                  f"max: {max(r['peak_daily_storage'] for r in apl_rows):.1f}")

    cap_df = pd.DataFrame(all_rows)
    cap_df.to_csv("implied_capacity.csv", index=False)
    print(f"\n  Saved → implied_capacity.csv")
    return cap_df

# MAIN

def main():
    (res, sp_df, intercept, beta_dist, beta_inc,
     clip_min, clip_max, oper_days, avg_single, avg_working,
     D_cand_km) = build_demand_model()
    cand_xy = res[["X","Y"]].values.astype(float)
    demand  = res["demand_annual"].values
    n_cand  = len(cand_xy)

    # Snap baseline SPs to unique CBS centroids
    sp_xy_raw = sp_df[["X","Y"]].values.astype(float)
    raw_dists, raw_idx = cKDTree(cand_xy).query(sp_xy_raw, k=len(cand_xy))
    claimed = set()
    s1_idx  = np.empty(len(sp_xy_raw), dtype=int)
    for sp_i in range(len(sp_xy_raw)):
        for cand_j in raw_idx[sp_i]:
            if cand_j not in claimed:
                s1_idx[sp_i] = cand_j
                claimed.add(cand_j)
                break

    def cost(sp_idx, apl_idx=None, **kwargs):
        return compute_cost(
            sp_idx, res, intercept, beta_dist, beta_inc,
            clip_min, clip_max, avg_single, avg_working,
            D_cand_km, oper_days,
            apl_idx if apl_idx is not None and len(apl_idx) > 0 else None,
            **kwargs)

    # SCENARIO 1
    print("\n" + "=" * 65)
    print("  SCENARIO 1 — Baseline: 35 SPs, current locations")
    print("=" * 65)
    s1            = cost(s1_idx)
    baseline_cost = s1["total"]
    print_cost("Scenario 1 — Baseline", s1, baseline_cost, 35)
    bd1 = per_facility_breakdown(cand_xy[s1_idx], None, s1, oper_days,
                                 sp_idx=s1_idx, D_cand_km=D_cand_km)
    print_breakdown(bd1)

    # SCENARIO 2 — RESIZE EXISTING SITES  (closure-only, no new SPs)
    # Merges the old relocate+resize: since Post&L will not open new sites, the
    # only lever is which of the existing 35 to keep. Sweep the count k over the
    # existing sites only and keep the cheapest subset.
    print("\n" + "=" * 65)
    print("  SCENARIO 2 — Resize existing sites: best subset of the 35")
    print("=" * 65)
    D_keep = D_cand_km[:, s1_idx]                 # distances to existing sites only
    print(f"\n  {'k':>4}  {'True cost':>13}  {'vs Baseline':>12}  {'Time':>5}")
    print(f"  {'─'*36}")
    sweep2 = []
    for k in range(K_MIN, 36):                    # cannot keep more than the 35 that exist
        loc_idx, _, t = solve_pmedian(k, D_keep, demand)
        if loc_idx is None:
            continue
        sub_idx = s1_idx[loc_idx]                 # local (0..34) → global candidate index
        r    = cost(sub_idx)
        feas = r["sp_feasible"]
        chg  = (r["total"] - baseline_cost) / baseline_cost * 100
        feas_so_far = [s for s in sweep2 if s["feasible"]]
        best = feas and (not feas_so_far or r["total"] < min(s["cost"] for s in feas_so_far))
        flag = "  ◄ best" if best else ("" if feas else "  ✗ over-cap")
        print(f"  {k:>4}  €{r['total']:>11,.0f}  {chg:>+11.1f}%  {t:>4.1f}s{flag}")
        sweep2.append({"k": k, "cost": r["total"], "result": r, "idx": sub_idx, "feasible": feas})

    feas2 = [s for s in sweep2 if s["feasible"]]
    if feas2:
        best2 = min(feas2, key=lambda s: s["cost"])
    else:
        best2 = min(sweep2, key=lambda s: s["cost"])
        print("\n  ⚠ No capacity-feasible subset of the existing sites; reporting cheapest.")
    s2_idx = best2["idx"]; s2 = best2["result"]; s2_k = best2["k"]
    print_cost(f"Scenario 2 — Resize existing ({s2_k} of 35 SPs)", s2, baseline_cost, s2_k)
    bd2 = per_facility_breakdown(cand_xy[s2_idx], None, s2, oper_days,
                                 sp_idx=s2_idx, D_cand_km=D_cand_km)
    print_breakdown(bd2)
    pd.DataFrame(cand_xy[s2_idx], columns=["X","Y"]).to_csv("s2_sp_locations.csv", index=False)
    bd2.to_csv("s2_breakdown.csv", index=False)
    pd.DataFrame([{"k": s["k"], "total_cost": round(s["cost"]),
                   "vs_baseline_%": round((s["cost"]-baseline_cost)/baseline_cost*100, 2)}
                  for s in sweep2]).to_csv("s2_sweep.csv", index=False)

    # SCENARIO 3
    print("\n" + "=" * 65)
    print("  SCENARIO 3 — Free-siting BENCHMARK (ignores the no-new-sites rule)")
    print(f"  Optimal k in [{K_MIN},{K_MAX}], sited at ANY CBS centroid.")
    print("  Not a recommendation — measures the opportunity cost of the rule.")
    print("=" * 65)
    print(f"\n  {'k':>4}  {'True cost':>13}  {'vs Baseline':>12}  {'Time':>5}")
    print(f"  {'─'*36}")
    sweep3 = []; s3_idx_base = {}
    for k in range(K_MIN, K_MAX+1):
        idx_k, _, t = solve_pmedian(k, D_cand_km, demand)
        if idx_k is None: continue
        r    = cost(idx_k)
        feas = r["sp_feasible"]
        chg  = (r["total"]-baseline_cost)/baseline_cost*100
        feas_so_far = [s for s in sweep3 if s["feasible"]]
        best = feas and (not feas_so_far or r["total"] < min(s["cost"] for s in feas_so_far))
        flag = "  ◄ best" if best else ("" if feas else "  ✗ over-cap")
        print(f"  {k:>4}  €{r['total']:>11,.0f}  {chg:>+11.1f}%  {t:>4.1f}s{flag}")
        sweep3.append({"k":k,"cost":r["total"],"result":r,"idx":idx_k,"feasible":feas})
        s3_idx_base[k] = idx_k   # store all k solutions for sensitivity reuse

    feas3 = [s for s in sweep3 if s["feasible"]]
    best3 = min(feas3, key=lambda s: s["cost"]) if feas3 else min(sweep3, key=lambda s: s["cost"])
    s3_idx = best3["idx"]; s3 = best3["result"]; s3_k = best3["k"]
    print_cost(f"Scenario 3 — Free-siting benchmark ({s3_k} SPs)", s3, baseline_cost, s3_k)
    bd3 = per_facility_breakdown(cand_xy[s3_idx], None, s3, oper_days,
                                 sp_idx=s3_idx, D_cand_km=D_cand_km)
    print_breakdown(bd3)
    pd.DataFrame(cand_xy[s3_idx], columns=["X","Y"]).to_csv("s3_sp_locations.csv", index=False)
    bd3.to_csv("s3_breakdown.csv", index=False)
    pd.DataFrame([{"k":s["k"],"total_cost":round(s["cost"]),
                   "vs_baseline_%":round((s["cost"]-baseline_cost)/baseline_cost*100,2)}
                  for s in sweep3]).to_csv("s3_sweep.csv", index=False)

    # SCENARIO 4
    print("\n" + "=" * 65)
    print("  SCENARIO 4 — Existing SPs (closure-only) + free APLs (Simulated Annealing)")
    print("=" * 65)
    t0 = time.time()
    s4_sp_idx, s4_apl_idx, s4_cost, s4_hist = simulated_annealing(
        list(s2_idx), cand_xy, res, intercept, beta_dist, beta_inc,
        clip_min, clip_max, avg_single, avg_working, baseline_cost,
        D_cand_km, oper_days, sp_allowed_idx=s1_idx)
    sa_time = time.time() - t0
    s4 = cost(s4_sp_idx, s4_apl_idx if len(s4_apl_idx) > 0 else None)
    print_cost(f"Scenario 4 — Full ({len(s4_sp_idx)} SPs + {len(s4_apl_idx)} APLs)",
               s4, baseline_cost, len(s4_sp_idx), len(s4_apl_idx))
    if len(s4_apl_idx) == 0:
        print("  Note: SA found no feasible APLs that reduce cost")
    bd4 = per_facility_breakdown(
        cand_xy[s4_sp_idx],
        cand_xy[s4_apl_idx] if len(s4_apl_idx) > 0 else None,
        s4, oper_days, sp_idx=s4_sp_idx, D_cand_km=D_cand_km)
    print_breakdown(bd4)
    pd.DataFrame(cand_xy[s4_sp_idx], columns=["X","Y"]).to_csv("s4_sp_locations.csv", index=False)
    if len(s4_apl_idx) > 0:
        pd.DataFrame(cand_xy[s4_apl_idx], columns=["X","Y"]).to_csv("s4_apl_locations.csv", index=False)
    bd4.to_csv("s4_breakdown.csv", index=False)
    pd.DataFrame(s4_hist, columns=["time_s","best_cost"]).to_csv("s4_sa_history.csv", index=False)

    # COMPARISON TABLE
    print("\n\n" + "=" * 65)
    print("  SCENARIO COMPARISON")
    print("=" * 65)
    scenarios = [
        ("S1 — Baseline (35)",                        s1,  35,   0),
        (f"S2 — Resize existing ({s2_k} SPs)",        s2,  s2_k, 0),
        (f"S3 — Free-siting benchmark ({s3_k} SPs)",  s3,  s3_k, 0),
        (f"S4 — Existing SPs+APLs ({len(s4_sp_idx)} SPs + {len(s4_apl_idx)} APLs)",
                                           s4,  len(s4_sp_idx), len(s4_apl_idx)),
    ]
    print(f"\n  {'Scenario':<35}  {'SPs':>4}  {'APLs':>4}  "
          f"{'Total cost':>12}  {'vs S1':>8}  {'Saving/yr':>11}  "
          f"{'Pickup%':>7}  {'Avg dist':>9}  {'Max dist':>9}")
    print(f"  {'─'*114}")
    comp_rows = []
    for label, r, n_sp, n_apl in scenarios:
        chg    = (r["total"]-baseline_cost)/baseline_cost*100
        saving = baseline_cost - r["total"]
        print(f"  {label:<35}  {n_sp:>4}  {n_apl:>4}  "
              f"€{r['total']:>10,.0f}  {chg:>+7.1f}%  €{saving:>9,.0f}  "
              f"{r['pickup_rate']:>6.1%}  {r['avg_dist_km']:>8.3f} km  {r['max_dist_km']:>8.3f} km")
        comp_rows.append({
            "scenario": label, "n_sp": n_sp, "n_apl": n_apl,
            "total_cost": round(r["total"]),
            "vs_baseline_%": round(chg, 2),
            "saving_vs_s1": round(saving),
            "fixed_cost": round(r["fixed"]),
            "storage_cost": round(r["storage"]),
            "delivery_cost": round(r["delivery"]),
            "pickup_rate": round(r["pickup_rate"], 4),
            "avg_dist_km": round(r["avg_dist_km"], 4),
            "max_dist_km": round(r["max_dist_km"], 4),
        })
    print(f"\n  Cost breakdown:")
    print(f"  {'Scenario':<35}  {'Fixed':>12}  {'Storage':>10}  "
          f"{'Delivery':>10}  {'Fixed %':>8}  {'Delivery %':>10}")
    print(f"  {'─'*90}")
    for label, r, n_sp, n_apl in scenarios:
        print(f"  {label:<35}  €{r['fixed']:>10,.0f}  €{r['storage']:>8,.0f}  "
              f"€{r['delivery']:>8,.0f}  {r['fixed']/r['total']:>7.1%}  "
              f"{r['delivery']/r['total']:>9.1%}")
    pd.DataFrame(comp_rows).to_csv("comparison.csv", index=False)

    capacity_scenarios = [
        ("S1 — Baseline (35)",                        s1, 35,   0),
        (f"S2 — Resize existing ({s2_k} SPs)",        s2, s2_k, 0),
        (f"S3 — Free-siting benchmark ({s3_k} SPs)",  s3, s3_k, 0),
        (f"S4 — Existing SPs+APLs ({len(s4_sp_idx)} SPs + {len(s4_apl_idx)} APLs)",
                                           s4, len(s4_sp_idx), len(s4_apl_idx)),
    ]
    compute_implied_capacity(capacity_scenarios, oper_days)

    run_sensitivity(res, D_cand_km, demand, cand_xy,
                    intercept, beta_dist, beta_inc, clip_min, clip_max,
                    avg_single, avg_working, oper_days,
                    s1_idx, baseline_cost, s3_idx_base)

    scenarios_map = [
        ("S1 — Baseline",                                s1_idx, None, s1),
        (f"S2 — Resize existing ({s2_k} SPs)",           s2_idx, None, s2),
        (f"S3 — Free-siting benchmark ({s3_k} SPs)",     s3_idx, None, s3),
        (f"S4 — Existing+APLs ({len(s4_sp_idx)}SP+{len(s4_apl_idx)}APL)",
                                                         s4_sp_idx,
                                                         s4_apl_idx if len(s4_apl_idx)>0 else None,
                                                         s4),
    ]
    generate_maps(res, cand_xy, s1_idx, scenarios_map)

    print(f"\n{'=' * 65}")
    print(f"  Saved → comparison.csv  sensitivity_analysis.csv  implied_capacity.csv")
    print(f"  Saved → s2/s3/s4_sp_locations!!!!.csv  s2/s3/s4_breakdown!!!.csv")
    print(f"  Saved → s2_sweep!!.csv  s3_sweep!!.csv  s4_sa_history!!.csv")
    print(f"  Saved → maps/map_s1.html … map_s4.html")
    print("=" * 65)

if __name__ == "__main__":
    main()