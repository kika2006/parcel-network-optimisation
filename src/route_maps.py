"""
route_optimizer_dijkstra.py  —  Post&L Maastricht
==================================================
Recalculates all route distances using the SAME Dijkstra road network
as all_scenarios_v6_FINISHED.py, then solves the CVRP and generates
Folium maps with the real routes drawn on them.

Run from your Case 2 folder:
    python route_optimizer_dijkstra.py

Outputs:
    routes_S3_dijkstra.csv
    routes_S4_dijkstra.csv
    map_s3_routes.html
    map_s4_routes.html
"""

import math, os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra as sp_dijkstra
from scipy.spatial import cKDTree
from pyproj import Transformer
import folium

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH = "data/data_Maastricht_2025.xlsx"
VAN_CAP   = 300        # parcels per large sprinter
RATE      = 1.50       # €/km
OPER_DAYS = 311

# Affine: local → RD New (metres)
RD_SCALE_X, RD_OFFSET_X = 0.06949351, 172244.5245
RD_SCALE_Y, RD_OFFSET_Y = 0.10824467, 312384.3812

# Depot: PostNL Born in LOCAL coordinate system
# (lat=51.06887, lon=5.81849 → RD New → local)
# rd_x=185231 → local_x=(185231-172244.5)/0.06949351 = 186868
# rd_y=342235 → local_y=(342235-312384.4)/0.10824467 = 275768
DEPOT_LOCAL = (186_868.0, 275_768.0)

# ── Coordinate helpers ────────────────────────────────────────────────────────
_t = Transformer.from_crs("EPSG:28992", "EPSG:4326", always_xy=True)

def local_to_latlon(lx, ly):
    rd_x = lx * RD_SCALE_X + RD_OFFSET_X
    rd_y = ly * RD_SCALE_Y + RD_OFFSET_Y
    lon, lat = _t.transform(rd_x, rd_y)
    return lat, lon

DEPOT_LATLON = local_to_latlon(*DEPOT_LOCAL)

# ── Step 1: Load road network ─────────────────────────────────────────────────
print("=" * 60)
print("  Loading road network from Excel...")
print("=" * 60)

nodes_df = pd.read_excel(DATA_PATH, sheet_name="Nodes")
edges_df = pd.read_excel(DATA_PATH, sheet_name="Edges")
nodes_df = nodes_df[[c for c in nodes_df.columns if not str(c).startswith("Unnamed")]]
edges_df = edges_df[[c for c in edges_df.columns if not str(c).startswith("Unnamed")]]

v1 = (edges_df["V1"] - 1).values.astype(np.int32)
v2 = (edges_df["V2"] - 1).values.astype(np.int32)
wt = edges_df["DIST"].values.astype(np.float64)   # metres
n  = int(max(edges_df["V1"].max(), edges_df["V2"].max()))

graph = csr_matrix(
    (np.concatenate([wt, wt]),
     (np.concatenate([v1, v2]), np.concatenate([v2, v1]))),
    shape=(n, n)
)

node_xy   = nodes_df[["X", "Y"]].values
node_ids0 = (nodes_df["NODE ID"] - 1).values.astype(int)
kdt       = cKDTree(node_xy)
print(f"  Nodes: {len(nodes_df):,}   Edges: {len(edges_df):,}")

# ── Step 2: Load S3 and S4 facility locations ─────────────────────────────────
print("\n  Loading facility locations...")

s3_bd = pd.read_csv("s3_breakdown.csv")
s4_bd = pd.read_csv("s4_breakdown.csv")
s3_sp = pd.read_csv("s3_sp_locations.csv")
s4_sp = pd.read_csv("s4_sp_locations.csv")
s4_ap = pd.read_csv("s4_apl_locations.csv")

# Attach facility names and daily demand
s3_sp["facility"] = [f"SP {i+1}" for i in range(len(s3_sp))]
s4_sp["facility"] = [f"SP {i+1}" for i in range(len(s4_sp))]
s4_ap["facility"] = [f"APL {i+1}" for i in range(len(s4_ap))]
s4_ap["type"]     = "APL"
s4_sp["type"]     = "SP"
s3_sp["type"]     = "SP"

demand_map_s3 = dict(zip(s3_bd["Facility"], s3_bd["daily_demand"]))
demand_map_s4 = dict(zip(s4_bd["Facility"], s4_bd["daily_demand"]))

s3_facs = s3_sp.copy()
s3_facs["daily_demand"] = s3_facs["facility"].map(demand_map_s3).fillna(0)

s4_facs = pd.concat([s4_sp, s4_ap], ignore_index=True).reset_index(drop=True)
s4_facs["daily_demand"] = s4_facs["facility"].map(demand_map_s4).fillna(0)

print(f"  S3: {len(s3_facs)} SPs")
print(f"  S4: {len(s4_sp)} SPs + {len(s4_ap)} APLs")

# ── Step 3: Snap depot + facilities to road nodes ─────────────────────────────
def snap_to_node(xy_arr):
    """Return road node IDs (0-based) for an array of (X,Y) local coords."""
    _, idx = kdt.query(xy_arr)
    return node_ids0[idx]

depot_node = snap_to_node([DEPOT_LOCAL])[0]
print(f"\n  Depot road node: {depot_node}")

# ── Step 4: Dijkstra from depot to all facilities ─────────────────────────────
def get_road_distances_km(facs_df):
    """
    Run Dijkstra from the depot node and from every facility node.
    Returns a dict: {facility_name: road_distance_km_from_depot}
    and the full pairwise distance matrix between [depot] + facilities (km).
    """
    fac_xy    = facs_df[["X", "Y"]].values
    fac_nodes = snap_to_node(fac_xy)

    # Sources: depot + all facility nodes
    all_sources = np.unique(np.concatenate([[depot_node], fac_nodes]))
    s2r = {int(nd): i for i, nd in enumerate(all_sources)}

    print(f"  Running Dijkstra from {len(all_sources)} sources...")
    D = sp_dijkstra(graph, directed=False, indices=all_sources)
    D = np.where(np.isinf(D), 1e9, D)

    # Build pairwise distance matrix between [depot] + facilities (in km)
    all_pts   = [depot_node] + list(fac_nodes)
    n_pts     = len(all_pts)
    dist_km   = np.zeros((n_pts, n_pts))
    for i, ni in enumerate(all_pts):
        row_i = s2r[int(ni)]
        for j, nj in enumerate(all_pts):
            dist_km[i, j] = D[row_i, int(nj)] / 1000.0

    depot_to_fac = {facs_df.iloc[j]["facility"]: dist_km[0, j+1]
                    for j in range(len(facs_df))}
    return dist_km, depot_to_fac, fac_nodes

print("\n" + "="*60)
print("  S3 — Dijkstra road distances")
print("="*60)
s3_dist_km, s3_depot_dists, s3_nodes = get_road_distances_km(s3_facs)

print("\n" + "="*60)
print("  S4 — Dijkstra road distances")
print("="*60)
s4_dist_km, s4_depot_dists, s4_nodes = get_road_distances_km(s4_facs)

# Print depot distances
print("\n  Depot (Born) → facility road distances:")
for sc, dd in [("S3", s3_depot_dists), ("S4", s4_depot_dists)]:
    print(f"\n  {sc}:")
    for fac, km in sorted(dd.items()):
        print(f"    {fac:<8}  {km:.2f} km")

# ── Step 5: Clarke-Wright CVRP with road distances ────────────────────────────
def clarke_wright(dist_km, demands, cap):
    """
    Clarke-Wright Parallel Savings CVRP.
    dist_km[0] = depot, dist_km[1..n] = facilities (0-based into demands).
    Returns list of routes (list of facility indices, 0-based into demands).
    """
    n = len(demands)
    if n == 0:
        return []

    routes  = [[i] for i in range(n)]
    r_load  = [float(demands[i]) for i in range(n)]
    s2r     = list(range(n))

    savings = sorted(
        ((dist_km[0, i+1] + dist_km[0, j+1] - dist_km[i+1, j+1], i, j)
         for i in range(n) for j in range(i+1, n)),
        reverse=True,
    )

    for _, i, j in savings:
        ri, rj = s2r[i], s2r[j]
        if ri == rj or not routes[ri] or not routes[rj]:
            continue
        ri_r, rj_r = routes[ri], routes[rj]
        i_tail = ri_r[-1] == i;  i_head = ri_r[0] == i
        j_tail = rj_r[-1] == j;  j_head = rj_r[0] == j
        if i_tail and j_head:
            new_route = ri_r + rj_r
        elif j_tail and i_head:
            new_route = rj_r + ri_r
        else:
            continue
        if r_load[ri] + r_load[rj] > cap:
            continue
        routes[ri] = new_route
        r_load[ri] += r_load[rj]
        routes[rj]  = []
        r_load[rj]  = 0.0
        for s in new_route:
            s2r[s] = ri

    return [r for r in routes if r]


def solve_scenario(label, facs_df, dist_km, depot_dists):
    print(f"\n{'='*60}")
    print(f"  CVRP — {label}")
    print(f"{'='*60}")

    demands = facs_df["daily_demand"].values.astype(float)
    names   = facs_df["facility"].values

    # Split facilities that exceed VAN_CAP into virtual stops
    v_coords, v_demands, v_labels, v_fac = [], [], [], []
    for i, row in facs_df.iterrows():
        d     = float(row["daily_demand"])
        trips = max(1, math.ceil(d / VAN_CAP))
        per   = d / trips
        for t in range(trips):
            v_coords.append(i)   # original fac index
            v_demands.append(per)
            lbl = (f"{row['facility']} [trip {t+1}/{trips}]"
                   if trips > 1 else row["facility"])
            v_labels.append(lbl)
            v_fac.append(row["facility"])

    # Build virtual distance matrix
    n_v   = len(v_coords)
    v_dist = np.zeros((n_v + 1, n_v + 1))
    # index 0 = depot
    for vi in range(n_v):
        fi = v_coords[vi]
        v_dist[0, vi+1] = dist_km[0, fi+1]
        v_dist[vi+1, 0] = dist_km[fi+1, 0]
        for vj in range(n_v):
            fj = v_coords[vj]
            v_dist[vi+1, vj+1] = dist_km[fi+1, fj+1]

    routes = clarke_wright(v_dist, v_demands, VAN_CAP)

    # Build results
    rows = []
    total_km = 0.0
    for idx, route in enumerate(routes):
        km   = v_dist[0, route[0]+1]
        for k in range(len(route)-1):
            km += v_dist[route[k]+1, route[k+1]+1]
        km += v_dist[route[-1]+1, 0]
        load      = sum(v_demands[s] for s in route)
        stops_str = " → ".join(v_labels[s] for s in route)
        total_km += km
        rows.append({
            "scenario":       label,
            "route":          idx + 1,
            "n_stops":        len(route),
            "load_parcels":   round(load, 1),
            "utilisation_%":  round(load / VAN_CAP * 100, 1),
            "distance_km":    round(km, 2),
            "route_cost_eur": round(km * RATE, 2),
            "stops":          stops_str,
            "_route_indices": route,
            "_v_labels":      v_labels,
            "_v_coords":      v_coords,
        })

    df = pd.DataFrame(rows)
    n_vans     = len(routes)
    total_cost = round(total_km * RATE, 2)
    avg_util   = df["utilisation_%"].mean()

    # Depot distances summary
    min_d = min(depot_dists.values())
    max_d = max(depot_dists.values())
    mean_d = sum(depot_dists.values()) / len(depot_dists)

    print(f"\n  Depot→facility road distances:")
    print(f"    min={min_d:.1f} km  max={max_d:.1f} km  mean={mean_d:.1f} km")
    print(f"\n  {'Rt':>3}  {'Stops':>5}  {'Load':>6}  {'Util':>6}  "
          f"{'Dist km':>8}  {'Cost €':>8}  Facilities")
    print(f"  {'─'*80}")
    for _, r in df.iterrows():
        print(f"  {int(r['route']):>3}  {int(r['n_stops']):>5}  "
              f"{r['load_parcels']:>6.0f}  {r['utilisation_%']:>5.1f}%  "
              f"{r['distance_km']:>8.2f}  {r['route_cost_eur']:>8.2f}  "
              f"{r['stops']}")

    print(f"\n  {'─'*50}")
    print(f"  Van trips/day        : {n_vans}")
    print(f"  Total distance/day   : {total_km:.2f} km")
    print(f"  First-mile cost/day  : €{total_cost:,.2f}")
    print(f"  Annual first-mile    : €{total_cost*OPER_DAYS:,.0f}")
    print(f"  Avg utilisation      : {avg_util:.1f}%")

    # Save CSV (drop internal columns)
    out_df = df.drop(columns=["_route_indices","_v_labels","_v_coords"])
    csv_name = f"routes_{label.replace(' ','_')}_dijkstra.csv"
    out_df.to_csv(csv_name, index=False)
    print(f"  Saved → {csv_name}")

    return df, total_km, total_cost, n_vans, avg_util


s3_df, s3_km, s3_cost, s3_vans, s3_util = solve_scenario(
    "S3", s3_facs, s3_dist_km, s3_depot_dists)
s4_df, s4_km, s4_cost, s4_vans, s4_util = solve_scenario(
    "S4", s4_facs, s4_dist_km, s4_depot_dists)

# Comparison
print(f"\n{'='*60}")
print(f"  S3 vs S4 — Dijkstra Road Distance Comparison")
print(f"{'='*60}")
print(f"\n  S3: {s3_vans} trips  {s3_km:.1f} km/day  €{s3_cost:,.2f}/day  "
      f"€{s3_cost*OPER_DAYS:,.0f}/yr  util={s3_util:.1f}%")
print(f"  S4: {s4_vans} trips  {s4_km:.1f} km/day  €{s4_cost:,.2f}/day  "
      f"€{s4_cost*OPER_DAYS:,.0f}/yr  util={s4_util:.1f}%")
saving_day  = s3_cost - s4_cost
saving_year = saving_day * OPER_DAYS
print(f"\n  S4 saves €{saving_day:.2f}/day = €{saving_year:,.0f}/year vs S3")


# ── Step 6: Generate Folium maps ──────────────────────────────────────────────
import requests, time

def get_road_geometry(coords_latlon: list) -> list:
    """
    Fetch actual road path geometry from OSRM public API.
    coords_latlon: list of (lat, lon) tuples — depot + stops + depot
    Returns list of (lat, lon) tuples following the road.
    Falls back to straight lines if API fails.
    """
    # OSRM expects lon,lat order
    waypoints = ";".join(f"{lon},{lat}" for lat, lon in coords_latlon)
    url = f"http://router.project-osrm.org/route/v1/driving/{waypoints}"
    params = {"overview": "full", "geometries": "geojson", "steps": "false"}
    try:
        r = requests.get(url, params=params, timeout=10)
        if r.status_code == 200:
            data = r.json()
            if data.get("code") == "Ok":
                geom = data["routes"][0]["geometry"]["coordinates"]
                # OSRM returns [lon, lat] — flip to (lat, lon) for Folium
                return [(pt[1], pt[0]) for pt in geom]
    except Exception:
        pass
    # Fallback: straight lines
    return coords_latlon


ROUTE_COLORS = ["#F97316","#34D399","#FBBF24","#F472B6","#60A5FA",
                "#A78BFA","#22D3EE","#FB923C","#86EFAC","#FCA5A5",
                "#C084FC","#6EE7B7","#93C5FD","#FDE68A","#67E8F9",
                "#FDBA74","#6EE7B7","#A5F3FC","#FBB6CE"]

def make_route_map(label, facs_df, df_routes, output_file):
    lats = [local_to_latlon(r.X, r.Y)[0] for _, r in facs_df.iterrows()]
    lons = [local_to_latlon(r.X, r.Y)[1] for _, r in facs_df.iterrows()]
    center = (sum(lats)/len(lats), sum(lons)/len(lons))

    m = folium.Map(location=center, zoom_start=13,
                   tiles="CartoDB positron",
                   control_scale=True)

    # Build lat/lon lookup
    fac_latlon = {}
    for _, row in facs_df.iterrows():
        fac_latlon[row["facility"]] = local_to_latlon(row.X, row.Y)

    # Draw routes
    route_group = folium.FeatureGroup(name="Routes", show=True)
    for _, r in df_routes.iterrows():
        route_idx = int(r["route"]) - 1
        color     = ROUTE_COLORS[route_idx % len(ROUTE_COLORS)]
        stops_raw = r["stops"]

        # Extract unique facility names from stops string
        fac_names = []
        seen = set()
        for part in stops_raw.split(" → "):
            base = part.split(" [")[0].strip()
            if base in fac_latlon and base not in seen:
                fac_names.append(base)
                seen.add(base)

        if not fac_names:
            continue

        coords_latlon = [DEPOT_LATLON] + [fac_latlon[f] for f in fac_names] + [DEPOT_LATLON]
        is_multi = len(fac_names) > 1

        # Fetch road geometry from OSRM
        print(f"    Fetching road geometry for route {int(r['route'])}...")
        road_coords = get_road_geometry(coords_latlon)
        time.sleep(0.3)  # be polite to the free API

        folium.PolyLine(
            road_coords,
            color=color,
            weight=3 if is_multi else 2,
            opacity=0.9 if is_multi else 0.45,
            dash_array=None if is_multi else "6 4",
            tooltip=f"Route {int(r['route'])}: {r['stops']} | "
                    f"{r['distance_km']:.1f} km road distance",
        ).add_to(route_group)
    route_group.add_to(m)

    # Facility markers
    fac_group = folium.FeatureGroup(name="Facilities", show=True)
    for _, row in facs_df.iterrows():
        lat, lon = fac_latlon[row["facility"]]
        is_apl   = row.get("type","SP") == "APL"
        color    = "purple" if is_apl else "orange"
        icon     = "cube"   if is_apl else "star"
        demand   = row["daily_demand"]
        folium.Marker(
            location=(lat, lon),
            icon=folium.Icon(color=color, icon=icon, prefix="fa"),
            tooltip=f"{row['facility']} — {demand:.0f} parcels/day",
            popup=f"<b>{row['facility']}</b><br>"
                  f"{'APL' if is_apl else 'Service Point'}<br>"
                  f"{demand:.0f} parcels/day",
        ).add_to(fac_group)
    fac_group.add_to(m)

    # Depot marker
    folium.Marker(
        location=DEPOT_LATLON,
        icon=folium.Icon(color="green", icon="home", prefix="fa"),
        tooltip="PostNL Depot — Born (~26 km from Maastricht)",
        popup="<b>PostNL Depot</b><br>Holtum-Noordweg 107, Born",
    ).add_to(m)

    # Stats summary in legend
    n_vans     = len(df_routes)
    total_km   = df_routes["distance_km"].sum()
    total_cost = df_routes["route_cost_eur"].sum()
    multi_routes = df_routes[df_routes["n_stops"] > 1]

    multi_html = "".join(
        f'<span style="color:{ROUTE_COLORS[int(r.route)-1]}">──</span> '
        f'Route {int(r.route)}: {r.stops}<br>'
        for _, r in multi_routes.iterrows()
    )

    legend = f"""
    <div style="position:fixed;bottom:30px;left:30px;z-index:1000;
                background:white;padding:14px 18px;border-radius:10px;
                border:1px solid #ddd;font-family:sans-serif;font-size:12px;
                box-shadow:0 2px 10px rgba(0,0,0,0.15);max-width:260px">
        <b style="font-size:14px">{label}</b><br>
        <hr style="margin:8px 0;border-color:#eee">
        <b>Van trips/day:</b> {n_vans}<br>
        <b>Total road distance:</b> {total_km:.1f} km/day<br>
        <b>First-mile cost:</b> €{total_cost:,.2f}/day<br>
        <b>Annual cost:</b> €{total_cost*OPER_DAYS:,.0f}/year<br>
        <hr style="margin:8px 0;border-color:#eee">
        <span style="color:#22c55e">⬛</span> Depot (Born, ~26 km away)<br>
        <span style="color:#f97316">⬛</span> Service Point<br>
        {"<span style='color:#a855f7'>⬛</span> APL (self-serve, no last-mile van)<br>" if any(facs_df.get("type","SP") == "APL") else ""}
        <hr style="margin:8px 0;border-color:#eee">
        <b>Multi-stop routes:</b><br>
        {multi_html if multi_html else "None"}
        <hr style="margin:8px 0;border-color:#eee">
        <span style="font-size:10px;color:#888">Distances via Dijkstra shortest path<br>
        on Maastricht road network</span>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend))
    folium.LayerControl().add_to(m)

    m.save(output_file)
    print(f"  Saved → {output_file}")


print(f"\n{'='*60}")
print("  Generating Folium maps with road-distance routes...")
print("="*60)
make_route_map(f"S3 — {len(s3_facs)} SPs", s3_facs, s3_df, "map_s3_routes.html")
make_route_map(f"S4 — {len(s4_sp)} SPs + {len(s4_ap)} APLs", s4_facs, s4_df, "map_s4_routes.html")

print(f"\n{'='*60}")
print("  All done!")
print(f"{'='*60}")
print(f"  routes_S3_dijkstra.csv")
print(f"  routes_S4_dijkstra.csv")
print(f"  map_s3_routes.html")
print(f"  map_s4_routes.html")