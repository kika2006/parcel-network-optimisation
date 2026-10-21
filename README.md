# Optimising a Parcel Network's Last Mile in Maastricht

**Author:** Francisca (Kika) Freixo  

A location-optimisation model that decides which parcel **service points (SPs)** to keep, move or close, and where to add **automated parcel lockers (APLs)**, to minimise total network cost while keeping pickup convenient.

**Python · PuLP (MILP) · simulated annealing · SciPy (Dijkstra) · statsmodels · Folium**

---

## The problem
A parcel carrier ("Post&L") runs 35 service points in Maastricht, which handle about **1.35M parcel events a year**. **68% of the €2.58M annual network cost is fixed SP cost**, and volume is very uneven: one SP handles 157k parcels a year, another fewer than 5k.

## Approach (team of 6)
1. **Data preparation**: 284 CBS 500 m grid squares with demographics. Missing city-centre data was imputed from the 8 nearest neighbours.
2. **Demand model**: parcel demand allocated by households and calibrated to each SP's observed volume.
3. **Pickup-behaviour regression**: pickup rate falls with distance and income (`pickup = β₀ − 0.164·dist − 0.0022·income`).
4. **Road-network distances** with Dijkstra shortest paths on the city graph.
5. **Optimisation**:
   - a **p-median MILP** (PuLP) for SP locations, swept over k = 5–40
   - **simulated annealing** with reheating for the joint SP + APL configuration (6 move types)
   - capacity, 3 km accessibility and failed-delivery costs built into the cost function
6. **Sensitivity analysis** (tornado charts) on the key cost and behaviour assumptions.

## Results
| Scenario | Network | Annual cost | vs baseline | Pickup rate |
|---|---|---|---|---|
| S1 Baseline | 35 SPs | €2.58M | – | 64.3% |
| S2 Relocate | 35 SPs, better spots | €2.27M | −12% | 66.2% |
| S3 Resize | 12 SPs | €1.57M | −39% | 63.1% |
| **S4 Recommended** | **11 SPs + 4 APLs** | **€1.52M** | **−41%** | **63.8%** |

The recommended network saves **over €1M a year**. It also halves the number of grid squares without a pickup point within 3 km compared with resizing alone. The result is robust: S4's cost moves by less than €400 across ±20% APL-uplift assumptions.

## Repository structure
```
src/network_optimisation.py   # end-to-end model: demand, pickup regression, MILP, SA, sensitivity
src/eda.py                    # exploratory analysis (PDF report)
src/route_maps.py             # van routing (Clarke-Wright) and interactive route maps
notebooks/eda_maastricht.ipynb
maps/map_s1_baseline.html     # interactive map of the current network
```

## Running it
```bash
pip install -r requirements.txt
# put data_Maastricht_2025.xlsx in data/
python src/network_optimisation.py
```

*Maastricht University, eLab Business Case II (2026). The dataset was provided by the course and is not included.*
