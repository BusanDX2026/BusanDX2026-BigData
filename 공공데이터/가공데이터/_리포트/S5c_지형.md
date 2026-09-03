# S5c. DEM 파생지표 → 100m 격자
생성: 2026-09-03 18:17

100m 격자: 81,435
DEM 30m: shape (1884, 1631), 픽셀 30.0m, 유효 28.4%
목표 100m 래스터: 565 x 490, 원점 (360700, 312500)

- DEM 값 없는 격자(해상·경계밖): 211 (0.3%) → NaN 유지 (S0 §4)
- elev_mean: {'count': 81224.0, 'mean': 107.7, 'std': 127.6, 'min': -6.1, '10%': 1.0, '50%': 60.1, '90%': 295.3, 'max': 784.9}
- slope_mean(deg): {'count': 79822.0, 'mean': 11.8, 'std': 9.3, 'min': 0.0, '50%': 11.2, '90%': 24.8, 'max': 55.7}
- 저지대(≤5m) 비율>0.5 격자: 18,311  / TPI<-1(주변보다 1m+ 낮음): 37,579
- 산출: C:\project_git\BusanDX2026-BigData\공공데이터\가공데이터\02_레이어별\지형_grid.parquet  컬럼 ['grid_id', 'elev_mean', 'elev_min', 'slope_mean', 'lowland5_ratio', 'lowland3_ratio', 'tpi']