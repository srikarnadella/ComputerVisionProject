## Error Analysis

Out of **6300** test images the model made **74** errors (1.17% error rate).

- **False Positives** (nowildfire → wildfire): 20
- **False Negatives** (wildfire missed): 54

### Failure Mode Breakdown

| Failure Mode | Count | % of Errors | Description |
|---|---:|---:|---|
| SMOKE_HAZE | 1 | 1.4% | Smoke or haze — warm tones, low saturation |
| BRIGHT_TERRAIN | 0 | 0.0% | Bright terrain — high brightness, low saturation |
| LOW_CONTRAST | 7 | 9.5% | Low contrast — night or heavy overcast |
| OTHER | 66 | 89.2% | Other / ambiguous |

### Confidence Distribution

Mean wildfire confidence on **correct** predictions: **0.545**  
Mean wildfire confidence on **errors**: **0.373**

See `score_distribution.png` for the full histogram.

> Failure modes are assigned by a heuristic colour/contrast scorer. Use the image grids for qualitative inspection.
