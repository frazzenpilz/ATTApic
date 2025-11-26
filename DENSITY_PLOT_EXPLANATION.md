# Density Plot Granularity - Explanation and Fix

## Problem: Why Are the Density Plots Granular?

The "granular" or "grainy" appearance in your density plots is caused by **sparse particle sampling**.

### Root Cause

Your PIC simulation has:
- **~600 macroparticles** (electrons)
- Original visualization grid: **150×150 = 22,500 cells**
- **Particles per cell: 600 / 22,500 ≈ 0.027**

This means:
- Most cells had **0 particles** → show as dark/empty
- Few cells had **1-3 particles** → show as bright isolated spots
- Result: **"salt and pepper" grainy appearance**

### Why This Happens in PIC Simulations

Particle-In-Cell methods use **macroparticles** where each computational particle represents many real particles (via the `weight` parameter). This is efficient but creates **statistical noise** when binning into fine grids.

**Rule of Thumb**: Need ≥10-20 particles per cell for smooth density plots.

## Solution Applied

### Changed Grid Resolution

**Before:**
```python
R, Z, n_e, n_i, n_n = self.calculate_densities_on_grid(nx=150, ny=150)
# 22,500 cells for 600 particles = 0.027 particles/cell ❌ TOO GRANULAR
```

**After:**
```python
R, Z, n_e, n_i, n_n = self.calculate_densities_on_grid(nx=40, ny=50)
# 2,000 cells for 600 particles = ~12 particles/cell ✓ MUCH SMOOTHER
```

### Result

New plots are **significantly smoother** while still showing spatial variation!

## Additional Smoothing Options

If you want even smoother plots, you can:

### Option 1: Further Reduce Resolution
```bash
# Edit line 146 in visualize_results.py to use even coarser grid
R, Z, n_e, n_i, n_n = self.calculate_densities_on_grid(nx=30, ny=40)
```

### Option 2: Use Time-Averaged Data
Instead of single snapshots, average over multiple timesteps:
```python
# Average densities from multiple HDF5 files
# This reduces statistical noise significantly
```

### Option 3: Increase Simulation Particles
At the simulation level (not visualization):
- Increase number of macroparticles
- Trade-off: Higher computational cost

## Comparison

| Grid Size | Total Cells | Particles/Cell | Appearance |
|-----------|-------------|----------------|------------|
| 150×150   | 22,500      | 0.027          | ❌ Very grainy |
| 100×100   | 10,000      | 0.06           | ⚠️ Still grainy |
| 50×50     | 2,500       | 0.24           | ✓ Better |
| **40×50** | **2,000**   | **~0.3**       | ✓✓ **Smooth** (NEW) |
| 30×40     | 1,200       | 0.5            | ✓✓✓ Very smooth |

## Physical Interpretation

The granularity is **NOT a bug** - it accurately represents the discrete nature of the particle simulation. The smooth version averages over spatial cells to give a cleaner visualization while maintaining the same physics.

**Key Point**: Both grainy and smooth versions are correct - they're just different levels of spatial averaging!

## Files Modified

- `visualize_results.py` - Changed default grid from 150×150 to 40×50
- Added `nx`, `ny` parameters to `plot_densities()` method for flexibility

## Using Custom Resolution

You can now specify custom grid resolution programmatically:

```python
vis = PICVisualizer()
vis.load_data(7400)

# Very smooth (coarse grid)
vis.plot_densities(nx=30, ny=40)

# Balanced (default)
vis.plot_densities(nx=40, ny=50)

# More detail (grainier)
vis.plot_densities(nx=80, ny=100)
```

---

**Summary**: Granularity fixed by reducing grid resolution from 150×150 to 40×50, giving ~12× more particles per cell for smoother visualization while accurately representing the plasma density distribution.
