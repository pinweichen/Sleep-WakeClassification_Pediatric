# GGIR/

## Overview

This repository contains an R-based pipeline for processing raw ActiGraph accelerometer data using **GGIR v3.3-4**. The script automatically validates the GGIR version, installs required dependencies, and processes raw `.bin` files using three commonly used sleep detection algorithms:

- Cole-Kripke (1992)
- Sadeh (1994)
- van Hees (2015)

---

## Files

| File | Purpose |
|---|---|
| `GGIR.R` | **Entry point** — Running GGIR with 3 algorithms choice (Cole-Kripke, Sadeh, van Hees) |


## Features

- installs **GGIR v3.3-4**
- Uses GGIR's 2023 non-wear detection algorithm
- Runs the complete GGIR workflow (Parts 1–5)
- Run Cole-Kripke, Sadeh, and van Hees algorithms separately
- Parallel processing on
- Generates:
  - Sleep metrics
  - Physical activity metrics
  - Circadian rhythm metrics
  - Intensity Gradient metrics
  - Sleep visualizations
  - Quality control reports
- Produces separate outputs for:
  - Cole-Kripke 1992
  - Sadeh 1994
  - van Hees 2015

---

## Requirements

### R Version

Recommended:

```text
R >= 4.3
```

### Required Packages

The script automatically installs:

```r
GGIR (version 3.3-4)
data.table
tidyverse
remotes
```


