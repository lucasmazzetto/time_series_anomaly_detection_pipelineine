# Benchmark Stress Test

This folder contains the stress-test notebook:

- `benchmark_stress.ipynb`

The notebook benchmarks the real API endpoints for:

- `POST /fit/{series_id}`
- `POST /predict/{series_id}`

## Requirements

From inside the `benchmark/` folder:

```bash
pip install -r requirements.txt
```

## How To Use

1. Start the API and database you want to test.
2. Open the notebook `benchmark_stress.ipynb`.
3. Update `BASE_URL` in the config cell if needed (for example `http://localhost:8000`).
4. Run cells from top to bottom.
5. Review latency summaries and plots for both fit and predict sections.

## Important Caution

This benchmark uses the **actual API** and performs real requests.

- The fit benchmark writes model/data artifacts.
- The benchmark creates records and therefore **populates the database**.

Use this only in **test** or **staging** environments.

Do **NOT** run this benchmark in production.
