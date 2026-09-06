# Bronze

Raw ENTSO-E Statistical Reports files, downloaded as-is from
<https://www.entsoe.eu/data/power-stats/>. Nothing is cleaned or renamed here —
this layer is just the untouched source, kept so any silver/gold output can be
traced back to an original download.

## What's in here

Three datasets, all `.xlsx`, all aggregated per country:

| File | Dataset | Granularity | Content |
|---|---|---|---|
| `monthly_hourly_load_values-*.xlsx` | Monthly Hourly Load Values | hourly, per country | actual total load (MW) per hour |
| `physical_energy_power_flows-*.xlsx` | Physical Energy & Power Flows | hourly, per country pair | cross-border physical flows (MW/MWh), from → to, netted per hour |
| `inventory_of_generation-*.xlsx` | Inventory of Generation | yearly, per country | national generation fleet capacity by technology |

## Naming

`<dataset>-<period>.xlsx`, where the period is either a single year or the range
covered by the file, matching how ENTSO-E publishes it:

```
monthly_hourly_load_values-2015-2019.xlsx   # multi-year archive file
monthly_hourly_load_values-2020.xlsx        # single-year file
```

ENTSO-E splits older data into range archives and recent data into one file per
year, so both shapes exist side by side. They are not deduplicated — check for
overlap before concatenating.

## Notes

- Load values are the two big files (~45 MB and ~14 MB); read them in chunks
  rather than loading whole sheets.
- Timestamps in the load files are UTC.
- Column layouts differ between the archive ranges and the per-year files;
  don't assume a stable schema across the two.
