# Research Pipeline

A minimal research pipeline for a power-trading desk. Upload a strategy, watch it
run stage by stage, and get inside any stage that breaks.

It sits on top of the `backend/data` lake (ENTSO-E and Ember, bronze → silver →
gold) and adds the layer above it: orchestration, checkpointing, live progress,
and debugging.

## The five stages

```
collect → prepare → fit │ simulate → evaluate
└──── your Strategy ────┘└── pipeline-owned ──┘
```

You write collect, prepare, and optional fit. `on_tick` is called from
`simulate`, which the engine owns: it fills, tracks inventory, and marks PnL.
`evaluate` is pipeline code too, so two runs are actually comparable.

Each stage runs in **its own subprocess** and writes a checkpoint. That is what
makes a stage independently re-runnable: fix stage three, resume from stage
three, and stages one and two are not recomputed.

## Quickstart

```bash
uv run uvicorn backend.pipeline.api:app --reload
```

Then open <http://localhost:8000>. The UI is the whole loop: upload a strategy,
edit its config, watch the five stages advance live, click any stage to see what
it was handed and what it produced, and resume from a broken stage — with or
without a debugger waiting.

### Or by hand

```bash
uv run uvicorn backend.pipeline.api:app --reload

# upload a strategy
curl -s -F file=@backend/strategies/spread_meanrev_toFix.py localhost:8000/strategies

# start a run
curl -s -X POST localhost:8000/runs -H 'content-type: application/json' -d '{
  "strategy_id": "<id from above>",
  "config": {"pairs": [["FRA","DEU"]], "return_mode": "diff", "cost_bps": 5}
}'

# watch it, then read the result
curl -s localhost:8000/runs/<run_id>
curl -s localhost:8000/runs/<run_id>/artifacts/metrics.json
```

## Writing a strategy

Subclass `Strategy` and implement `collect`, `prepare`, and `on_tick`. `fit` is
optional. Return polars or pandas — either is accepted at the boundary.

```python
from backend.pipeline.contract import Strategy

class MyStrategy(Strategy):
    def collect(self, config):                 # raw data, any shape
        ...
    def prepare(self, raw, config):            # ts, asset, price + features
        ...
    def fit(self, prepared, config):           # optional; return the model
        return None
    def on_tick(self, ts, market, inventory, model, config):
        ...                                    # -> asset, qty, optional delta
```

`prepare` must include `ts, asset, price`. Optional `volume` caps the fill
(missing means full fill). Optional `fee_bps` / `slippage_bps` /
`fee_per_unit` / `slippage_per_unit` price execution row by row. Extra columns
are features: the engine passes them to `on_tick` as the market snapshot.

`on_tick` sees the **filled** inventory (a copy) and returns signed trades.
Unfilled remainder is dropped. `delta` is the unit greek of that asset
(default 1); portfolio delta is `sum(inventory * delta)`.

Two things to know:

- **`fit` returns the model** rather than setting `self.model`. Stages are
  separate processes, so instance state would not survive between them.
  Tick-local state on `self` is allowed inside `simulate`.
- **Previous inventory earns this tick's price change; this tick's fill starts
  earning next tick.** That is the defence against lookahead. `prepare` still
  lags features (the DA noon rule).

`backend/strategies/spread_meanrev_toFix.py` is a working example.

## Config

Passed as `config` when creating a run; reaches both your strategy and the engine.

| Key | Default | Meaning |
|---|---|---|
| `return_mode` | `pct` | `pct` compounds fractional returns. `diff` marks the absolute price change in currency per unit — **use this for power**, where prices go negative and a percentage return is undefined. |
| `fee_bps` | `cost_bps` | Commission per unit of turnover (`|fill|`). |
| `slippage_bps` | `0.0` | Execution shortfall per unit of turnover, charged the same way. Together with `fee_bps` this is the whole cost model — linear in turnover, no market impact. |
| `cost_bps` | `1.0` | The single cost knob this started with. Still works, and is what `fee_bps` falls back to, so an older config charges exactly what it always did. |
| `permutation` | off | `{n, method, metric, seed}` — see below. |
| `split` | off unless set | `{enabled, test_frac, test_start, purge, train, test}` — chronological train/test, see below. |
| `periods_per_year` | `365` | Annualization factor (365 for daily power, not 252). |
| `stage_timeout_s` | `900` | Wall-clock limit per stage. |

A value in the panel beats the config: `fee_bps` in the data wins over `fee_bps`
in the config. Negative costs are refused — a rebate is possible, a sign error is
likelier, and it inflates the Sharpe silently.

Anything else you put in `config` is yours; the strategy reads it directly.

## Is it better than luck?

A Sharpe ratio alone says nothing. Turn on the permutation test and the
`evaluate` stage builds the null empirically: rotate or shuffle **feature**
columns in time, re-run the event loop, repeat `n` times, and report where the
real result falls.

```json
{"permutation": {"n": 200, "method": "rotate", "metric": "sharpe", "seed": 0}}
```

Only features move. Prices, timestamps, volume and the cost model are
untouched, so every permutation is a book that could actually have been traded
on that path with scrambled information.
The result lands in `metrics.json`:

```json
"permutation": {
  "metric": "sharpe", "method": "rotate", "n": 200, "seed": 1,
  "observed": 4.4004, "p_value": 0.004975, "beat_by_chance": 0,
  "null_mean": 0.024185, "null_p95": 0.5002, "null_max": 0.7625
}
```

`p_value` is `(1 + beaten) / (n + 1)`, never 0: a finite sample of permutations
cannot prove impossibility, and reporting 0 would claim it did.

**The chart.** The UI draws every permuted equity path in grey with the real one
on top — if the strategy is luck, its line disappears into the crowd. The paths
live in their own artifact so `metrics.json` stays the eight numbers you
actually read:

```bash
curl -s localhost:8000/runs/<run_id>/artifacts/permutation.json
```

It is capped at 50 plotted paths and thinned to 240 points: past that a chart
has more lines than it has pixels.

**Why `rotate` is the default.** It shifts features circularly, which preserves
their autocorrelation, so the permuted book trades about as often as the real one
and pays about the same costs. `shuffle` draws an i.i.d. permutation and destroys
that structure. Use it for a feature with no time structure to begin with;
otherwise prefer `rotate`.

Off by default — it costs `n` full event loops. That is cheap on daily panels
and will hurt on true tick tapes. Progress streams to the log as it runs.

A low p-value is not a generalization claim. The permutation test asks whether
the *current* book is distinguishable from a scrambled information set. How that
book was produced — train only, then score held-out test — is the train/test split.

## Train / test split

Put a chronological cut **first**. Fit (and any ML hyperparameter search) sees
**train only**. The reported simulate/evaluate is **test only**. Omit `split` (or set
`enabled: false`) to keep the old full-sample path.

```json
{
  "split": {
    "enabled": true,
    "test_frac": 0.3,
    "train": { "mode": "once", "chunk_size": 60 },
    "test": { "mode": "once", "chunk_size": 60 }
  }
}
```

`test_start` (when set) overrides `test_frac`. `purge` defaults to 1,
dropping the last training bars so a forward label cannot straddle the cut.

**Train sessions.** `train.mode: "once"` fits once on the whole train window.
`rolling` walks expanding prefixes of train (chunk length `train.chunk_size`)
and keeps every session’s diagnostics; the last session’s weights feed test
unless you structure the model pack yourself. HPO belongs inside `fit`.

**Test sessions.** `test.mode: "once"` runs the event loop on the full test set
with frozen weights and an empty book. `rolling` partitions test into fixed
chunks: session `i` may re-`fit` on `train ∪ chunks[0..i)`, then simulates only
chunk `i` from a flat book. Tapes are stitched into one OOS book; every
session’s metrics land in `metrics.split` and `sessions.json`.

```bash
curl -s localhost:8000/runs/<run_id>/artifacts/split.json
curl -s localhost:8000/runs/<run_id>/artifacts/sessions.json
```

## Watching it run

Every stage streams. Whatever your strategy `print`s arrives **while the stage
is still running**, grouped under the stage that wrote it — in the log pane and
again in that stage's own panel when you click it. Stages that printed nothing
show nothing; the pipeline adds no chatter of its own.

That works because the stage subprocess is launched with `-u`. Its stdout is a
pipe, so Python would otherwise block-buffer it and hold every line until the
stage exited — a ten-minute `collect` would look hung, then dump its whole log
at the end.

## Deleting a run

There is no delete endpoint. **The run directory is the record**, so removing it
is how you remove a run:

```bash
rm -rf runs/<run_id>          # a run
rm runs/strategies/<id>.py    # a strategy
```

SQLite only indexes those files. The next listing reconciles the index with the
filesystem and drops whatever no longer exists, along with its steps and events,
so a purge outside the app is reflected inside it without a restart. Deleting a
strategy leaves its finished runs alone — they kept their own checkpoints, and
those are still results worth reading.

## When a stage breaks

The run goes **`paused`**, not `failed` — every completed stage kept its
checkpoint.

**Read the context.** One endpoint returns everything needed to diagnose the
stage: the source that ran, the config, the traceback, the schema/shape/head of
what it was handed, and the logs.

```bash
curl -s localhost:8000/runs/<run_id>/steps/prepare/context
```

This is also the **agent hook**. It is deliberately JSON and nothing more —
no model is called anywhere in this codebase. Wiring an LLM up later means
POSTing this payload somewhere; no restructuring is needed.

**Or attach a real debugger.** Resume the broken stage with `debug: true`:

```bash
curl -s -X POST localhost:8000/runs/<run_id>/resume -H 'content-type: application/json' \
  -d '{"from_stage": "prepare", "debug": true}'
```

The stage waits for a client before running a line. Poll `GET /runs/<run_id>` for
its `debug_port`, then attach VS Code (*Python: Remote Attach*, `127.0.0.1:<port>`).
The port is only reported once it is genuinely accepting connections.

Debugging is opt-in per resume — a normal run never waits for a debugger.

## API

| | |
|---|---|
| `POST /strategies` | upload a `.py` file |
| `GET /strategies` | list uploads, newest first |
| `GET /strategies/{id}` | metadata + source |
| `POST /runs` | `{strategy_id, config}` → starts a run |
| `GET /runs` · `GET /runs/{id}` | status and per-stage state |
| `POST /runs/{id}/resume` | `{from_stage, debug}` |
| `GET /runs/{id}/steps/{stage}/context` | the debug / agent hook |
| `GET /runs/{id}/artifacts/{name}` | download a checkpoint |
| `GET`/`WS /runs/{id}/events` | history, then live tail |

Interactive docs at `/docs`. The UI at `/` is served from `frontend/` by this
same app — one process, one origin, so there is no CORS layer and no dev proxy.

## Frontend

```
frontend/
  index.html   the shell
  style.css
  js/api.js    one function per endpoint
  js/render.js pure state → DOM
  js/chart.js  the permutation chart, as an SVG string
  js/logs.js   groups log lines by stage
  js/esc.js    HTML escaping, shared
  js/app.js    state, WebSocket, handlers
```

Vanilla ES modules: no npm, no bundler, no build step. Edit a file and reload.
State is one object that is replaced rather than mutated, which is enough
structure at this size and keeps the repo's immutability rule intact.

No charting library: the chart is ~90 lines that turn numbers into an SVG
string. A dependency would have to be bundled, and there is no bundler.

Because there is no build step there is no JS test runner either. The frontend
is covered from the backend side — `tests/test_api.py::TestFrontend` asserts the
static mount serves the module tree *and* does not shadow the API, which is the
failure mode that route ordering would otherwise cause silently. `chart.js` and
`logs.js` are pure (data in, markup or data out, no DOM), so they can be checked
with plain `node`.

## Tests

```bash
uv run pytest --cov=backend/pipeline
```

The engine tests are the ones that matter: they pin fills, inventory, and
same-tick MTM. An off-by-one there produces a plausible-looking equity curve
that is simply false.

## Scope

Single user, single machine. No auth, no tenancy. Uploaded code runs in a
subprocess with a timeout, which is **isolation, not a security boundary** — it
runs with your full local privileges. Do not expose this to untrusted uploads
without putting a container around the runner.
