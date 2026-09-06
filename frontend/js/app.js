/** State, wiring, and the WebSocket. The only module that mutates anything. */

import * as api from "./api.js";
import * as assist from "./assist.js";
import { render, STAGES } from "./render.js";

const DEFAULT_CONFIG = {
  return_mode: "diff",
  fee_bps: 3,
  slippage_bps: 2,
  // Opt-in: each permutation is a full re-run of the book.
  permutation: { n: 0, method: "rotate", metric: "sharpe", seed: 0 },
  // Chronological train/test; omit or enabled:false for a full-sample book.
  split: {
    enabled: true,
    test_frac: 0.3,
    train: { mode: "once", chunk_size: 60 },
    test: { mode: "once", chunk_size: 60 },
  },
};

let state = {
  strategies: [], strategyId: null,
  runs: [], runId: null, runStatus: "idle",
  steps: {}, logs: [], metrics: null, curves: null,
  selected: null, context: null, debugPort: null,
  error: null,
  deepseekKey: assist.getKey(),
  assistMessages: [],
  assistBusy: false,
  assistError: null,
  assistExplainedFor: null,
};

let socket = null;

const el = (id) => document.getElementById(id);

/** Always a new state object -- never a field assignment on the old one. */
function set(patch) {
  state = { ...state, ...patch };
  render(state);
}

const withStep = (stage, patch) => ({
  ...state.steps,
  [stage]: { ...(state.steps[stage] || {}), ...patch },
});

/** Drop every stage at or after `stage`, mirroring store.reset_steps. */
const stepsBefore = (stage) =>
  Object.fromEntries(
    Object.entries(state.steps)
      .filter(([name]) => STAGES.indexOf(name) < STAGES.indexOf(stage)),
  );

async function guard(work) {
  try {
    // Do not set() here: render() rebuilds context-body and would wipe the
    // assist input before Send can read it.
    if (state.error) set({ error: null });
    await work();
  } catch (err) {
    set({ error: err.message });
  }
}

// --- events -----------------------------------------------------------------

/** Keep an open context panel truthful: a stage that just re-ran must not go on
 *  showing the traceback from the attempt before it. */
const reopen = (stage) =>
  stage && state.selected === stage ? guard(() => openStage(stage)) : undefined;

/** Equity paths are their own artifacts, fetched only when the metrics say
 *  the corresponding test actually ran. */
async function loadCurves(runId, metrics) {
  const patch = { curves: null };
  if (metrics?.permutation) {
    const body = await api.getPermutation(runId).catch(() => null);
    patch.curves = body?.curves || null;
  }
  if (state.runId === runId) set(patch);
}

function applyEvent({ stage, event, payload }) {
  switch (event) {
    case "log":
      // Keep the stage with the line: the pane groups by it, and a line with
      // no owner cannot be shown next to the step that produced it.
      return set({ logs: [...state.logs, { stage, line: payload.line }] });
    case "start":
      return set({ steps: withStep(stage, { status: "running", error: null }) });
    case "done":
      set({ steps: withStep(stage, {
        status: "completed", summary: payload.summary, ms: payload.ms }) });
      return reopen(stage);
    case "error":
      set({ steps: withStep(stage, {
        status: "failed",
        error: payload.error,
        traceback: payload.traceback || null,
      }) });
      return reopen(stage);
    case "debug_ready":
      return set({ debugPort: payload.port,
                   steps: withStep(stage, { debug_port: payload.port }) });
    case "debug_unavailable":
      return set({ error: `debugpy never opened a port for '${stage}'` });
    case "run_started":
      return set({ runStatus: "running" });
    case "run_resumed":
      set({ runStatus: "running", metrics: null, curves: null, debugPort: null,
            steps: stepsBefore(payload.from_stage) });
      return reopen(state.selected);
    case "run_completed":
      set({ runStatus: "completed", metrics: payload.metrics });
      return guard(() => loadCurves(state.runId, payload.metrics));
    case "run_paused":
      if (payload.failed_stage && payload.error) {
        set({
          runStatus: "paused",
          steps: withStep(payload.failed_stage, {
            status: "failed",
            error: payload.error,
            traceback: payload.traceback || null,
          }),
        });
      } else {
        set({ runStatus: "paused" });
      }
      return openStage(payload.failed_stage);
    default:
      return undefined;
  }
}

function connect(runId) {
  socket?.close();
  set({ runId, steps: {}, logs: [], metrics: null, curves: null, context: null,
        selected: null, debugPort: null, error: null, runStatus: "running",
        assistMessages: [], assistBusy: false, assistError: null,
        assistExplainedFor: null });

  socket = api.openEvents(runId);
  socket.onmessage = (message) => applyEvent(JSON.parse(message.data));
  // A dropped socket is not worth a reconnect loop: one resync tells us where
  // the run actually got to.
  socket.onclose = () => { if (state.runId === runId) guard(() => resync(runId)); };
}

async function resync(runId) {
  const run = await api.getRun(runId);
  const steps = Object.fromEntries(run.steps.map((step) => [step.name, step]));
  set({ runStatus: run.status, steps });
  if (run.status === "completed") {
    const metrics = await api.getMetrics(runId).catch(() => null);
    set({ metrics });
    await loadCurves(runId, metrics);
  }
  if (run.status === "paused") {
    const failed = run.steps.find((step) => step.status === "failed");
    if (failed) await openStage(failed.name);
  }
}

// --- assist -----------------------------------------------------------------

const assistKey = () => `${state.runId}:${state.selected}`;

async function explainOnce() {
  const key = state.deepseekKey;
  const tag = assistKey();
  if (!key || !state.runId || !state.selected) return;
  if (state.assistExplainedFor === tag || state.assistBusy) return;
  const step = state.steps[state.selected] || {};
  const err = state.context?.error || step.error;
  if (!err) return;

  const prior = [
    { role: "user", content: "Explain this failure in plain English." },
  ];
  set({
    assistMessages: prior,
    assistBusy: true,
    assistError: null,
    assistExplainedFor: tag,
  });
  try {
    const { reply } = await api.assistChat({
      run_id: state.runId,
      stage: state.selected,
      messages: prior,
    }, key);
    if (assistKey() === tag) {
      set({
        assistMessages: [...prior, { role: "assistant", content: reply }],
        assistBusy: false,
      });
    }
  } catch (err) {
    if (assistKey() === tag) {
      set({ assistBusy: false, assistError: err.message, assistExplainedFor: null });
    }
  }
}

async function sendAssist(text) {
  if (text == null) text = (el("assist-input")?.value || "").trim();
  text = (text || "").trim();
  if (!text) return;
  if (!state.deepseekKey) {
    set({ assistError: "Save a DeepSeek API key first." });
    return;
  }
  const input = el("assist-input");
  if (input) input.value = "";

  // Typing "fix" (etc.) routes to the patch path, same as the Fix button.
  if (/\bfix\b/i.test(text) && assist.FIXABLE_STAGES.includes(state.selected)) {
    return fixStrategy(text);
  }

  const prior = [...state.assistMessages, { role: "user", content: text }];
  set({ assistMessages: prior, assistBusy: true, assistError: null });
  try {
    const { reply } = await api.assistChat({
      run_id: state.runId,
      stage: state.selected,
      messages: prior,
    }, state.deepseekKey);
    set({
      assistMessages: [...prior, { role: "assistant", content: reply }],
      assistBusy: false,
    });
  } catch (err) {
    set({ assistBusy: false, assistError: err.message });
  }
}

async function fixStrategy(userText = "Fixing strategy") {
  if (!state.deepseekKey || !assist.FIXABLE_STAGES.includes(state.selected)) return;
  if (userText === "Fixing strategy") {
    if (!confirm("Create a minimal fixed strategy file and select it?")) return;
  }
  const prior = [...state.assistMessages, { role: "user", content: userText }];
  set({ assistMessages: prior, assistBusy: true, assistError: null });
  try {
    const strategy = await api.assistFix({
      run_id: state.runId,
      stage: state.selected,
      messages: prior,
    }, state.deepseekKey);
    await refreshStrategies(strategy.id);
    set({
      assistBusy: false,
      assistMessages: [
        ...prior,
        {
          role: "assistant",
          content: `Saved **${strategy.filename}** (\`${strategy.id}\`) and selected it. Run or resume when ready.`,
        },
      ],
    });
  } catch (err) {
    set({ assistBusy: false, assistError: err.message });
  }
}

// --- actions ----------------------------------------------------------------

async function openStage(stage) {
  const same = state.selected === stage;
  set({
    selected: stage, context: null,
    ...(same ? {} : {
      assistMessages: [], assistBusy: false, assistError: null,
      assistExplainedFor: null,
    }),
  });
  const context = await api.getContext(state.runId, stage);
  if (state.selected === stage) {
    set({ context });
    await explainOnce();
  }
}

const refreshStrategies = async (selectId) => {
  const { strategies } = await api.listStrategies();
  set({ strategies, strategyId: selectId || state.strategyId || strategies[0]?.id });
};

const refreshRuns = async () => set({ runs: (await api.listRuns()).runs });

function readConfig() {
  const raw = el("config").value.trim() || "{}";
  try {
    const config = JSON.parse(raw);
    if (config === null || typeof config !== "object" || Array.isArray(config)) {
      throw new Error("config must be a JSON object");
    }
    return config;
  } catch (err) {
    throw new api.ApiError(`config is not valid JSON: ${err.message}`);
  }
}

// --- wiring -----------------------------------------------------------------

el("upload").onclick = () => guard(async () => {
  const [file] = el("file").files;
  if (!file) throw new api.ApiError("choose a .py file first");
  const strategy = await api.uploadStrategy(file);
  await refreshStrategies(strategy.id);
});

el("strategy").onchange = (e) => set({ strategyId: e.target.value || null });

el("past-run").onchange = (e) => guard(async () => {
  if (e.target.value) connect(e.target.value);
});

el("run").onclick = () => guard(async () => {
  const config = readConfig();
  const { run_id } = await api.createRun(state.strategyId, config);
  connect(run_id);
  await refreshRuns();
});

el("resume").onclick = () => guard(async () => {
  if (!state.selected) throw new api.ApiError("select a stage to resume from");
  set({ debugPort: null });
  await api.resumeRun(state.runId, state.selected, el("debug").checked);
});

el("stages").onclick = (e) => {
  const pill = e.target.closest("[data-stage]");
  if (pill && state.runId) guard(() => openStage(pill.dataset.stage));
};

el("save-deepseek").onclick = () => {
  const value = (el("deepseek-key").value || "").trim();
  assist.setKey(value);
  set({ deepseekKey: value });
};

el("context-body").addEventListener("click", (e) => {
  if (e.target.closest("#assist-send")) {
    const text = (el("assist-input")?.value || "").trim();
    guard(() => sendAssist(text));
  }
  if (e.target.closest("#assist-fix")) guard(() => fixStrategy());
});

el("context-body").addEventListener("keydown", (e) => {
  if (e.target.id === "assist-input" && e.key === "Enter") {
    e.preventDefault();
    const text = (e.target.value || "").trim();
    guard(() => sendAssist(text));
  }
});

el("deepseek-key").value = state.deepseekKey;
el("config").value = JSON.stringify(DEFAULT_CONFIG, null, 2);
render(state);
guard(async () => { await refreshStrategies(); await refreshRuns(); });
