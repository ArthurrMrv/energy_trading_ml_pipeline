/** Pure state -> DOM. Nothing here fetches, and nothing here mutates state. */

export const STAGES = [
  "collect", "prepare", "fit", "simulate", "evaluate",
];

import { permutationChart } from "./chart.js";
import * as assist from "./assist.js";
import { esc, pretty } from "./esc.js";

const el = (id) => document.getElementById(id);
const show = (node, visible) => { node.hidden = !visible; };

function summarize(step) {
  if (step.error) {
    const line = String(step.error).split("\n")[0];
    return line.length > 72 ? `${line.slice(0, 69)}…` : line;
  }
  const parts = [];
  const summary = step.summary || {};
  if (summary.rows != null) parts.push(`${summary.rows} x ${summary.cols}`);
  else if (summary.type && summary.type !== "none") parts.push(summary.type);
  if (step.ms != null) parts.push(`${Math.round(step.ms)} ms`);
  if (step.debug_port) parts.push(`debug :${step.debug_port}`);
  return parts.join(" · ") || (step.status || "pending");
}

function pillMarkup(name, state) {
  const step = state.steps[name] || {};
  return `<li class="pill" data-stage="${name}"
    data-status="${esc(step.status || "pending")}"
    aria-current="${state.selected === name}">
    <span class="name">${name}</span>
    <span class="meta">${esc(summarize(step))}</span>
  </li>`;
}

function renderStages(state) {
  el("stages").innerHTML = `<ol class="pipeline-steps">${
    STAGES.flatMap((name, i) => {
      const pill = pillMarkup(name, state);
      return i === 0
        ? [pill]
        : [`<li class="pipeline-arrow" aria-hidden="true"></li>`, pill];
    }).join("")
  }</ol>`;
}

function renderOptions(node, items, selected, label) {
  node.innerHTML = items.length
    ? items.map((item) =>
        `<option value="${esc(item.id)}" ${item.id === selected ? "selected" : ""}
         >${esc(label(item))}</option>`).join("")
    : `<option value="">none yet</option>`;
}

/** Prefer reading order: trades and win rate first among the trading stats. */
const METRIC_ORDER = [
  "sharpe", "n_trades", "win_rate", "hit_rate", "total_return", "max_drawdown",
  "annual_return", "volatility", "avg_turnover", "n_obs", "exposed_periods",
];

function formatMetric(key, value) {
  if (typeof value !== "number") return value;
  if (key === "win_rate" || key === "hit_rate" || key === "time_in_market") {
    return `${(value * 100).toFixed(1)}%`;
  }
  if (key === "n_trades" || key === "n_obs" || key === "exposed_periods") {
    return String(Math.round(value));
  }
  return Number(value.toFixed(4));
}

/** A readable one-liner for a nested metric, so the headline number is visible
 *  without expanding the block. */
function headline(key, value) {
  if (key === "permutation") {
    return `permutation · p = ${value.p_value ?? "?"} on ${value.metric ?? "?"}` +
      ` (${value.n ?? "?"} × ${value.method ?? "?"})`;
  }
  if (key === "split") {
    return `split · train ${value.n_train ?? "?"} / test ${value.n_test ?? "?"}` +
      ` · ${value.n_train_sessions ?? "?"} train sess · ${value.n_test_sessions ?? "?"} test sess`;
  }
  return key;
}

const GREEK_ORDER = ["delta", "gross", "time_in_market"];

function renderMetrics(state) {
  show(el("metrics-panel"), Boolean(state.metrics));
  if (!state.metrics) return;

  const entries = Object.entries(state.metrics);
  const nested = ([key, value]) =>
    key !== "greeks" && value !== null && typeof value === "object";
  const rank = (key) => {
    const i = METRIC_ORDER.indexOf(key);
    return i === -1 ? METRIC_ORDER.length : i;
  };

  el("metrics").innerHTML = entries
    .filter((entry) => !nested(entry) && entry[0] !== "greeks")
    .sort(([a], [b]) => rank(a) - rank(b))
    .map(([key, value]) =>
      `<div><dt>${esc(key)}</dt><dd>${esc(formatMetric(key, value))}</dd></div>`)
    .join("");

  const greeks = state.metrics.greeks;
  const greeksNode = el("metrics-greeks");
  if (greeks && typeof greeks === "object") {
    const greekEntries = Object.entries(greeks).sort(([a], [b]) => {
      const ia = GREEK_ORDER.indexOf(a);
      const ib = GREEK_ORDER.indexOf(b);
      return (ia === -1 ? 99 : ia) - (ib === -1 ? 99 : ib);
    });
    greeksNode.innerHTML = greekEntries
      .map(([key, value]) =>
        `<div><dt>${esc(key)}</dt><dd>${esc(formatMetric(key, value))}</dd></div>`)
      .join("");
    show(greeksNode, greekEntries.length > 0);
  } else {
    greeksNode.innerHTML = "";
    show(greeksNode, false);
  }

  // Nested values (permutation, split) get collapsible blocks.
  el("metrics-extra").innerHTML = entries
    .filter(nested)
    .map(([key, value]) =>
      block(headline(key, value), pretty(value)))
    .join("");
}

function renderChart(state) {
  const permNode = el("permutation-chart");
  const permMarkup = state.curves
    ? permutationChart(state.curves, state.metrics?.permutation || {})
    : "";
  show(permNode, Boolean(permMarkup));
  permNode.innerHTML = permMarkup;
}

function block(title, body, { open = false } = {}) {
  return `<details${open ? " open" : ""}><summary>${esc(title)}</summary>` +
    `<pre>${body}</pre></details>`;
}

function describe(name, artifact) {
  if (!artifact) return block(name, esc("(no checkpoint)"));
  if (artifact.type === "error" && artifact.error) {
    return block(`${name} · error`, esc(artifact.error), { open: true });
  }
  const shape = artifact.shape ? ` · ${artifact.shape.join(" x ")}` : "";
  const detail = artifact.type === "dataframe"
    ? pretty({ schema: artifact.schema, nulls: artifact.null_counts, head: artifact.head })
    : pretty(artifact);
  return block(`${name} · ${artifact.type || "?"}${shape}`, detail);
}

function assistBlock(state, failed) {
  if (!failed) return "";
  if (!state.deepseekKey) {
    return `<p class="muted">Add a DeepSeek API key above to explain this error.</p>`;
  }
  return assist.markup({
    messages: state.assistMessages,
    busy: state.assistBusy,
    canFix: assist.FIXABLE_STAGES.includes(state.selected),
    error: state.assistError,
  });
}

function renderContext(state) {
  const panel = el("context-panel");
  show(panel, Boolean(state.selected));
  if (!state.selected) return;

  const step = state.steps[state.selected] || {};
  el("context-title").textContent = `${state.selected} · ${step.status || "pending"}`;

  const ctx = state.context;
  // Prefer the context payload, but the live step already carries the error
  // from the websocket -- show it immediately so a slow context fetch cannot
  // hide why the stage failed.
  const error = ctx?.error || step.error || null;
  const traceback = ctx?.traceback || step.traceback || null;
  const failed = Boolean(error);

  if (!ctx) {
    const waiting = [];
    if (error) waiting.push(`<p class="error">${esc(error)}</p>`);
    const chat = assistBlock(state, failed);
    if (chat) waiting.push(chat);
    if (traceback) waiting.push(block("traceback", esc(traceback), { open: true }));
    waiting.push(`<p class="muted">loading…</p>`);
    el("context-body").innerHTML = waiting.join("");
    const thread = el("assist-thread");
    if (thread) thread.scrollTop = thread.scrollHeight;
    return;
  }

  const parts = [];
  if (error) parts.push(`<p class="error">${esc(error)}</p>`);
  const chat = assistBlock(state, failed);
  if (chat) parts.push(chat);
  if (traceback) parts.push(block("traceback", esc(traceback), { open: true }));
  // What this stage printed, kept with the stage rather than lost in the
  // run-wide pane. Absent entirely when the strategy printed nothing.
  if (ctx.logs?.length) {
    parts.push(block(`output · ${ctx.logs.length} line(s)`, esc(ctx.logs.join("\n"))));
  }
  if (ctx.source) parts.push(block("source", esc(ctx.source)));
  for (const [name, artifact] of Object.entries(ctx.inputs || {})) {
    parts.push(describe(`input: ${name}`, artifact));
  }
  parts.push(describe("output", ctx.output));
  parts.push(block("config", pretty(ctx.config)));

  el("context-body").innerHTML = parts.join("");
  const thread = el("assist-thread");
  if (thread) thread.scrollTop = thread.scrollHeight;

  const hint = el("debug-hint");
  show(hint, Boolean(state.debugPort));
  hint.textContent = state.debugPort
    ? `waiting for a debugger on 127.0.0.1:${state.debugPort} ` +
      `— attach with VS Code "Python: Remote Attach"`
    : "";
}

/** One flat terminal line per print, stage as a prefix. */
function renderLogs(state) {
  const node = el("logs");
  const logs = state.logs || [];
  node.innerHTML = logs.length
    ? logs.map(({ stage, line }) =>
        `<div class="term-line">` +
        `<span class="term-stage">${esc(stage || "run")}</span>` +
        `<span class="term-text">${esc(line)}</span>` +
        `</div>`).join("")
    : `<div class="term-empty">waiting for output…</div>`;
  node.scrollTop = node.scrollHeight;
}

export function render(state) {
  const banner = el("banner");
  show(banner, Boolean(state.error));
  banner.textContent = state.error || "";

  el("run-status").textContent = state.runStatus;
  el("run-status").dataset.status = state.runStatus;
  el("run-id").textContent = state.runId ? `run ${state.runId}` : "";

  renderOptions(el("strategy"), state.strategies, state.strategyId,
    (s) => `${s.filename} · ${s.class_name || "?"} · ${s.id}`);
  renderOptions(el("past-run"), state.runs, state.runId,
    (r) => `${r.id} · ${r.status}`);

  el("run").disabled = !state.strategyId || state.runStatus === "running";
  el("resume").disabled = state.runStatus === "running";

  renderStages(state);
  renderMetrics(state);
  renderChart(state);
  renderContext(state);

  renderLogs(state);
}
