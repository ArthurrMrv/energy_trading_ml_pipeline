/** DeepSeek error assist: key in localStorage, chat markup, no server persistence. */

import { esc } from "./esc.js";
import { toHtml } from "./md.js";

const KEY = "pipeline.deepseek_api_key";

/** Stages whose failures can be patched in the uploaded strategy file.
 *  ``simulate`` is included because that is where ``on_tick`` runs. */
export const FIXABLE_STAGES = ["collect", "prepare", "fit", "simulate"];
export const STRATEGY_STAGES = FIXABLE_STAGES;

export function getKey() {
  try {
    return localStorage.getItem(KEY) || "";
  } catch {
    return "";
  }
}

export function setKey(value) {
  try {
    if (value) localStorage.setItem(KEY, value);
    else localStorage.removeItem(KEY);
  } catch {
    /* private mode / quota — ignore */
  }
}

function body(role, content) {
  if (role === "assistant") {
    if (content === "...") {
      return `<div class="assist-thinking" aria-label="thinking">...</div>`;
    }
    return `<div class="md">${toHtml(content)}</div>`;
  }
  return `<pre>${esc(content)}</pre>`;
}

function bubble(role, content) {
  return `<div class="assist-msg" data-role="${esc(role)}">` +
    `<span class="assist-role">${esc(role)}</span>` +
    `${body(role, content)}</div>`;
}

/** Markup for the assist bubble; wired by app.js after render. */
export function markup({ messages, busy, canFix, error }) {
  const lines = (messages || []).map((m) => bubble(m.role, m.content));
  // While the model is working, show a live assistant "..." so the thread
  // does not look frozen.
  if (busy) lines.push(bubble("assistant", "..."));

  return `<div class="assist" id="assist-panel">
  <div class="row spread">
    <strong>DeepSeek assist</strong>
    <span class="muted">${busy ? "thinking…" : ""}</span>
  </div>
  <div class="assist-thread" id="assist-thread">${lines.length
    ? lines.join("")
    : `<p class="muted">Explain this error in plain English, or ask a follow-up.</p>`}
  </div>
  ${error ? `<p class="error">${esc(error)}</p>` : ""}
  <div class="row assist-actions">
    <input type="text" id="assist-input" placeholder="ask a follow-up…"
           ${busy ? "disabled" : ""} />
    <button type="button" id="assist-send" ${busy ? "disabled" : ""}>Send</button>
    <button type="button" id="assist-fix" ${busy || !canFix ? "disabled" : ""}>
      Fix strategy
    </button>
  </div>
</div>`;
}
