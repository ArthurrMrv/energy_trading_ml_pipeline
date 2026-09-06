/** The Monte Carlo permutation chart, as an SVG string.
 *
 *  Every permuted equity path in grey, the real one on top. If the strategy is
 *  luck, its line disappears into the crowd; if it is not, it leaves.
 *
 *  Pure -- it takes numbers and returns markup, touches no DOM and fetches
 *  nothing, which is what makes it checkable without a browser.
 */

import { esc } from "./esc.js";

const WIDTH = 640;
const HEIGHT = 200;
const PAD = { top: 12, right: 10, bottom: 22, left: 52 };

const PLOT_W = WIDTH - PAD.left - PAD.right;
const PLOT_H = HEIGHT - PAD.top - PAD.bottom;

const finite = (value) => Number.isFinite(value);

/** Low and high across every curve, so the real line can never clip. */
function bounds(curves) {
  let low = Infinity;
  let high = -Infinity;
  for (const curve of curves) {
    for (const value of curve) {
      if (!finite(value)) continue;
      if (value < low) low = value;
      if (value > high) high = value;
    }
  }
  if (low > high) return null;
  // A book that never moved is still worth drawing; it just needs a box.
  return low === high ? [low - 1, high + 1] : [low, high];
}

const label = (value) =>
  Math.abs(value) >= 10000 || (value !== 0 && Math.abs(value) < 0.01)
    ? value.toExponential(1)
    : String(Number(value.toFixed(2)));

/**
 * @param {{x: number[], strategy: number[], null: number[][]}} curves
 * @param {object} summary the scalar permutation result, for the caption
 * @returns {string} SVG markup, or "" when there is nothing to draw
 */
export function permutationChart(curves, summary = {}) {
  const strategy = curves?.strategy || [];
  const paths = curves?.null || [];
  if (strategy.length < 2) return "";

  const span = bounds([strategy, ...paths]);
  if (!span) return "";
  const [low, high] = span;

  const sx = (i, n) => PAD.left + (n < 2 ? PLOT_W / 2 : (i / (n - 1)) * PLOT_W);
  const sy = (value) =>
    PAD.top + PLOT_H - ((value - low) / (high - low)) * PLOT_H;

  /** A non-finite point breaks the line rather than inventing a segment. */
  const line = (curve) => {
    let d = "";
    let pen = "M";
    curve.forEach((value, i) => {
      if (!finite(value)) { pen = "M"; return; }
      d += `${pen}${sx(i, curve.length).toFixed(1)} ${sy(value).toFixed(1)}`;
      pen = "L";
    });
    return d;
  };

  const nulls = paths
    .map((curve) => `<path class="null-path" d="${line(curve)}" />`)
    .join("");

  const start = finite(strategy[0]) ? strategy[0] : 0;
  const ticks = curves.x || [];
  const first = ticks[0] ?? 0;
  const last = ticks[ticks.length - 1] ?? strategy.length - 1;

  return `<figure class="chart">
  <svg viewBox="0 0 ${WIDTH} ${HEIGHT}" role="img"
       aria-label="strategy equity against ${paths.length} permuted paths">
    <rect class="chart-bg" x="${PAD.left}" y="${PAD.top}"
          width="${PLOT_W}" height="${PLOT_H}" />
    <line class="chart-base" x1="${PAD.left}" x2="${WIDTH - PAD.right}"
          y1="${sy(start).toFixed(1)}" y2="${sy(start).toFixed(1)}" />
    ${nulls}
    <path class="real-path" d="${line(strategy)}" />
    <text class="tick" x="${PAD.left - 6}" y="${PAD.top + 4}"
          text-anchor="end">${esc(label(high))}</text>
    <text class="tick" x="${PAD.left - 6}" y="${PAD.top + PLOT_H}"
          text-anchor="end">${esc(label(low))}</text>
    <text class="tick" x="${PAD.left}" y="${HEIGHT - 6}">${esc(first)}</text>
    <text class="tick" x="${WIDTH - PAD.right}" y="${HEIGHT - 6}"
          text-anchor="end">${esc(last)}</text>
  </svg>
  <figcaption>
    <span class="key real">strategy</span>
    <span class="key null">${paths.length} permuted</span>
    <span>p = ${esc(summary.p_value ?? "?")} on ${esc(summary.metric ?? "?")}
      · ${esc(summary.n ?? paths.length)} × ${esc(summary.method ?? "?")}</span>
  </figcaption>
</figure>`;
}
