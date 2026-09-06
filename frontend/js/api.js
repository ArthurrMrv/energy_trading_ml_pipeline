/**
 * One thin function per endpoint, and one place where an error becomes a
 * sentence a human can read.
 */

export class ApiError extends Error {}

/** FastAPI reports `detail` as a string, or -- for a 422 -- as a list of
 *  {loc, msg} objects. Both collapse to one line here so callers never have to
 *  care which shape they got. */
function readDetail(detail) {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail
      .map((item) => {
        const field = (item.loc || []).slice(1).join(".") || "body";
        return `${field}: ${item.msg}`;
      })
      .join("; ");
  }
  return JSON.stringify(detail);
}

async function request(path, options) {
  let response;
  try {
    response = await fetch(path, options);
  } catch {
    throw new ApiError(`cannot reach the API at ${location.origin}`);
  }

  const body = await response.json().catch(() => null);
  if (response.ok) return body;

  throw new ApiError(
    body && "detail" in body
      ? readDetail(body.detail)
      : `${response.status} ${response.statusText}`,
  );
}

const postJson = (path, payload, headers = {}) =>
  request(path, {
    method: "POST",
    headers: { "content-type": "application/json", ...headers },
    body: JSON.stringify(payload),
  });

export const listStrategies = () => request("/strategies");

export function uploadStrategy(file) {
  const form = new FormData();
  form.append("file", file);
  return request("/strategies", { method: "POST", body: form });
}

export const listRuns = () => request("/runs");
export const getRun = (runId) => request(`/runs/${runId}`);
export const getMetrics = (runId) => request(`/runs/${runId}/artifacts/metrics.json`);
/** The chart data: written only when the run actually ran a permutation test. */
export const getPermutation = (runId) =>
  request(`/runs/${runId}/artifacts/permutation.json`);
export const getContext = (runId, stage) =>
  request(`/runs/${runId}/steps/${stage}/context`);

export const createRun = (strategyId, config) =>
  postJson("/runs", { strategy_id: strategyId, config });

export const resumeRun = (runId, fromStage, debug) =>
  postJson(`/runs/${runId}/resume`, { from_stage: fromStage, debug });

export const assistChat = (payload, apiKey) =>
  postJson("/assist/chat", payload, { Authorization: `Bearer ${apiKey}` });

export const assistFix = (payload, apiKey) =>
  postJson("/assist/fix", payload, { Authorization: `Bearer ${apiKey}` });

/** Replays the run's history, then tails it live. */
export const openEvents = (runId) =>
  new WebSocket(
    `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}` +
      `/runs/${runId}/events`,
  );
