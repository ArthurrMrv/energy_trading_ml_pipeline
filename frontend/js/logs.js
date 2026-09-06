/** Grouping for the log pane. Pure, so it can be checked without a browser. */

/**
 * Consecutive lines from the same stage become one block.
 *
 * Grouping by stage *name* instead would fuse a resumed stage's output with
 * the output of the attempt that failed, which is the one moment you most need
 * to tell them apart.
 *
 * @param {{stage: string|null, line: string}[]} logs
 * @returns {{stage: string, lines: string[]}[]}
 */
export function groupLogs(logs) {
  const groups = [];
  for (const { stage, line } of logs || []) {
    const current = groups[groups.length - 1];
    // The accumulator is local, so appending to it mutates nothing shared.
    if (current && current.stage === stage) current.lines.push(line);
    else groups.push({ stage: stage || "run", lines: [line] });
  }
  return groups;
}
