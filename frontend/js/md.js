/** Minimal Markdown → HTML for assist replies. Escapes raw HTML first. */

import { esc } from "./esc.js";

function inline(text) {
  let out = esc(text);
  out = out.replace(/`([^`]+)`/g, "<code>$1</code>");
  out = out.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  out = out.replace(/(?<!\*)\*([^*]+)\*(?!\*)/g, "<em>$1</em>");
  out = out.replace(
    /\[([^\]]+)\]\((https?:[^)\s]+)\)/g,
    '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>',
  );
  return out;
}

/** Convert a Markdown string to safe HTML (subset: fences, lists, headings, inline). */
export function toHtml(src) {
  if (!src) return "";
  const lines = String(src).replace(/\r\n/g, "\n").split("\n");
  const parts = [];
  let i = 0;
  let para = [];

  const flushPara = () => {
    if (!para.length) return;
    parts.push(`<p>${inline(para.join(" "))}</p>`);
    para = [];
  };

  while (i < lines.length) {
    const line = lines[i];

    if (line.startsWith("```")) {
      flushPara();
      const lang = esc(line.slice(3).trim());
      i += 1;
      const body = [];
      while (i < lines.length && !lines[i].startsWith("```")) {
        body.push(lines[i]);
        i += 1;
      }
      i += 1; // closing fence
      parts.push(`<pre class="md-code"${lang ? ` data-lang="${lang}"` : ""}>` +
        `<code>${esc(body.join("\n"))}</code></pre>`);
      continue;
    }

    const heading = /^(#{1,3})\s+(.+)$/.exec(line);
    if (heading) {
      flushPara();
      const level = heading[1].length;
      parts.push(`<h${level}>${inline(heading[2])}</h${level}>`);
      i += 1;
      continue;
    }

    if (/^\s*[-*]\s+/.test(line)) {
      flushPara();
      const items = [];
      while (i < lines.length && /^\s*[-*]\s+/.test(lines[i])) {
        items.push(`<li>${inline(lines[i].replace(/^\s*[-*]\s+/, ""))}</li>`);
        i += 1;
      }
      parts.push(`<ul>${items.join("")}</ul>`);
      continue;
    }

    if (/^\s*\d+\.\s+/.test(line)) {
      flushPara();
      const items = [];
      while (i < lines.length && /^\s*\d+\.\s+/.test(lines[i])) {
        items.push(`<li>${inline(lines[i].replace(/^\s*\d+\.\s+/, ""))}</li>`);
        i += 1;
      }
      parts.push(`<ol>${items.join("")}</ol>`);
      continue;
    }

    if (!line.trim()) {
      flushPara();
      i += 1;
      continue;
    }

    para.push(line.trim());
    i += 1;
  }
  flushPara();
  return parts.join("\n");
}
