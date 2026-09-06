/** HTML escaping, in one place because two modules build markup from strings. */

/** Tracebacks, strategy source, column names and metric labels are arbitrary
 *  text and must never be parsed as markup. */
export const esc = (value) =>
  String(value ?? "").replace(
    /[&<>"']/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c],
  );

export const pretty = (value) => esc(JSON.stringify(value, null, 2));
