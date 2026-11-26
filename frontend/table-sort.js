/* Lightweight client-side sorting for the generated team pages.
   The main app has its own sortable headers; these pages are static HTML. */
(function () {
  function numericValue(text) {
    const value = text.replace(/^Wk\s*/i, "").trim();
    const match = value.match(/^([+-]?\d+(?:\.\d+)?)/);
    return match ? Number(match[1]) : null;
  }

  function cellValue(row, index) {
    const text = (row.cells[index]?.innerText || "").trim();
    if (!text || text === "—") return null;
    const numeric = numericValue(text);
    return numeric === null ? text : numeric;
  }

  function compare(a, b, direction) {
    const aMissing = a === null;
    const bMissing = b === null;
    if (aMissing && bMissing) return 0;
    if (aMissing) return 1;
    if (bMissing) return -1;
    let result;
    if (typeof a === "string" || typeof b === "string") {
      result = String(a).localeCompare(String(b), undefined, { numeric: true });
    } else {
      result = a - b;
    }
    return direction === "asc" ? result : -result;
  }

  document.querySelectorAll("table.dtable").forEach((table) => {
    const headerRow = table.tHead?.rows[0];
    const body = table.tBodies[0];
    if (!headerRow || !body) return;
    const headers = [...headerRow.cells];
    headers.forEach((th, index) => {
      if (!th.textContent.trim() || th.querySelector("button")) return;
      const label = th.textContent.trim();
      const button = document.createElement("button");
      button.type = "button";
      button.appendChild(document.createTextNode(label));
      const caret = document.createElement("span");
      caret.className = "caret";
      caret.textContent = "⇅";
      button.appendChild(caret);
      button.title = `Sort by ${label.toLowerCase()}.`;
      th.textContent = "";
      th.appendChild(button);
      th.setAttribute("aria-sort", "none");

      button.addEventListener("click", () => {
        const direction =
          th.getAttribute("aria-sort") === "ascending" ? "desc" : "asc";
        headers.forEach((header) => {
          header.setAttribute("aria-sort", "none");
          header.querySelector("button")?.classList.remove("sorted");
          const marker = header.querySelector(".caret");
          if (marker) marker.textContent = "⇅";
        });
        th.setAttribute(
          "aria-sort",
          direction === "asc" ? "ascending" : "descending"
        );
        button.classList.add("sorted");
        caret.textContent = direction === "asc" ? "▲" : "▼";

        const rows = [...body.rows];
        rows.sort((a, b) =>
          compare(cellValue(a, index), cellValue(b, index), direction)
        );
        rows.forEach((row) => body.appendChild(row));
        if (window.nflTrack) window.nflTrack('sort_change', label);
      });
    });
  });
})();
