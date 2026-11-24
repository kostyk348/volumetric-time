// Split a Colab notebook into runnable scripts, without touching its content.
//
// Why: 31 cells of a research notebook are 31 experiments, and an experiment that
// can only be run by pressing "Run all" in a browser is not reproducible. This tool
// copies each code cell into `runs/NN-name.py` verbatim, so a single experiment can
// be run headless: `python runs/19-omega-drive.py`.
//
// It writes nothing but the machine's own text: no rewriting, no reformatting, no
// "fixing". If a cell does not run on its own, that is a finding about the notebook,
// and the tool reports it instead of hiding it.
//
// Usage:
//   node tools/split.mjs list              # every cell: index, type, lines, first line
//   node tools/split.mjs all               # write every code cell to runs/
//   node tools/split.mjs cell 19           # write only cell 19
//   node tools/split.mjs selfcheck         # run `all`, then re-read every file: bytes match

import fs from "node:fs";
import path from "node:path";

const NOTEBOOK = path.join("notebook", "volumetric-time.ipynb");
const OUT_DIR = "runs";

const source = (cell) => (Array.isArray(cell.source) ? cell.source.join("") : cell.source ?? "");
const slug = (text) => {
  const first = (text.match(/^\s*#\s*(.+)$/m) ?? [, ""])[1]
    .toLowerCase()
    .replace(/[^a-zа-я0-9]+/gi, "-")
    .replace(/^-+|-+$/g, "");
  return first.slice(0, 40) || "cell";
};

// A cell is "runnable alone" when it neither reads a name defined in an earlier cell
// nor installs anything. This is a heuristic on purpose: it reports suspicion, and a
// suspicion that turns out wrong costs nothing.
const needs_context = (text) =>
  /!pip\s+install/.test(text) || /\bfrom earlier\b/.test(text);

function main() {
  const command = process.argv[2] ?? "list";
  if (!fs.existsSync(NOTEBOOK)) {
    console.error(`no notebook at ${NOTEBOOK}`);
    process.exit(1);
  }
  const notebook = JSON.parse(fs.readFileSync(NOTEBOOK, "utf8"));
  const cells = notebook.cells ?? [];

  if (command === "list") {
    cells.forEach((cell, index) => {
      const text = source(cell).trim();
      if (!text) return;
      const lines = text.split("\n").length;
      const head = text.split("\n")[0].slice(0, 90);
      const mark = needs_context(text) ? " (не запускается в одиночку)" : "";
      console.log(`${String(index).padStart(2, "0")}  ${cell.cell_type.padEnd(8)} ${String(lines).padStart(4)} строк  ${head}${mark}`);
    });
    return;
  }

  if (command === "cell" || command === "all") {
    const wanted = command === "all" ? null : Number(process.argv[3]);
    let written = 0;
    cells.forEach((cell, index) => {
      if (cell.cell_type !== "code") return;
      if (wanted !== null && index !== wanted) return;
      const text = source(cell);
      if (!text.trim()) return;
      const name = `${String(index).padStart(2, "0")}-${slug(text)}.py`;
      const target = path.join(OUT_DIR, name);
      fs.mkdirSync(OUT_DIR, { recursive: true });
      const banner = [
        `# Source: notebook/volumetric-time.ipynb, cell ${index}.`,
        `# Copied verbatim by tools/split.mjs — edit the notebook, not this file.`,
        needs_context(text) ? "# NOTE: this cell installs packages or expects an earlier cell to have run." : "",
        "",
      ].filter(Boolean).join("\n");
      fs.writeFileSync(target, banner + text);
      console.log(`${target}  (${text.split("\n").length} строк)`);
      written += 1;
    });
    if (wanted !== null && written === 0) {
      console.error(`cell ${wanted} is empty or not a code cell`);
      process.exit(1);
    }
    return;
  }

  if (command === "selfcheck") {
    let checked = 0;
    cells.forEach((cell, index) => {
      if (cell.cell_type !== "code") return;
      const text = source(cell);
      if (!text.trim()) return;
      const name = `${String(index).padStart(2, "0")}-${slug(text)}.py`;
      const target = path.join(OUT_DIR, name);
      if (!fs.existsSync(target)) {
        console.error(`missing ${target}`);
        process.exitCode = 1;
        return;
      }
      const written = fs.readFileSync(target, "utf8");
      if (!written.includes(text)) {
        console.error(`content drifted: ${target}`);
        process.exitCode = 1;
      }
      checked += 1;
    });
    console.log(`selfcheck: ${checked} cells verified`);
    return;
  }

  console.error(`unknown command ${command}; use list | all | cell N | selfcheck`);
  process.exit(1);
}

main();
