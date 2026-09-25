// Builds outputs/report/report_template.docx: section skeleton with [WRITE: ...] placeholders (no prose),
// tables T1-T6 and figures from outputs/report/, numbered captions from figures_list.md, equations in Methodology.
// Usage (from this folder): npm install && node build.js
const fs = require("fs");
const path = require("path");
const {
  AlignmentType, Document, Footer, HeadingLevel, ImageRun, Math: OMath, MathFraction, MathRadical, MathRun,
  MathSubScript, MathSubSuperScript, MathSum, MathSuperScript, Packer, PageBreak, PageNumber, Paragraph,
  ShadingType, Table, TableCell, TableRow, TabStopType, TextRun, WidthType, BorderStyle,
} = require("docx");

const ROOT = path.resolve(__dirname, "..", "..");
const REPORT = path.join(ROOT, "outputs", "report");
const FIG = path.join(REPORT, "figures");
const TAB = path.join(REPORT, "tables");
const OUT = path.join(REPORT, "report_template.docx");

const PAGE_W = 11906, PAGE_H = 16838, MARGIN = 1440;          // A4, 1-inch margins (DXA)
const CONTENT_W = PAGE_W - 2 * MARGIN;                       // 9026 DXA = 6.27 in
const FONT = "Times New Roman";

// ---------------------------------------------------------------------------------------------- inputs
function mdRows(file) {
  const lines = fs.readFileSync(file, "utf8").split(/\r?\n/);
  const title = (lines.find((l) => l.startsWith("# ")) || "").slice(2).trim();
  const table = lines.filter((l) => l.startsWith("|"));
  const split = (l) => l.trim().replace(/^\|/, "").replace(/\|$/, "").split(/(?<!\\)\|/)
    .map((c) => c.trim().replace(/\\\|/g, "|"));
  const rows = table.filter((l, i) => i !== 1).map(split);   // drop the |---| separator
  const notes = lines.filter((l) => l.startsWith("- ")).map((l) => l.slice(2));
  return { title, header: rows[0], body: rows.slice(1), notes };
}

function tableFile(prefix) {
  const f = fs.readdirSync(TAB).find((x) => x.startsWith(prefix + "_") && x.endsWith(".md"));
  if (!f) throw new Error(`missing table ${prefix}`);
  return path.join(TAB, f);
}

const captions = {};
for (const r of mdRows(path.join(REPORT, "figures_list.md")).body) captions[r[0]] = r[1];

const methods = fs.readFileSync(path.join(REPORT, "methods_facts.md"), "utf8");
const capMatch = /Recency cap: (\d+) days/.exec(methods);
if (!capMatch) throw new Error("recency cap not found in methods_facts.md");
const RECENCY_CAP = capMatch[1];

function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}

// ---------------------------------------------------------------------------------------------- helpers
const P = (text, opts = {}) => new Paragraph({ children: [new TextRun({ text, ...opts.run })], ...opts.para });
const placeholder = (text) => new Paragraph({
  spacing: { after: 120 },
  children: [new TextRun({ text: `[${text}]`, italics: true, highlight: "yellow" })],
});
const h1 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(text)] });
const h2 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(text)] });

let figNo = 0, tabNo = 0, eqNo = 0;

function figure(file, widthIn = 6.2, maxHeightIn = 8.2) {
  const f = path.join(FIG, file);
  if (!fs.existsSync(f)) throw new Error(`missing figure ${file}`);
  if (!captions[file]) throw new Error(`no caption for ${file} in figures_list.md`);
  const { w, h } = pngSize(f);
  let wIn = widthIn, hIn = (widthIn * h) / w;
  if (hIn > maxHeightIn) { hIn = maxHeightIn; wIn = (maxHeightIn * w) / h; }
  figNo += 1;
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 120 },
      children: [new ImageRun({
        type: "png", data: fs.readFileSync(f), transformation: { width: Math.round(wIn * 96), height: Math.round(hIn * 96) },
        altText: { title: file, description: captions[file], name: file },
      })],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER, spacing: { after: 240 },
      children: [new TextRun({ text: `Figure ${figNo}. `, bold: true }), new TextRun(captions[file])],
    }),
  ];
}

const cellBorder = { style: BorderStyle.SINGLE, size: 4, color: "999999" };
const borders = { top: cellBorder, bottom: cellBorder, left: cellBorder, right: cellBorder };

function cellParas(text, size, bold) {
  // "0.7031 [0.6890, 0.7174]" -> value and CI on separate lines
  const m = /^(\S+) (\[[^\]]+\])$/.exec(text);
  const parts = m ? [m[1], m[2]] : [text];
  return parts.map((t) => new Paragraph({
    spacing: { line: 240, before: 0, after: 0 },
    children: [new TextRun({ text: t, size, bold, font: FONT })],
  }));
}

function table(prefix, { transpose = false } = {}) {
  const t = mdRows(tableFile(prefix));
  let header = t.header, body = t.body;
  if (transpose) {
    const all = [header, ...body];
    header = all.map((r) => r[0]);
    body = header.length ? all[0].slice(1).map((_, j) => all.map((r) => r[j + 1])) : [];
  }
  const ncol = header.length;
  const size = ncol >= 8 ? 15 : ncol >= 6 ? 16 : 18;            // half-points: 7.5 / 8 / 9 pt
  const score = header.map((hd, j) => {
    const longest = Math.max(...[hd, ...body.map((r) => r[j] || "")].map((c) => {
      const m = /^(\S+) (\[[^\]]+\])$/.exec(c);
      return m ? Math.max(m[1].length, m[2].length) : c.length;
    }));
    return Math.min(Math.max(longest, 6), 34);
  });
  const total = score.reduce((a, b) => a + b, 0);
  const widths = score.map((s) => Math.floor((s / total) * CONTENT_W));
  widths[widths.length - 1] += CONTENT_W - widths.reduce((a, b) => a + b, 0);
  const row = (cells, isHeader) => new TableRow({
    tableHeader: isHeader, cantSplit: true,
    children: cells.map((c, j) => new TableCell({
      width: { size: widths[j], type: WidthType.DXA }, borders,
      margins: { top: 40, bottom: 40, left: 60, right: 60 },
      shading: isHeader ? { type: ShadingType.CLEAR, color: "auto", fill: "E7E6E6" } : undefined,
      children: cellParas(c || "", size, isHeader),
    })),
  });
  tabNo += 1;
  const title = t.title.replace(/^T\w+\.\s*/, "");
  return [
    new Paragraph({
      keepNext: true, spacing: { before: 240, after: 120 },
      children: [new TextRun({ text: `Table ${tabNo}. `, bold: true }), new TextRun(title)],
    }),
    new Table({ width: { size: CONTENT_W, type: WidthType.DXA }, columnWidths: widths,
      rows: [row(header, true), ...body.map((r) => row(r, false))] }),
    ...t.notes.map((n) => new Paragraph({ spacing: { line: 240, before: 60 },
      children: [new TextRun({ text: `Note: ${n}`, size: 20 })] })),
    new Paragraph({ spacing: { after: 120 }, children: [] }),
  ];
}

// ---------------------------------------------------------------------------------------------- equations
const r = (t) => new MathRun(t);
const frac = (num, den) => new MathFraction({ numerator: num, denominator: den });
const sub = (base, s) => new MathSubScript({ children: base, subScript: s });
const sup = (base, s) => new MathSuperScript({ children: base, superScript: s });
const sum = (children, lo, hi) => new MathSum({ children, subScript: lo, superScript: hi });

function equation(name, children) {
  eqNo += 1;
  return [
    new Paragraph({
      tabStops: [{ type: TabStopType.CENTER, position: CONTENT_W / 2 }, { type: TabStopType.RIGHT, position: CONTENT_W }],
      spacing: { before: 60, after: 60 },
      children: [new TextRun("\t"), new OMath({ children }), new TextRun(`\t(${eqNo})`)],
    }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 120 },
      children: [new TextRun({ text: name, size: 20, italics: true })] }),
  ];
}

const yi = [sub([r("y")], [r("i")])];
const yhat = [sub([r("ŷ")], [r("i")])];
const sum_n = (children) => sum(children, [r("i=1")], [r("n")]);

const EQ = {
  recency: equation(`Recency (R_u = ${RECENCY_CAP} days when the user has no purchase in the window)`,
    [sub([r("R")], [r("u")]), r(" = "), sub([r("t")], [r("ref")]), r(" − "),
    sub([r("max")], [r("i∈"), sub([r("P")], [r("u")])]), sub([r(" t")], [r("i")])]),
  frequency: equation("Frequency", [sub([r("F")], [r("u")]), r(" = |"), sub([r("P")], [r("u")]), r("|")]),
  monetary: equation("Monetary", [sub([r("M")], [r("u")]), r(" = "),
    sum([sub([r("price")], [r("i")])], [r("i∈"), sub([r("P")], [r("u")])], [])]),
  kmeans: equation("K-means objective", [r("J = "),
    sum([sum([sup([r("‖"), sub([r("x")], [r("i")]), r(" − "), sub([r("μ")], [r("k")]), r("‖")], [r("2")])],
      [sub([r("x")], [r("i")]), r("∈"), sub([r("C")], [r("k")])], [])], [r("k=1")], [r("K")])]),
  rf: equation("Random Forest average over T trees", [r("ŷ(x) = "), frac([r("1")], [r("T")]),
    sum([sub([r("h")], [r("t")]), r("(x)")], [r("t=1")], [r("T")])]),
  hurdle: equation("Hurdle model expected spend", [r("E[S | x] = P(Y = 1 | x) · E[S | Y = 1, x]")]),
  prior: equation("Prior correction of class-weighted probabilities", [r("odds = "), frac([r("p")], [r("1 − p")]),
    r(" · "), frac([sub([r("n")], [r("1")])], [sub([r("n")], [r("0")])]), r(",   p′ = "),
    frac([r("odds")], [r("1 + odds")])]),
  mae: equation("Mean absolute error", [r("MAE = "), frac([r("1")], [r("n")]),
    sum_n([r("|"), ...yi, r(" − "), ...yhat, r("|")])]),
  rmse: equation("Root mean squared error", [r("RMSE = "), new MathRadical({ children: [frac([r("1")], [r("n")]),
    sum_n([sup([r("("), ...yi, r(" − "), ...yhat, r(")")], [r("2")])])] })]),
  r2: equation("Coefficient of determination", [sup([r("R")], [r("2")]), r(" = 1 − "),
    frac([sum_n([sup([r("("), ...yi, r(" − "), ...yhat, r(")")], [r("2")])])],
      [sum_n([sup([r("("), ...yi, r(" − ȳ)")], [r("2")])])])]),
  precision: equation("Precision", [r("Precision = "), frac([r("TP")], [r("TP + FP")])]),
  recall: equation("Recall", [r("Recall = "), frac([r("TP")], [r("TP + FN")])]),
  f1: equation("F1 score", [sub([r("F")], [r("1")]), r(" = "), frac([r("2 · Precision · Recall")],
    [r("Precision + Recall")])]),
};

// ---------------------------------------------------------------------------------------------- document
const children = [
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 2400, after: 480 },
    children: [new TextRun({ text: "[WRITE: Title, 10–15 words]", bold: true, size: 32, italics: true, highlight: "yellow" })] }),
  new Paragraph({ alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "[WRITE: Author names and register numbers]", italics: true, highlight: "yellow" })] }),
  new Paragraph({ alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "[WRITE: Course code and title, faculty name, SCOPE, VIT, month and year]", italics: true, highlight: "yellow" })] }),
  new Paragraph({ children: [new PageBreak()] }),

  h1("Aim"), placeholder("WRITE: Aim, 1–2 sentences, 30–60 words"),
  h1("Objective"), placeholder("WRITE: Objectives as 4–6 numbered points, 80–150 words"),
  h1("Abstract"), placeholder("WRITE: Abstract, 500–700 words"),
  placeholder("WRITE: Keywords, 5–7 terms"),
  h1("Introduction"),
  placeholder("WRITE: Background and motivation, 300–400 words"),
  placeholder("WRITE: Problem statement, 150–200 words"),
  placeholder("WRITE: Contributions of this work, 150–250 words"),
  placeholder("WRITE: Organisation of the report, 50–100 words"),
  h1("Related Material / Literature Review"),
  placeholder("WRITE: Literature review, 1,000–1,500 words"),
  placeholder("ADD: Summary table of 8–12 related studies (author, year, dataset, method, key result) and a research-gap paragraph, 150–250 words"),

  h1("Proposed Methodology"),
  h2("Overall architecture"),
  ...figure("architecture.png", 5.6, 8.0),
  placeholder("WRITE: Walk-through of Figure 1, 150–250 words"),
  h2("Dataset"),
  ...table("T1"),
  placeholder("WRITE: Dataset description and user sampling, 150–250 words"),
  h2("Observation and prediction windows"),
  placeholder("WRITE: Time-window design and why it prevents leakage, 100–200 words"),
  h2("Feature engineering (RFM and behavioral features)"),
  ...EQ.recency, ...EQ.frequency, ...EQ.monetary,
  placeholder("WRITE: Symbols in Equations (1)–(3) and the behavioral features, 200–300 words"),
  h2("Customer segmentation (K-means)"),
  ...EQ.kmeans,
  ...figure("elbow_plot.png", 4.6), ...figure("silhouette_plot.png", 4.6),
  placeholder("WRITE: Scaling, choice of K and segment naming, 150–250 words"),
  h2("Predictive models"),
  ...EQ.rf,
  placeholder("WRITE: Classification and regression models, class weighting and hyperparameters, 200–300 words"),
  h2("Hurdle model for expected spend"),
  ...EQ.hurdle, ...EQ.prior,
  placeholder("WRITE: Two-stage design, stage-1 selection and probability correction, 150–250 words"),
  h2("Evaluation metrics"),
  ...EQ.mae, ...EQ.rmse, ...EQ.r2, ...EQ.precision, ...EQ.recall, ...EQ.f1,
  placeholder("WRITE: Metrics, threshold tuning, bootstrap confidence intervals and leakage audit, 150–250 words"),

  h1("Results and Discussion"),
  h2("Purchase prediction and feature-set comparison"),
  ...table("T2"), ...figure("roc_curves.png"),
  placeholder("WRITE: Discussion of Table 2 and Figure 4, 200–300 words"),
  h2("Spend prediction: single-stage vs hurdle"),
  ...table("T3"), ...figure("actual_vs_predicted.png"),
  placeholder("WRITE: Discussion of Table 3 and Figure 5, 200–300 words"),
  h2("Leakage audit"),
  ...table("T4"),
  placeholder("WRITE: Discussion of Table 4, 150–200 words"),
  h2("Customer segments"),
  ...table("T5", { transpose: true }), ...figure("cluster_profiles.png"),
  placeholder("WRITE: Discussion of Table 5 and Figure 6, 150–250 words"),
  h2("Explainability"),
  ...table("T6"), ...figure("shap_summary.png", 4.9), ...figure("rf_feature_importance.png", 4.9),
  ...figure("shap_summary_segment_casual_visitors.png", 3.9),
  ...figure("shap_summary_segment_engaged_browsers.png", 3.9),
  ...figure("shap_summary_segment_recent_buyers.png", 3.9),
  placeholder("WRITE: Discussion of Table 6 and Figures 7–11, 250–350 words"),
  h2("Discussion"),
  placeholder("WRITE: Overall discussion, limitations and threats to validity, 300–500 words"),

  h1("Conclusion"),
  placeholder("WRITE: Conclusion, 250–400 words"),
  placeholder("WRITE: Future work, 100–150 words"),
  h1("References"),
  placeholder("ADD: 15–25 references in the required citation style"),
];

const doc = new Document({
  creator: "pipeline", title: "Report template",
  styles: {
    default: {
      document: { run: { font: FONT, size: 24 }, paragraph: { spacing: { line: 360 } } },
    },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 28, bold: true }, paragraph: { spacing: { before: 360, after: 120, line: 360 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 24, bold: true }, paragraph: { spacing: { before: 240, after: 120, line: 360 }, outlineLevel: 1 } },
    ],
  },
  sections: [{
    properties: { page: { size: { width: PAGE_W, height: PAGE_H }, margin: { top: MARGIN, bottom: MARGIN, left: MARGIN, right: MARGIN } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 24 })] })] }) },
    children,
  }],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(OUT, buf);
  console.log(`wrote ${path.relative(ROOT, OUT)}: ${tabNo} tables, ${figNo} figures, ${eqNo} equations`);
});
