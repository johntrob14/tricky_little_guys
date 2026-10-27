  const FCOL = () => ({ "Base gpt-oss-120b": "var(--cds-chart-muted)", "CoT-trained": dash.colors[0], "Output-only RL": dash.colors[1], "Length/effort RL": dash.colors[3], "Output-only RL, CoT off": dash.colors[2], "Control: CoT off, no behaviors": dash.colors[5] });
  function fillMarks() {
    document.querySelectorAll(".rx [data-source][data-field]").forEach(el => {
      if (el.tagName === "TH") return; const rows = ok(el.dataset.source); el.classList.remove("dash-skeleton");
      if (!rows) { el.textContent = "—"; return; }
      const conds = (el.dataset.where || "").split("&").map(p => p.split("=").map(z => decodeURIComponent(z)));
      const r = rows.find(x => conds.every(([k, v]) => String(x[k]) === v)); if (!r) { el.textContent = "—"; return; }
      const f = el.dataset.field, v = r[f];
      el.textContent = /^(length|control|raw|auc_s|auc_c)$/.test(f) ? pts(v) : /^(out|cot|out_all|cot_neel)$/.test(f) ? pc(v) : String(v);
    });
  }
  function table(id, rows, cells, key) {
    const tb = document.querySelector(`#${id} tbody`); tb.innerHTML = ""; if (!rows) return;
    for (const r of rows) { const tr = document.createElement("tr"); if (key) tr.dataset.key = key(r);
      cells.forEach(([f, num, fmt]) => { const td = document.createElement("td"); if (num) td.className = "num"; const v = r[f]; td.textContent = v == null ? "—" : fmt ? fmt(v) : v; tr.appendChild(td); }); tb.appendChild(tr); }
  }
  function scatter(el, rows, xk, yk, opts) {
    const h = opts.h || 320, m = { t: 12, r: 16, b: 40, l: 48 }; const [svg, w] = svgIn(el, h); const t = tip(el);
    const x = d3.scaleLinear([0, opts.xmax || 100], [m.l, w - m.r]), y = d3.scaleLinear([0, 100], [h - m.b, m.t]);
    svg.append("g").attr("transform", `translate(0,${h - m.b})`).call(d3.axisBottom(x).ticks(5).tickFormat(opts.xfmt || (v => `${v}%`)).tickSize(-(h - m.t - m.b))).call(axisStyle);
    svg.append("g").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(5).tickFormat(v => `${v}%`).tickSize(-(w - m.l - m.r))).call(axisStyle).call(g => g.select(".domain").remove());
    if (!opts.nodiag) svg.append("line").attr("x1", x(0)).attr("y1", y(0)).attr("x2", x(100)).attr("y2", y(100)).attr("stroke", "var(--cds-chart-reference)").attr("stroke-dasharray", "4 4");
    svg.append("text").attr("x", (m.l + w - m.r) / 2).attr("y", h - 6).attr("text-anchor", "middle").attr("fill", "var(--color-fg-muted)").style("font-size", "var(--cds-font-size-caption)").text(opts.xlab);
    svg.append("text").attr("transform", `translate(12,${(m.t + h - m.b) / 2}) rotate(-90)`).attr("text-anchor", "middle").attr("fill", "var(--color-fg-muted)").style("font-size", "var(--cds-font-size-caption)").text(opts.ylab);
    rows.forEach(r => {
      if (r[xk] == null || r[yk] == null) return;
      const c = opts.color(r);
      svg.append("circle").attr("cx", x(r[xk])).attr("cy", y(r[yk])).attr("r", opts.r || 6).attr("fill", c).attr("fill-opacity", opts.alpha || 1).attr("stroke", "var(--color-panel)").attr("stroke-width", 1.5)
        .on("mousemove", ev => showTip(el, t, ev, opts.title(r), [[c, opts.ylab, pc(r[yk])], [null, opts.xlab, opts.xfmt ? opts.xfmt(r[xk]) : pc(r[xk])]])).on("mouseleave", () => { t.hidden = true; });
      if (opts.label) svg.append("text").attr("x", x(r[xk]) + 8).attr("y", y(r[yk])).attr("dy", ".35em").attr("fill", "var(--color-fg-muted)").style("font-size", "var(--cds-font-size-caption)").style("pointer-events", "none").text(opts.label(r));
    });
  }
  let curveMetric = "partial";
  for (const [id, mm] of [["cm-partial", "partial"], ["cm-strict", "strict"]]) document.getElementById(id).addEventListener("click", () => {
    curveMetric = mm; document.getElementById("cm-partial").setAttribute("aria-pressed", mm === "partial"); document.getElementById("cm-strict").setAttribute("aria-pressed", mm === "strict"); draw(); });

  function draw() {
    fillMarks(); const FC = FCOL();
    // 1. length decomposition
    const LD = ok("length_decomp"), cD = document.getElementById("c-decomp");
    legend(document.getElementById("lg-decomp"), [[dash.colors[3], "Length part"], [dash.colors[0], "Control-at-length part"]]);
    if (LD) {
      const rowH = 34, top = 8, bottom = 28, h = top + bottom + LD.length * rowH; const [svg, w] = svgIn(cD, h); const t = tip(cD);
      const lw = Math.min(w * 0.4, measure(svg, LD.map(r => r.label)) + 14);
      const lo = d3.min(LD, r => Math.min(r.length_lo, r.control_lo, 0)), hi = d3.max(LD, r => Math.max(r.length_hi, r.control_hi, 0));
      const x = d3.scaleLinear([lo - 2, hi + 2], [lw, w - 22]).nice();
      svg.append("g").attr("transform", `translate(0,${top + LD.length * rowH})`).call(d3.axisBottom(x).ticks(Math.max(3, Math.floor((w - lw) / 70))).tickFormat(v => `${v > 0 ? "+" : ""}${v}`).tickSize(-(LD.length * rowH))).call(axisStyle);
      svg.append("line").attr("x1", x(0)).attr("x2", x(0)).attr("y1", top).attr("y2", top + LD.length * rowH).attr("stroke", "var(--cds-chart-reference)");
      LD.forEach((r, i) => { const y0 = top + i * rowH;
        svg.append("text").attr("x", lw - 8).attr("y", y0 + rowH / 2).attr("dy", ".35em").attr("text-anchor", "end").attr("fill", "var(--color-fg)").style("font-size", "var(--cds-font-size-caption)").text(r.label).call(sel => fit(sel, lw - 12));
        [["length", dash.colors[3], 4], ["control", dash.colors[0], 17]].forEach(([k, c, dy]) => {
          const a = x(Math.min(0, r[k])), b = x(Math.max(0, r[k]));
          svg.append("rect").attr("x", a).attr("width", Math.max(1, b - a)).attr("y", y0 + dy).attr("height", 11).attr("rx", 2).attr("fill", c);
          svg.append("line").attr("x1", x(r[k + "_lo"])).attr("x2", x(r[k + "_hi"])).attr("y1", y0 + dy + 5.5).attr("y2", y0 + dy + 5.5).attr("stroke", "var(--color-fg-muted)").attr("stroke-width", 1); });
        svg.append("rect").attr("x", 0).attr("width", w).attr("y", y0).attr("height", rowH).attr("fill", "transparent")
          .on("mousemove", ev => showTip(cD, t, ev, `${r.label} · ${r.family}`, [[null, "raw", `${pts(r.raw)} [${pts(r.raw_lo)}, ${pts(r.raw_hi)}]`], [dash.colors[3], "length part", `${pts(r.length)} [${pts(r.length_lo)}, ${pts(r.length_hi)}]`], [dash.colors[0], "control part", `${pts(r.control)} [${pts(r.control_lo)}, ${pts(r.control_hi)}]`]]))
          .on("mouseleave", () => { t.hidden = true; }); });
    } else fail(cD, "length_decomp");
    table("t-len", LD, [["label"], ["family"], ["median_words", 1], ["raw", 1, pts], ["length", 1, pts], ["control", 1, pts], ["auc_s", 1, pts], ["auc_c", 1, pts]], r => r.model);
    // curves
    const LC = ok("length_curves"), cC = document.getElementById("c-curves");
    const CUR = ["base", "mt1b-s150", "nocotrl1-s80x", "orga2-s40", "spillx1b-s20"], ccol = ["var(--cds-chart-muted)", dash.colors[0], dash.colors[2], dash.colors[1], dash.colors[3]];
    if (LC) {
      legend(document.getElementById("lg-curves"), CUR.map((mo, i) => [ccol[i], (LC.find(r => r.model === mo) || {}).label || mo]));
      const f = curveMetric, bins = [...new Set(LC.map(r => r.bin))]; const h = 230, m = { t: 10, r: 16, b: 28, l: 44 }; const [svg, w] = svgIn(cC, h); const t = tip(cC);
      const x = d3.scalePoint(bins, [m.l, w - m.r]), y = d3.scaleLinear([0, 100], [h - m.b, m.t]);
      svg.append("g").attr("transform", `translate(0,${h - m.b})`).call(d3.axisBottom(x).tickSizeOuter(0)).call(axisStyle);
      svg.append("g").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(5).tickFormat(v => `${v}%`).tickSize(-(w - m.l - m.r))).call(axisStyle).call(g => g.select(".domain").remove());
      CUR.forEach((mo, i) => { const rs = LC.filter(r => r.model === mo).sort((a, b) => a.bin_i - b.bin_i);
        svg.append("path").datum(rs).attr("fill", "none").attr("stroke", ccol[i]).attr("stroke-width", 2).attr("stroke-linejoin", "round").attr("d", d3.line().x(r => x(r.bin)).y(r => y(r[f])));
        rs.forEach(r => svg.append("circle").attr("cx", x(r.bin)).attr("cy", y(r[f])).attr("r", 3).attr("fill", ccol[i])); });
      bins.forEach(bn => svg.append("rect").attr("x", x(bn) - 20).attr("width", 40).attr("y", m.t).attr("height", h - m.t - m.b).attr("fill", "transparent")
        .on("mousemove", ev => showTip(cC, t, ev, `${bn} words`, CUR.map((mo, i) => { const r = LC.find(z => z.model === mo && z.bin === bn); return [ccol[i], r ? r.label : mo, r ? `${pc(r[f])} (n ${r.n})` : "—"]; })))
        .on("mouseleave", () => { t.hidden = true; }));
    } else fail(cC, "length_curves");
    // adaptive
    const AL = ok("adaptive_len"), cA = document.getElementById("c-adapt");
    if (AL) forest(cA, AL.map(r => ({ ...r, family: r.family || "Base gpt-oss-120b", adapt_diff: 100 * (r.ratio - 1), adapt_lo: 100 * (r.lo - 1), adapt_hi: 100 * (r.hi - 1), strict_pct: null, samples: "" })), "adapt", { name: "length change when constrained (%)", color: dash.colors[0] });
    else fail(cA, "adaptive_len");
    // 2. output vs cot
    const OV = ok("out_vs_cot"), cS = document.getElementById("c-scatter");
    legend(document.getElementById("lg-scatter"), Object.entries(FC));
    if (OV) scatter(cS, OV, "cot", "out", { xlab: "CoT control (gated pass)", ylab: "Output control, CoT off (gated pass)", color: r => FC[r.family] || dash.colors[4], title: r => `${r.label} · ${r.family}`, label: r => r.label, h: 340 });
    else fail(cS, "out_vs_cot");
    table("t-ovc", OV, [["label"], ["family"], ["cot", 1, pc], ["out", 1, pc], ["med_cot", 1], ["med_out", 1], ["gap", 1, pts], ["out_length_part", 1, pts], ["out_control_part", 1, pts]], r => r.model);
    const OT = ok("out_vs_cot_type"), cT = document.getElementById("c-type");
    if (OT) scatter(cT, OT, "cot", "out", { xlab: "CoT control, this constraint", ylab: "Output control, this constraint", color: r => FC[r.family] || dash.colors[4], title: r => `${r.label} · ${r.constraint.replace(/_/g, " ")}`, r: 4.5, alpha: 0.8, h: 340 });
    else fail(cT, "out_vs_cot_type");
    // 3. other models
    const OM = ok("other_models"), cO = document.getElementById("c-or");
    legend(document.getElementById("lg-or"), [[dash.colors[1], "Output, reasoning off"], [dash.colors[3], "Output, model still reasoned"], [dash.colors[0], "CoT control"]]);
    const ORS = OM && OM.slice().sort((a, b) => (b.reasoning_off - a.reasoning_off) || (b.main - a.main));
    if (OM) hbars(cO, ORS, r => r.reasoning_off ? r.model : `${r.model} (reasoned)`,
      [{ key: "main", label: "Output", color: r => r.reasoning_off ? dash.colors[1] : dash.colors[3] }, { key: "cot", label: "CoT", color: dash.colors[0] }], { rowH: 34 });
    else fail(cO, "other_models");
    const cN = document.getElementById("c-orncri");
    if (OM) scatter(cN, OM.filter(r => r.reasoning_off && r.ncri != null), "ncri", "main", { xmax: 170, nodiag: true, xfmt: v => (+v).toFixed(0), xlab: "NCRI", ylab: "Output pass, reasoning off", color: () => dash.colors[1], title: r => r.model, label: r => r.model.split("/")[1], h: 320 });
    else fail(cN, "other_models");
    table("t-or", ORS, [["model"], ["status"], ["setting"], ["main", 1, pc], ["main_lo", 1, pc], ["main_hi", 1, pc], ["n_clean", 1], ["out_clean6", 1, pc], ["median_rtok", 1], ["n_filtered", 1], ["cot", 1, pc], ["cot_source"], ["ncri", 1, v => (+v).toFixed(1)], ["prev_run", 1, pc]], r => r.model);
    table("t-ortype", ok("other_models_type"), [["model"], ["constraint"], ["out", 1, pc]], r => `${encodeURIComponent(r.model)},${encodeURIComponent(r.constraint)}`);
    // 4. neel
    const NB = ok("neel_bins"), cB = document.getElementById("c-nbins");
    if (NB) hbars(cB, NB, r => `${r.bin} tokens`, [{ key: "pass_pct", label: "Pass", color: dash.colors[0] }], { max: 12, rowH: 30 });
    else fail(cB, "neel_bins");
    const NM = ok("neel_models"), cN = document.getElementById("c-nsplit");
    legend(document.getElementById("lg-nsplit"), [[dash.colors[0], "one model"]]);
    if (NM) {
      const h = 260, m = { t: 12, r: 16, b: 40, l: 48 }; const [svg, w] = svgIn(cN, h); const t = tip(cN);
      const ext = d3.extent(NM.flatMap(r => [r.length_part, r.control_part])); const lim = Math.max(Math.abs(ext[0]), Math.abs(ext[1])) + 1;
      const x = d3.scaleLinear([-lim, lim], [m.l, w - m.r]), y = d3.scaleLinear([-lim, lim], [h - m.b, m.t]);
      svg.append("g").attr("transform", `translate(0,${h - m.b})`).call(d3.axisBottom(x).ticks(5).tickFormat(v => `${v > 0 ? "+" : ""}${v}`).tickSize(-(h - m.t - m.b))).call(axisStyle);
      svg.append("g").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(5).tickFormat(v => `${v > 0 ? "+" : ""}${v}`).tickSize(-(w - m.l - m.r))).call(axisStyle).call(g => g.select(".domain").remove());
      svg.append("text").attr("x", (m.l + w - m.r) / 2).attr("y", h - 6).attr("text-anchor", "middle").attr("fill", "var(--color-fg-muted)").style("font-size", "var(--cds-font-size-caption)").text("Length part (points)");
      svg.append("text").attr("transform", `translate(12,${(m.t + h - m.b) / 2}) rotate(-90)`).attr("text-anchor", "middle").attr("fill", "var(--color-fg-muted)").style("font-size", "var(--cds-font-size-caption)").text("Control part (points)");
      NM.forEach(r => svg.append("circle").attr("cx", x(r.length_part)).attr("cy", y(r.control_part)).attr("r", 5).attr("fill", dash.colors[0]).attr("fill-opacity", .8).attr("stroke", "var(--color-panel)")
        .on("mousemove", ev => showTip(cN, t, ev, r.model, [[null, "raw pass", pc(r.raw)], [null, "length-matched", pc(r.matched)], [null, "length part", pts(r.length_part)], [null, "control part", pts(r.control_part)]])).on("mouseleave", () => { t.hidden = true; }));
    } else fail(cN, "neel_models");
    table("t-neel", NM && NM.slice().sort((a, b) => b.raw - a.raw), [["model"], ["median_tokens", 1], ["raw", 1, pc], ["matched", 1, pc], ["length_part", 1, pts], ["control_part", 1, pts], ["ncri", 1, v => (+v).toFixed(1)]], r => r.model);
  }
  dash.onData(draw);
})();
</script>
