"use strict";
// Render real events immediately, without a timed playback queue.
window.ProcessView = (() => {
  const el = (id) => document.getElementById(id);
  const ns = "http://www.w3.org/2000/svg";
  let trace = new TraceState(), inline = null;
  let mode = "", model = "";
  const labels = {AR: "普通生成", DFlash: "投机解码", ASD: "ASD", "BudgetTree-ASD": "ASD + 成本树"};

  function open(value = true) {
    el("process-panel").hidden = !value;
    el("workspace").classList.toggle("process-open", value);
    el("process-toggle").classList.toggle("active", value);
    el("process-toggle").setAttribute("aria-expanded", String(value));
    if (value) render();
  }
  function reset(method = "", label = "") {
    trace = new TraceState(); inline = null; mode = method; model = label;
    render();
  }
  function chip(token, state, relaxed = false) {
    const node = document.createElement("span");
    node.className = "token-chip " + state + (relaxed ? " relaxed" : "");
    node.textContent = token.text;
    node.title = `Token ${token.id}` + (relaxed ? " · ASD 近似放行" : "");
    return node;
  }
  function strip(target, tokens, state, relaxed = []) {
    target.replaceChildren();
    tokens.forEach((token, i) => target.append(chip(token, typeof state === "function" ? state(i) : state, relaxed.includes(i))));
  }
  function svg(tag, attrs, text) {
    const node = document.createElementNS(ns, tag);
    for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, String(value));
    if (text !== undefined) node.textContent = text;
    return node;
  }
  function treeView(tree, verified) {
    const selected = new Set(verified?.selected_path || []), relaxed = verified?.relaxed || [];
    const state = (i) => !verified ? "pending" : selected.has(i) ? "accepted" : "rejected";
    const children = tree.parents.map(() => []);
    tree.parents.forEach((p, i) => { if (i) children[p].push(i); });
    const leaves = children.map((nodes, i) => nodes.length ? -1 : i).filter((i) => i >= 0);
    const y = [], gap = 35, width = 112; let leaf = 0;
    function place(i) {
      children[i].sort((a, b) => Number(selected.has(b)) - Number(selected.has(a)));
      // Place the adopted path at the top so it is visible without scrolling.
      y[i] = children[i].length ? Math.min(...children[i].map(place)) : 24 + leaf++ * gap;
      return y[i];
    }
    place(0);
    const graph = svg("svg", {width: (Math.max(...tree.depths) + 1) * width + 20,
      height: Math.max(70, leaves.length * gap + 20), role: "img", "aria-label": "实际候选树，绿色为采用路径，红色为未采用分支"});
    tree.parents.forEach((parent, i) => {
      if (!i) return;
      graph.append(svg("path", {d: `M ${tree.depths[parent] * width + 104} ${y[parent]} L ${tree.depths[i] * width + 10} ${y[i]}`,
        class: "tree-edge" + (selected.has(i) ? " selected" : "")}));
    });
    tree.tokens.forEach((token, i) => {
      const group = svg("g", {class: "tree-node " + (i ? state(i) : "root"), transform: `translate(${tree.depths[i] * width + 10},${y[i] - 12})`});
      group.append(svg("title", {}, `${i ? token.text : "已确认起点: " + token.text} · Token ${token.id}${relaxed.includes(i) ? " · ASD 近似放行" : ""}`));
      group.append(svg("rect", {width: 94, height: 24}));
      const label = Array.from(token.text);
      group.append(svg("text", {x: 47, y: 16, "text-anchor": "middle"}, label.length > 7 ? label.slice(0, 6).join("") + "…" : token.text));
      graph.append(group);
    });
    el("tree-canvas").replaceChildren(graph);
    el("tree-canvas").scrollTop = 0;
    el("tree-size").textContent = `${tree.tokens.length} 节点 · ${leaves.length} 条路径`;
    const candidates = el("tree-candidates"); candidates.replaceChildren();
    // Put the selected branch first when there are more paths than fit on screen.
    const last = verified?.selected_path?.at(-1);
    function path(i) { const result = []; while (i > 0) { result.unshift(i); i = tree.parents[i]; } return result; }
    const paths = leaves.map(path).sort((a, b) => Number(b.includes(last)) - Number(a.includes(last)));
    paths.slice(0, 6).forEach((nodes, index) => {
      const box = document.createElement("div"), title = document.createElement("div"), tokens = document.createElement("div");
      box.className = "candidate-path"; title.className = "candidate-label"; tokens.className = "token-strip";
      title.textContent = `候选 ${index + 1}` + (verified && nodes.includes(last) ? " · 包含采用路径" : "");
      nodes.forEach((i) => tokens.append(chip(tree.tokens[i], state(i), relaxed.includes(i))));
      box.append(title, tokens); candidates.append(box);
    });
    if (paths.length > 6) {
      const note = document.createElement("p"); note.className = "candidate-label";
      note.textContent = `展示其中 6 条，全部 ${paths.length} 条可在上方树图查看。`; candidates.append(note);
    }
    el("selected-path-block").hidden = !verified;
    if (verified) strip(el("selected-path"), verified.committed || [], (i) => i < verified.accepted ? "accepted" : "correction",
      (verified.selected_path || []).slice(1).map((n, i) => relaxed.includes(n) ? i : -1));
  }
  function render() {
    const {rows, current, total, finished, follow} = trace;
    const row = rows.get(current), draft = row?.draft, verified = row?.verified;
    el("process-mode").textContent = mode ? `${model} · ${labels[mode] || mode}` : "选择投机模式，开始一段对话";
    el("stat-round").textContent = current || "—";
    el("stat-accepted").textContent = verified ? verified.accepted : "—";
    el("stat-proposed").textContent = draft ? (draft.tree ? draft.tree.tokens.length - 1 : draft.proposal.length) : "—";
    el("process-state").textContent = !draft ? (mode === "AR" ? "逐词生成，无草稿" : "等待起草") :
      `第 ${current} / ${total} 轮 · ${verified ? "已验证" : finished ? "未完成验证" : "待验证"}`;
    el("trace-live").textContent = follow ? (finished ? "最近轮次" : "跟随实时") : "回到最新";
    const keys = [...rows.keys()];
    el("round-prev").disabled = keys.indexOf(current) <= 0;
    el("round-next").disabled = keys.indexOf(current) >= keys.length - 1;
    el("process-empty").hidden = Boolean(draft);
    el("linear-section").hidden = !draft || Boolean(draft.tree);
    el("tree-section").hidden = !draft?.tree;
    el("asd-note").hidden = !verified?.asd;
    if (draft?.tree && !el("process-panel").hidden) treeView(draft.tree, verified);
    else if (draft && !draft.tree && !el("process-panel").hidden) {
      strip(el("draft-tokens"), draft.proposal, (i) => !verified ? "pending" : i < verified.accepted ? "accepted" : "rejected", verified?.relaxed || []);
      el("correction-block").hidden = !verified?.correction?.length;
      strip(el("correction-tokens"), verified?.correction || [], "correction");
    }
    if (verified?.asd) el("asd-note").textContent = `ASD 本轮近似放行 ${verified.asd.relaxed} 个词元（虚线标记）。累计预算 ${Number(verified.asd.spent).toFixed(3)} / ${verified.asd.budget}。`;
    el("process-note").textContent = (draft?.tree ?
      "树来自本轮实际候选。起点已确认；绿色为采用路径，红色为未采用分支，不代表每个分支都逐一判错。蓝色为 Target 补充。" :
      "事件到达即更新，当前状态跟随最新轮次。动画只平滑颜色，不排队回放。#编号表示无法单独显示的词元片段。") +
      (rows.size && rows.size < total ? ` 当前保留最近 ${rows.size} 轮。` : "");
  }
  function add(event) {
    trace.add(event); renderInline(); render();
  }
  function load(history, method, label) {
    reset(method, label); trace.load(history); open();
  }
  function move(direction) {
    trace.move(direction); render();
  }
  function attach(body) {
    const root = document.createElement("span"); root.className = "inline-speculation";
    root.setAttribute("aria-hidden", "true"); inline = {body, root, text: ""}; renderInline();
  }
  function detach(body) { if (inline?.body === body) inline.root.remove(); }
  function syncBody(body, text) {
    if (inline?.body !== body) return;
    inline.text = text; renderInline();
  }
  function ranges(length) {
    const verified = trace.latest?.verified || trace.previousVerified?.verified, value = verified?.inline;
    if (!inline || !value || ![value.start, value.accepted_end, value.end].every(Number.isInteger)
        || value.start < 0 || value.start > value.accepted_end || value.accepted_end > value.end || value.end > length) return [];
    return [{start:value.start,end:value.accepted_end,kind:"accepted"},
      {start:value.accepted_end,end:value.end,kind:"correction"}].filter(r=>r.end>r.start);
  }
  function renderInline() {
    if (!inline || trace.finished) return;
    const {root, body, text} = inline, row = trace.latest;
    root.remove(); root.replaceChildren(); root.dataset.round = trace.total;
    root.dataset.phase = !row ? "waiting" : row.verified ? "verified" : "draft";
    const status = document.createElement("span"); status.className = "spec-status";
    status.textContent = !row ? "起草中…" : row.verified ? `采用 ${row.verified.accepted}` : "草稿";
    root.append(status);
    function piece(value, kind, limit) {
      if (!value) return;
      const span = document.createElement("span"); span.className = "spec-preview " + kind;
      const compact = value.replace(/\s+/g, " ");
      const characters = Array.from(compact);
      span.textContent = characters.slice(0, limit).join("") + (characters.length > limit ? "…" : "");
      span.title = kind === "rejected" ? "拒绝 / 未采用的候选片段" : "临时 Draft 预览";
      root.append(span);
    }
    if (row?.verified) {
      const value = row.verified.inline;
      if (value && text.length < value.end) {
        piece(value.accepted, "accepted", 18); piece(value.target, "correction", 8);
      }
      piece(value?.rejected, "rejected", 10);
    } else piece(row?.draft?.inline?.candidate, "pending", 18);
    let parent = body.lastElementChild || body;
    if (parent.tagName === "PRE") parent = parent.querySelector("code");
    parent.append(root);
  }
  el("process-toggle").onclick = () => open(el("process-panel").hidden);
  el("process-close").onclick = () => open(false);
  el("round-prev").onclick = () => move(-1); el("round-next").onclick = () => move(1);
  el("trace-live").onclick = () => { trace.live(); render(); };
  return {open, reset, add, load, attach, detach, syncBody, ranges, finish: () => { trace.finish(); inline?.root.remove(); inline = null; render(); }};
})();
