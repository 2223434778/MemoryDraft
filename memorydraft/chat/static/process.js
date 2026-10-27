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
    const y = [], gap = 35, width = 94; let leaf = 0;
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
      graph.append(svg("path", {d: `M ${tree.depths[parent] * width + 84} ${y[parent]} L ${tree.depths[i] * width + 10} ${y[i]}`,
        class: "tree-edge" + (selected.has(i) ? " selected" : "")}));
    });
    tree.tokens.forEach((token, i) => {
      const group = svg("g", {class: "tree-node " + (i ? state(i) : "root"), transform: `translate(${tree.depths[i] * width + 10},${y[i] - 12})`});
      group.append(svg("title", {}, `${i ? token.text : "已确认起点: " + token.text} · Token ${token.id}${relaxed.includes(i) ? " · ASD 近似放行" : ""}`));
      group.append(svg("rect", {width: 74, height: 24}));
      const label = Array.from(token.text);
      group.append(svg("text", {x: 37, y: 16, "text-anchor": "middle"}, label.length > 9 ? label.slice(0, 8).join("") + "…" : token.text));
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
  function attach(container) {
    const root = document.createElement("section"), header = document.createElement("div");
    root.className = "live-draft running"; root.setAttribute("aria-label", "实时 Draft 过程");
    header.className = "live-draft-header";
    const title = document.createElement("span"), status = document.createElement("span");
    title.append(Icons.make("draft"), document.createTextNode("Draft 实时过程")); status.className = "live-draft-status";
    header.append(title, status);
    const live = document.createElement("div"), previous = document.createElement("div"), note = document.createElement("p");
    live.className = "live-round"; previous.className = "live-previous"; note.className = "live-draft-note";
    note.textContent = "黄：草稿　绿：采用　红：拒绝 / 丢弃　蓝：Target 补充";
    root.append(header, live, previous, note); container.insertBefore(root, container.querySelector(".message-meta"));
    inline = {root, status, live, previous}; renderInline();
  }
  function paths(tree, verified) {
    const parents = new Set(tree.parents.slice(1)), result = [];
    tree.parents.forEach((_, i) => {
      if (parents.has(i)) return;
      const path = []; while (i > 0) { path.unshift(i); i = tree.parents[i]; }
      result.push(path);
    });
    const last = verified?.selected_path?.at(-1);
    return result.sort((a, b) => Number(b.includes(last)) - Number(a.includes(last)));
  }
  function inlineRound(target, row, previous = false) {
    const draft = row?.draft, verified = row?.verified;
    target.hidden = !draft;
    if (!draft) return;
    const key = `${draft.round}:${Boolean(verified)}:${trace.finished}`;
    if (target.dataset.key === key) return;
    target.dataset.key = key; target.replaceChildren();
    const title = document.createElement("p"); title.className = "live-round-label";
    title.textContent = `${previous ? "上一轮验证" : "当前草稿"} · 第 ${draft.round} 轮 · ` +
      (verified ? `采用 ${verified.accepted} 个词元` : trace.finished ? "未完成验证" : "Target 验证中");
    target.append(title);
    function tokens(items, state, relaxed = []) {
      const line = document.createElement("div"); line.className = "token-strip";
      strip(line, items, state, relaxed);
      if (verified) for (const node of line.children) node.classList.add("verified-transition");
      target.append(line);
    }
    if (!draft.tree) {
      tokens(draft.proposal, (i) => !verified ? "pending" : i < verified.accepted ? "accepted" : "rejected", verified?.relaxed || []);
    } else {
      const tree = draft.tree, selected = new Set(verified?.selected_path || []);
      const shown = paths(tree, verified).slice(0, previous ? 1 : 3);
      shown.forEach((path, index) => {
        const label = document.createElement("small"); label.textContent = `候选 ${index + 1}`; target.append(label);
        tokens(path.map((i) => tree.tokens[i]), (i) => !verified ? "pending" : selected.has(path[i]) ? "accepted" : "rejected",
          path.map((i, j) => verified?.relaxed?.includes(i) ? j : -1));
      });
    }
    if (verified?.correction?.length) tokens(verified.correction, "correction");
  }
  function renderInline() {
    if (!inline) return;
    inline.root.classList.toggle("running", !trace.finished);
    inline.root.dataset.round = trace.total;
    inline.root.dataset.phase = !trace.latest ? "waiting" : trace.latest.verified ? "verified" : "draft";
    inline.status.textContent = trace.finished ? "已结束" : "实时";
    if (!trace.latest) {
      inline.live.hidden = false; inline.live.textContent = trace.finished ? "本次没有草稿事件" : "等待小模型起草…";
      inline.previous.hidden = true; return;
    }
    inlineRound(inline.live, trace.latest); inlineRound(inline.previous, trace.previousVerified, true);
  }
  el("process-toggle").onclick = () => open(el("process-panel").hidden);
  el("process-close").onclick = () => open(false);
  el("round-prev").onclick = () => move(-1); el("round-next").onclick = () => move(1);
  el("trace-live").onclick = () => { trace.live(); render(); };
  return {open, reset, add, load, attach, finish: () => { trace.finish(); renderInline(); render(); }};
})();
