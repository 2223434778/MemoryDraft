"use strict";
// Display only observed backend events. Playback timing never delays answer text.
window.ProcessView = (() => {
  const el = (id) => document.getElementById(id);
  const ns = "http://www.w3.org/2000/svg";
  let rows = new Map(), current = 0, follow = true, timer = null, shownAt = 0;
  let mode = "", model = "", finished = false, total = 0;
  const labels = {AR: "普通生成", DFlash: "投机解码", ASD: "ASD", "BudgetTree-ASD": "ASD + 成本树"};

  function open(value = true) {
    el("process-panel").hidden = !value;
    el("workspace").classList.toggle("process-open", value);
    el("process-toggle").classList.toggle("active", value);
    el("process-toggle").setAttribute("aria-expanded", String(value));
  }
  function reset(method = "", label = "") {
    clearTimeout(timer); timer = null; rows = new Map(); current = total = 0;
    follow = true; finished = false; mode = method; model = label;
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
    if (draft?.tree) treeView(draft.tree, verified);
    else if (draft) {
      strip(el("draft-tokens"), draft.proposal, (i) => !verified ? "pending" : i < verified.accepted ? "accepted" : "rejected", verified?.relaxed || []);
      el("correction-block").hidden = !verified?.correction?.length;
      strip(el("correction-tokens"), verified?.correction || [], "correction");
    }
    if (verified?.asd) el("asd-note").textContent = `ASD 本轮近似放行 ${verified.asd.relaxed} 个词元（虚线标记）。累计预算 ${Number(verified.asd.spent).toFixed(3)} / ${verified.asd.budget}。`;
    el("process-note").textContent = (draft?.tree ?
      "树来自本轮实际候选。起点已确认；绿色为采用路径，红色为未采用分支，不代表每个分支都逐一判错。蓝色为 Target 补充。" :
      "事件来自实际解码。黄色展示至少 180ms，可能为短暂回放；回答输出不等待动画。#编号表示无法单独显示的词元片段。") +
      (rows.size && rows.size < total ? ` 当前保留最近 ${rows.size} 轮。` : "");
  }
  function advance() {
    clearTimeout(timer); timer = null;
    if (!follow) return;
    const row = rows.get(current);
    if (row?.pending) {
      const remaining = Math.max(0, 180 - (performance.now() - shownAt));
      if (remaining) { timer = setTimeout(advance, remaining); return; }
      row.verified = row.pending; delete row.pending; row.verifiedAt = performance.now(); render();
      timer = setTimeout(advance, 160); return;
    }
    const next = [...rows.keys()].find((number) => number > current);
    if (next && (!row || row.verified)) {
      const remaining = row?.verifiedAt ? 160 - (performance.now() - row.verifiedAt) : 0;
      if (remaining > 0) { timer = setTimeout(advance, remaining); return; }
      // Keep playback bounded if a very fast backend gets ahead of the display.
      current = rows.size > 12 ? [...rows.keys()].at(-1) : next;
      shownAt = performance.now(); render(); timer = setTimeout(advance, 180);
    }
  }
  function add(event) {
    if (event.version !== 1) return;
    const row = rows.get(event.round) || {}; rows.set(event.round, row);
    total = Math.max(total, event.round);
    if (event.phase === "draft") row.draft = event;
    else row.pending = event;
    while (rows.size > 64) rows.delete(rows.keys().next().value);
    if (!current || !rows.has(current)) { current = event.round; shownAt = performance.now(); }
    if (follow) { render(); advance(); }
  }
  function load(history, method, label) {
    reset(method, label); finished = true;
    for (const row of history?.rounds || []) {
      const number = row.draft?.round || row.verified?.round;
      if (number) rows.set(number, row);
    }
    total = history?.round_count || 0; current = [...rows.keys()].at(-1) || 0; render(); open();
  }
  function move(direction) {
    follow = false; clearTimeout(timer);
    // A user explicitly browsing past rounds can see completed verification immediately.
    for (const row of rows.values()) if (row.pending) { row.verified = row.pending; delete row.pending; }
    const keys = [...rows.keys()], index = keys.indexOf(current) + direction;
    current = keys[Math.max(0, Math.min(keys.length - 1, index))] || 0; render();
  }
  el("process-toggle").onclick = () => open(el("process-panel").hidden);
  el("process-close").onclick = () => open(false);
  el("round-prev").onclick = () => move(-1); el("round-next").onclick = () => move(1);
  el("trace-live").onclick = () => { follow = true; current = [...rows.keys()].at(-1) || 0; shownAt = 0; render(); advance(); };
  return {open, reset, add, load, finish: () => { finished = true; advance(); render(); }};
})();
