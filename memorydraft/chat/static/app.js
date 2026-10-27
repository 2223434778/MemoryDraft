"use strict";
const $ = (id) => document.getElementById(id);
let csrf = "", selected = null, registerMode = false, controller = null, busy = false;
let catalog = [];
const deleting = new Set();
const modeLabels = {AR: "普通生成", DFlash: "投机解码", ASD: "ASD", "BudgetTree-ASD": "ASD + 成本树"};

async function api(path, options = {}) {
  const response = await fetch(path, {...options, credentials: "same-origin", headers: {
    "Content-Type": "application/json", "X-CSRF-Token": csrf, ...(options.headers || {})
  }});
  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    if (response.status === 401 && path !== "/api/login") showAuth();
    throw new Error(typeof data.detail === "string" ? data.detail : "请求失败，请检查输入或稍后重试");
  }
  return response;
}
const post = (path, body = {}) => api(path, {method: "POST", body: JSON.stringify(body)});
const notice = (text = "") => { $("notice").textContent = text; };

function showAuth() {
  $("auth").hidden = false;
  $("workspace").hidden = true;
  csrf = "";
  ProcessView.reset(); ProcessView.open(false);
}

function authMode(isRegister) {
  registerMode = isRegister;
  $("invite-label").hidden = !isRegister;
  $("invite").required = isRegister;
  $("password").autocomplete = isRegister ? "new-password" : "current-password";
  $("auth-submit").textContent = isRegister ? "创建账号" : "登录";
  $("login-tab").classList.toggle("selected", !isRegister);
  $("register-tab").classList.toggle("selected", isRegister);
  $("auth-error").textContent = "";
}

$("login-tab").onclick = () => authMode(false);
$("register-tab").onclick = () => authMode(true);
$("auth-form").onsubmit = async (event) => {
  event.preventDefault();
  $("auth-submit").disabled = true;
  $("auth-error").textContent = "";
  try {
    const body = {username: $("username").value.trim(), password: $("password").value};
    if (registerMode) body.invite_code = $("invite").value.trim();
    const result = await (await post(registerMode ? "/api/register" : "/api/login", body)).json();
    $("password").value = ""; $("invite").value = "";
    await enter(result);
  } catch (error) { $("auth-error").textContent = error.message; }
  finally { $("auth-submit").disabled = false; }
};

async function enter(result) {
  csrf = result.csrf;
  $("account-name").textContent = result.user.username;
  $("greeting").textContent = "你好，" + result.user.username;
  $("auth").hidden = true;
  $("workspace").hidden = false;
  selected = null;
  $("messages").replaceChildren();
  $("welcome").hidden = false;
  $("delete-chat").hidden = true;
  ProcessView.reset();
  await list();
  await updateModels();
}

async function updateModels() {
  try {
    const data = await (await api("/api/models")).json(); catalog = data.models;
    $("model-select").replaceChildren();
    for (const model of catalog) {
      const option = document.createElement("option"); option.value = model.id;
      option.textContent = model.label + (model.ready ? "" : " · 离线"); option.disabled = !model.ready;
      $("model-select").append(option);
    }
    $("model-select").value = catalog.find((m) => m.id === data.default_model && m.ready)?.id || catalog.find((m) => m.ready)?.id || "";
    $("method-select").value = data.default_method; selectModel();
  } catch { $("model-status").textContent = "连接状态未知，请刷新重试"; $("send").disabled = true; }
}
function selectModel() {
  const model = catalog.find((m) => m.id === $("model-select").value);
  for (const option of $("method-select").options) option.disabled = !model?.methods.includes(option.value);
  if (!model?.methods.includes($("method-select").value)) $("method-select").value = model?.methods[0] || "";
  const speculative = $("method-select").value !== "AR";
  $("visualize").disabled = busy || !speculative || !model?.visualization;
  $("send").disabled = !model?.ready || !$("method-select").value;
  $("model-status").textContent = model?.ready ? `${model.label} · 已就绪` : "模型暂时离线";
  $("model-status").classList.toggle("online", Boolean(model?.ready));
  $("capability-note").textContent = model?.methods.length === 1 && model.methods[0] === "AR" ?
    "此模型当前提供普通生成；投机与成本树需要匹配的 Draft 和验证实现。" :
    speculative && !model?.visualization ? "当前服务尚未提供过程事件。" : "";
}
$("model-select").onchange = selectModel; $("method-select").onchange = selectModel;

async function list() {
  const conversations = await (await api("/api/conversations")).json();
  $("history").replaceChildren();
  if (!conversations.length) {
    const empty = document.createElement("p"); empty.className = "empty-history";
    empty.textContent = "你的对话会出现在这里"; $("history").append(empty);
  }
  for (const conversation of conversations) {
    const row = document.createElement("div"); row.className = "history-row";
    row.classList.toggle("active", conversation.id === selected);
    const button = document.createElement("button"); button.type = "button";
    button.className = "history-item"; button.textContent = conversation.title;
    button.classList.toggle("active", conversation.id === selected); button.disabled = busy;
    button.onclick = () => openChat(conversation.id).catch((e) => notice(e.message));
    const remove = document.createElement("button"); remove.type = "button"; remove.className = "history-delete";
    remove.setAttribute("aria-label", "删除对话：" + conversation.title); remove.title = "删除对话";
    remove.append(Icons.make("trash")); remove.disabled = busy || deleting.has(conversation.id);
    remove.onclick = () => removeConversation(conversation.id, conversation.title);
    row.append(button, remove); $("history").append(row);
  }
}

async function openChat(id) {
  const data = await (await api("/api/conversations/" + id)).json();
  selected = id; ProcessView.reset();
  $("messages").replaceChildren(); $("welcome").hidden = data.messages.length > 0;
  $("delete-chat").hidden = false; $("sidebar").classList.remove("open"); notice();
  for (const message of data.messages) appendMessage(message.role, message.content, message.status, message.metadata);
  const latest = [...data.messages].reverse().find((m) => m.metadata?.speculation);
  if (latest && !$("process-panel").hidden) ProcessView.load(latest.metadata.speculation, latest.metadata.method, latest.metadata.model_label);
  await list(); scrollBottom();
}

function inline(parent, text) {
  const parts = text.split(/(`[^`]+`|\*\*[^*]+\*\*)/g);
  for (const part of parts) {
    if (part.startsWith("`") && part.endsWith("`")) {
      const node = document.createElement("code"); node.textContent = part.slice(1, -1); parent.append(node);
    } else if (part.startsWith("**") && part.endsWith("**")) {
      const node = document.createElement("strong"); node.textContent = part.slice(2, -2); parent.append(node);
    } else parent.append(document.createTextNode(part));
  }
}

function markdown(target, text) {
  target.replaceChildren();
  // Parse fenced blocks without interpreting model output as HTML.
  let position = 0;
  const expression = /```[^\n]*\n([\s\S]*?)(?:```|$)/g;
  function prose(value) {
    value = value.replace(/^\n+|\n+$/g, "").replace(/\n{3,}/g, "\n\n");
    if (!value) return;
    for (const line of value.split("\n")) {
      const heading = /^#{1,4}\s+/.exec(line);
      const node = document.createElement(heading ? "h2" : "div");
      inline(node, heading ? line.slice(heading[0].length) : line);
      if (!line) node.append(document.createElement("br")); target.append(node);
    }
  }
  for (const match of text.matchAll(expression)) {
    prose(text.slice(position, match.index));
    const pre = document.createElement("pre"), code = document.createElement("code");
    code.textContent = match[1];
    const button = document.createElement("button"); button.type = "button"; button.className = "copy-code"; button.textContent = "复制";
    button.onclick = async () => {
      try { await navigator.clipboard.writeText(code.textContent); button.textContent = "已复制"; }
      catch { button.textContent = "请选中复制"; }
    };
    pre.append(button, code); target.append(pre); position = match.index + match[0].length;
  }
  prose(text.slice(position));
}

function streamText(target, text) {
  const previous = target.textContent;
  markdown(target, text);
  const current = target.textContent;
  let shared = 0;
  while (shared < previous.length && shared < current.length && previous[shared] === current[shared]) shared++;
  const walker = document.createTreeWalker(target, NodeFilter.SHOW_TEXT), additions = [];
  let node, offset = 0;
  while ((node = walker.nextNode())) {
    const end = offset + node.length;
    if (end > shared && !node.parentElement.closest("button")) additions.push([node, Math.max(0, shared - offset)]);
    offset = end;
  }
  for (const [node, start] of additions) {
    const tail = start ? node.splitText(start) : node;
    const span = document.createElement("span"); span.className = "text-reveal";
    tail.replaceWith(span); span.append(tail);
  }
}

function appendMessage(role, text, status = "complete", metadata = {}) {
  const article = document.createElement("article"); article.className = "message " + role;
  const avatar = document.createElement("span"); avatar.className = "avatar"; avatar.append(Icons.make("brand"));
  const main = document.createElement("div"); main.className = "message-main";
  const name = document.createElement("p"); name.className = "message-name"; name.textContent = role === "user" ? "你" : "MemoryDraft";
  const body = document.createElement("div"); body.className = "message-body";
  if (role === "user") body.textContent = text; else markdown(body, text);
  const meta = document.createElement("p"); meta.className = "message-meta";
  const labels = {cancelled: "已停止", interrupted: "上次连接已中断", failed: "生成失败", generating: "正在生成…"};
  meta.textContent = labels[status] || (metadata.finish_reason === "length" ? "已达到本次回答长度上限" : "");
  const process = document.createElement("button"); process.type = "button";
  process.className = "message-process"; process.append(Icons.make("draft"), document.createTextNode("查看推理过程")); process.hidden = true;
  process.disabled = busy;
  const message = {body, meta, process, main, article};
  main.append(name, body, meta, process); article.append(...(role === "user" ? [main] : [avatar, main])); $("messages").append(article);
  decorateMessage(message, metadata);
  return message;
}
function decorateMessage(message, metadata) {
  if (metadata.model_label && !message.meta.textContent) message.meta.textContent = `${metadata.model_label} · ${modeLabels[metadata.method] || metadata.method}`;
  if (metadata.speculation?.rounds?.length) {
    message.process.hidden = false;
    message.process.onclick = () => ProcessView.load(metadata.speculation, metadata.method, metadata.model_label);
  }
}

function scrollBottom() { $("scroll-area").scrollTop = $("scroll-area").scrollHeight; }
function setBusy(value) {
  busy = value; $("send").hidden = value; $("stop").hidden = !value;
  $("new-chat").disabled = value; $("logout").disabled = value; $("delete-chat").disabled = value;
  $("model-select").disabled = value; $("method-select").disabled = value;
  selectModel();
  for (const button of $("history").querySelectorAll("button")) button.disabled = value;
  for (const button of document.querySelectorAll(".message-process")) button.disabled = value;
}

$("new-chat").onclick = () => {
  if (busy) return;
  selected = null; $("messages").replaceChildren(); $("welcome").hidden = false;
  ProcessView.reset(); $("delete-chat").hidden = true;
  $("sidebar").classList.remove("open"); notice(); $("prompt").focus(); list().catch((e) => notice(e.message));
};
function toggleSidebar() {
  if (matchMedia("(max-width:760px)").matches) $("sidebar").classList.toggle("open");
  else $("workspace").classList.toggle("sidebar-collapsed");
}
$("menu").onclick = toggleSidebar; $("sidebar-close").onclick = toggleSidebar;
$("logout").onclick = async () => { try { await post("/api/logout"); showAuth(); } catch (error) { notice(error.message); } };
$("delete-chat").onclick = async () => {
  if (selected) await removeConversation(selected);
};
async function removeConversation(id, title = "当前对话") {
  if (busy || deleting.has(id) || !confirm(`删除“${title}”及其中的全部消息？此操作无法撤销。`)) return;
  deleting.add(id);
  try {
    await api("/api/conversations/" + id, {method: "DELETE"});
    if (id === selected) $("new-chat").click();
    else await list();
  }
  catch (error) { notice(error.message); }
  finally { deleting.delete(id); }
}
for (const button of document.querySelectorAll("[data-prompt]")) button.onclick = () => { $("prompt").value = button.dataset.prompt; $("prompt").focus(); };
$("prompt").onkeydown = (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) { event.preventDefault(); $("chat-form").requestSubmit(); }
};
$("stop").onclick = () => {
  if (selected) post("/api/conversations/" + selected + "/cancel").catch(() => {});
  if (controller) controller.abort();
};

$("chat-form").onsubmit = async (event) => {
  event.preventDefault();
  const content = $("prompt").value.trim();
  if (!content || busy) return;
  const model = catalog.find((m) => m.id === $("model-select").value), method = $("method-select").value;
  if (!model?.ready || !model.methods.includes(method)) { notice("请选择可用的模型和推理模式"); return; }
  const visualize = $("visualize").checked && model.visualization && method !== "AR";
  ProcessView.reset(method, model.label);
  if (visualize && matchMedia("(min-width:1191px)").matches) ProcessView.open();
  setBusy(true); notice("正在提交…");
  let message = null, completed = false;
  try {
    if (!selected) {
      const conversation = await (await post("/api/conversations")).json(); selected = conversation.id;
      $("delete-chat").hidden = false;
    }
    $("welcome").hidden = true; $("prompt").value = "";
    appendMessage("user", content); message = appendMessage("assistant", "", "generating"); scrollBottom();
    message.article.classList.add("generating");
    if (visualize) ProcessView.attach(message.main);
    controller = new AbortController();
    const response = await api("/api/conversations/" + selected + "/messages", {method: "POST", body: JSON.stringify({content, model: model.id, method, visualize}), signal: controller.signal});
    const reader = response.body.getReader(), decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const chunk = await reader.read();
      if (chunk.done) break;
      buffer += decoder.decode(chunk.value, {stream: true});
      if (buffer.length > 1048576) throw new Error("响应过大，请重试");
      let boundary;
      while ((boundary = buffer.indexOf("\n\n")) !== -1) {
        const block = buffer.slice(0, boundary); buffer = buffer.slice(boundary + 2);
        const lines = block.split("\n"), name = lines.find((line) => line.startsWith("event: "))?.slice(7);
        const payload = lines.filter((line) => line.startsWith("data: ")).map((line) => line.slice(6)).join("\n");
        if (!payload) continue;
        const data = JSON.parse(payload);
        if (name === "waiting") notice(data.queued ? "前面有一个回答正在生成，请稍候…" : "模型正在思考…");
        if (name === "context" && data.dropped_turns) notice("对话较长，本次使用最近能放入模型的历史。");
        if (name === "text") { streamText(message.body, data.text); notice("正在生成…"); scrollBottom(); }
        if (name === "speculation") { ProcessView.add(data); scrollBottom(); }
        if (name === "error") throw new Error(data.message);
        if (name === "stopped") { completed = true; markdown(message.body, data.text); message.meta.textContent = "已停止"; notice("已停止生成"); }
        if (name === "complete") {
          completed = true; markdown(message.body, data.text);
          message.meta.textContent = data.metadata.finish_reason === "length" ? "已达到本次回答长度上限" : "";
          decorateMessage(message, data.metadata);
          notice(); scrollBottom();
        }
      }
    }
    if (!completed) throw new Error("连接已中断，已收到的回答会保留在历史中");
  } catch (error) {
    const stopped = error.name === "AbortError";
    notice(stopped ? "已停止生成" : error.message);
    if (message) message.meta.textContent = stopped ? "已停止" : "本次回答未完成";
  } finally {
    controller = null; ProcessView.finish(); message?.article.classList.remove("generating");
    setBusy(false); await list().catch(() => {}); $("prompt").focus();
  }
};

api("/api/me").then((r) => r.json()).then(enter).catch(showAuth);
