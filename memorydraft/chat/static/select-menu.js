"use strict";
window.SelectMenu = (() => {
  const menus = new Map();
  function close(menu, focus = false) {
    menu.wrap.classList.remove("select-open"); menu.button.setAttribute("aria-expanded", "false");
    menu.list.setAttribute("aria-hidden", "true"); menu.list.inert = true;
    if (focus) menu.button.focus();
  }
  function open(menu, edge = null) {
    if (menu.native.disabled) return;
    for (const other of menus.values()) close(other);
    const rect = menu.button.getBoundingClientRect(), width = Math.min(320, innerWidth - 24);
    menu.list.style.width = width + "px";
    menu.list.style.left = Math.max(12, Math.min(rect.left, innerWidth - width - 12)) + "px";
    menu.list.style.bottom = (innerHeight - rect.top + 8) + "px";
    menu.wrap.classList.add("select-open"); menu.button.setAttribute("aria-expanded", "true");
    menu.list.setAttribute("aria-hidden", "false"); menu.list.inert = false;
    const options = [...menu.list.querySelectorAll("button:not(:disabled)")];
    (edge === "last" ? options.at(-1) : options.find(o => o.dataset.value === menu.native.value) || options[0])?.focus();
  }
  function refresh(id) {
    const menu = menus.get(id); if (!menu) return;
    const chosen = menu.native.selectedOptions[0];
    menu.label.textContent = chosen?.dataset.shortLabel || chosen?.dataset.label || chosen?.textContent || "暂不可用";
    menu.button.title = chosen?.textContent || "暂不可用"; menu.button.disabled = menu.native.disabled;
    menu.list.replaceChildren();
    for (const option of menu.native.options) {
      const item = document.createElement("button"); item.type = "button"; item.className = "select-option";
      item.dataset.value = option.value; item.setAttribute("role", "option");
      item.setAttribute("aria-selected", String(option.selected)); item.disabled = option.disabled;
      const text = document.createElement("span"); text.textContent = option.textContent;
      item.append(text, Icons.make("check"));
      item.onclick = () => {
        menu.native.value = option.value; close(menu, true);
        menu.native.dispatchEvent(new Event("change", {bubbles: true}));
      };
      menu.list.append(item);
    }
    if (menu.native.disabled) close(menu);
  }
  for (const id of ["model-select", "method-select"]) {
    const native = document.getElementById(id), wrap = native.parentElement;
    const button = document.createElement("button"), label = document.createElement("span"), list = document.createElement("div");
    button.type = "button"; button.id = id + "-button"; button.className = "select-trigger";
    button.setAttribute("role", "combobox"); button.setAttribute("aria-label", native.getAttribute("aria-label"));
    button.setAttribute("aria-haspopup", "listbox"); button.setAttribute("aria-controls", id + "-menu");
    button.setAttribute("aria-expanded", "false"); label.className = "select-label";
    button.append(label, Icons.make("chevron"));
    list.id = id + "-menu"; list.className = "select-menu"; list.setAttribute("role", "listbox");
    list.setAttribute("aria-label", native.getAttribute("aria-label") + "选项");
    list.setAttribute("aria-hidden", "true"); list.inert = true;
    native.hidden = true; wrap.append(button, list);
    const menu = {native, wrap, button, label, list}; menus.set(id, menu);
    button.onclick = () => wrap.classList.contains("select-open") ? close(menu) : open(menu);
    button.onkeydown = event => {
      if (["ArrowDown", "ArrowUp"].includes(event.key)) { event.preventDefault(); open(menu, event.key === "ArrowUp" ? "last" : null); }
      else if (event.key === "Escape") close(menu, true);
      else if (event.key === "Tab") close(menu);
    };
    list.onkeydown = event => {
      const options = [...list.querySelectorAll("button:not(:disabled)")], index = options.indexOf(document.activeElement);
      if (event.key === "Escape") { event.preventDefault(); close(menu, true); }
      else if (event.key === "Tab") close(menu);
      else if (["ArrowDown", "ArrowUp", "Home", "End"].includes(event.key)) {
        event.preventDefault();
        const next = event.key === "Home" ? 0 : event.key === "End" ? options.length - 1 :
          (index + (event.key === "ArrowDown" ? 1 : -1) + options.length) % options.length;
        options[next]?.focus();
      }
    };
    refresh(id);
  }
  document.addEventListener("pointerdown", event => {
    for (const menu of menus.values()) if (!menu.wrap.contains(event.target)) close(menu);
  });
  window.addEventListener("resize", () => { for (const menu of menus.values()) close(menu); });
  return {refresh, refreshAll: () => { for (const id of menus.keys()) refresh(id); }};
})();
