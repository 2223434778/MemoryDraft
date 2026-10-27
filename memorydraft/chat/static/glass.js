"use strict";

function createGlassEffects(root, view) {
  const selector = ".pill-button,.new-chat,.icon-button,.select-trigger,.select-option,.suggestions button,.send,.stop,.primary,.message-action,.editor-button,.composer";
  const media = [view.matchMedia("(prefers-reduced-motion: reduce)"),
    view.matchMedia("(prefers-reduced-transparency: reduce)"), view.matchMedia("(hover: hover) and (pointer: fine)")];
  let active = null, point = null, frame = 0;
  const enabled = () => !media[0].matches && !media[1].matches && media[2].matches;

  function clear() {
    if (frame) view.cancelAnimationFrame(frame);
    frame = 0; point = null;
    if (active) {
      active.classList.remove("glass-tracking");
      for (const key of ["--glass-x", "--glass-y", "--glass-angle"]) active.style.removeProperty(key);
    }
    active = null;
  }
  function paint() {
    frame = 0;
    if (!enabled() || !active?.isConnected || !point) { clear(); return; }
    const rect = active.getBoundingClientRect();
    const x = Math.max(0, Math.min(rect.width, point.x - rect.left));
    const y = Math.max(0, Math.min(rect.height, point.y - rect.top));
    active.style.setProperty("--glass-x", x.toFixed(1) + "px");
    active.style.setProperty("--glass-y", y.toFixed(1) + "px");
    active.style.setProperty("--glass-angle", (Math.atan2(y - rect.height / 2, x - rect.width / 2) * 180 / Math.PI + 90).toFixed(1) + "deg");
    active.classList.add("glass-tracking");
  }
  function move(event) {
    if (!enabled() || event.pointerType === "touch") { clear(); return; }
    const target = event.target.closest?.(selector);
    if (!target || target.disabled) { clear(); return; }
    if (target !== active) { clear(); active = target; }
    point = {x: event.clientX, y: event.clientY};
    if (!frame) frame = view.requestAnimationFrame(paint);
  }
  function leave(event) { if (active && !active.contains(event.relatedTarget)) clear(); }
  function visibility() { if (root.hidden) clear(); }
  root.addEventListener("pointermove", move, {passive: true});
  root.addEventListener("pointerout", leave, {passive: true});
  root.addEventListener("visibilitychange", visibility);
  view.addEventListener("blur", clear);
  for (const query of media) query.addEventListener("change", clear);
  return {destroy() {
    clear(); root.removeEventListener("pointermove", move); root.removeEventListener("pointerout", leave);
    root.removeEventListener("visibilitychange", visibility); view.removeEventListener("blur", clear);
    for (const query of media) query.removeEventListener("change", clear);
  }};
}
if (typeof module !== "undefined") module.exports = createGlassEffects;
if (typeof window !== "undefined") window.GlassEffects = createGlassEffects(document, window);
