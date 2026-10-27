const test = require("node:test");
const assert = require("node:assert/strict");
const createGlassEffects = require("../memorydraft/chat/static/glass.js");

function fixture() {
  const events = new Map(), windowEvents = new Map(), frames = new Map(), values = new Map(), classes = new Set();
  const media = [false, false, true].map(matches => ({matches, listeners: new Set(),
    addEventListener(_, callback) { this.listeners.add(callback); },
    removeEventListener(_, callback) { this.listeners.delete(callback); }}));
  const root = {hidden: false, addEventListener: (name, fn) => events.set(name, fn), removeEventListener: name => events.delete(name)};
  let count = 0, mediaIndex = 0;
  const view = {matchMedia: () => media[mediaIndex++],
    addEventListener: (name, fn) => windowEvents.set(name, fn), removeEventListener: name => windowEvents.delete(name),
    requestAnimationFrame: fn => { frames.set(++count, fn); return count; }, cancelAnimationFrame: id => frames.delete(id)};
  const target = {disabled: false, isConnected: true, contains: value => value === target,
    getBoundingClientRect: () => ({left: 10, top: 20, width: 100, height: 40}),
    classList: {add: v => classes.add(v), remove: v => classes.delete(v)},
    style: {setProperty: (key, value) => values.set(key, value), removeProperty: key => values.delete(key)}};
  const effects = createGlassEffects(root, view);
  function move(x, y, type = "mouse") { events.get("pointermove")({target: {closest: () => target}, clientX: x, clientY: y, pointerType: type}); }
  function flush() { const callbacks = [...frames.values()]; frames.clear(); callbacks.forEach(fn => fn()); }
  return {events, windowEvents, frames, values, classes, media, root, target, effects, move, flush};
}

test("reflection follows the latest pointer location and schedules no idle animation", () => {
  const f = fixture();
  f.move(20, 25); f.move(90, 55);
  assert.equal(f.frames.size, 1);
  f.flush();
  assert.equal(f.values.get("--glass-x"), "80.0px");
  assert.equal(f.values.get("--glass-y"), "35.0px");
  const angle = f.values.get("--glass-angle");
  assert(f.classes.has("glass-tracking")); assert.equal(f.frames.size, 0);
  f.move(12, 22); f.flush();
  assert.notEqual(f.values.get("--glass-angle"), angle);
  f.events.get("pointerout")({relatedTarget: null});
  assert.equal(f.values.size, 0); assert.equal(f.classes.size, 0);
});

test("reduced motion, transparency, coarse pointers and touch skip live reflection", () => {
  for (const index of [0, 1, 2]) {
    const f = fixture(); f.media[index].matches = index !== 2;
    f.move(30, 30); assert.equal(f.frames.size, 0); assert.equal(f.values.size, 0);
  }
  const f = fixture(); f.move(30, 30, "touch"); assert.equal(f.frames.size, 0);
  f.target.disabled = true; f.move(30, 30); assert.equal(f.frames.size, 0);
});

test("visibility and preference changes cancel pending frames; teardown removes listeners", () => {
  const f = fixture(); f.move(30, 30); f.root.hidden = true;
  f.events.get("visibilitychange")(); assert.equal(f.frames.size, 0);
  f.root.hidden = false; f.move(30, 30); f.flush();
  f.media[0].matches = true; for (const fn of f.media[0].listeners) fn();
  assert.equal(f.values.size, 0);
  f.effects.destroy(); assert.equal(f.events.size, 0); assert.equal(f.windowEvents.size, 0);
  assert(f.media.every(item => item.listeners.size === 0));
});
