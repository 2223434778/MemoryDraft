const test = require("node:test");
const assert = require("node:assert/strict");
const TraceState = require("../memorydraft/chat/static/trace-state.js");
const draft = (round) => ({version: 1, phase: "draft", round, proposal: [{id: 9, text: "例"}], tree: null});
const verified = (round) => ({version: 1, phase: "verified", round, accepted: 1, committed: [{id: 9, text: "例"}]});

test("draft and verification become visible immediately in received order", () => {
  const state = new TraceState();
  state.add(draft(1)); assert.equal(state.current, 1); assert.equal(state.latest.verified, undefined);
  state.add(verified(1)); assert.equal(state.latest.verified.accepted, 1);
  state.add(draft(2)); assert.equal(state.current, 2); assert.equal(state.latest.draft.round, 2);
  assert.equal(state.previousVerified.verified.round, 1);
});

test("bursts cannot create a playback backlog and history stays bounded", () => {
  const state = new TraceState();
  for (let i = 1; i <= 1000; i++) { state.add(draft(i)); state.add(verified(i)); }
  assert.equal(state.current, 1000); assert.equal(state.total, 1000);
  assert.equal(state.rows.size, 64); assert.equal(state.latest.verified.round, 1000);
  state.finish(); assert.equal(state.current, 1000); assert.equal(state.finished, true);
});

test("browsing history does not stop the inline live state from updating", () => {
  const state = new TraceState();
  for (let i = 1; i <= 3; i++) { state.add(draft(i)); state.add(verified(i)); }
  state.move(-1); assert.equal(state.current, 2);
  state.add(draft(4)); state.add(verified(4));
  assert.equal(state.current, 2); assert.equal(state.latest.verified.round, 4);
  state.live(); assert.equal(state.current, 4); assert.equal(state.follow, true);
});

test("persisted truncated histories support previous and next without invented rounds", () => {
  const state = new TraceState(); state.add(draft(1));
  state.load({round_count: 36, rounds: [35, 36].map((i) => ({draft: draft(i), verified: verified(i)}))});
  assert.equal(state.rows.has(1), false); assert.equal(state.current, 36);
  state.move(-1); assert.equal(state.current, 35);
  state.move(-1); assert.equal(state.current, 35);
  state.move(1); assert.equal(state.current, 36); assert.equal(state.total, 36);
});

test("unknown phases never change the displayed inference state", () => {
  const state = new TraceState();
  state.add({...draft(100), version: 2}); state.add({...draft(100), phase: "guessed"});
  assert.equal(state.total, 0); assert.equal(state.rows.size, 0);
});
