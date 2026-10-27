"use strict";
// State follows received events immediately; animation never schedules inference state.
class TraceState {
  constructor(capacity = 64) {
    this.capacity = capacity; this.rows = new Map(); this.current = 0;
    this.total = 0; this.follow = true; this.finished = false;
  }
  add(event) {
    if (event.version !== 1 || !["draft", "verified"].includes(event.phase)) return;
    const row = this.rows.get(event.round) || {};
    row[event.phase] = event; this.rows.set(event.round, row);
    this.total = Math.max(this.total, event.round);
    while (this.rows.size > this.capacity) this.rows.delete(this.rows.keys().next().value);
    if (this.follow || !this.rows.has(this.current)) this.current = this.total;
  }
  load(history) {
    this.rows.clear(); this.follow = true;
    this.finished = true;
    for (const row of history?.rounds || []) {
      const number = row.draft?.round || row.verified?.round;
      if (number) this.rows.set(number, row);
    }
    while (this.rows.size > this.capacity) this.rows.delete(this.rows.keys().next().value);
    this.total = history?.round_count || 0; this.current = [...this.rows.keys()].at(-1) || 0;
  }
  move(direction) {
    this.follow = false;
    const keys = [...this.rows.keys()], index = keys.indexOf(this.current) + direction;
    this.current = keys[Math.max(0, Math.min(keys.length - 1, index))] || 0;
  }
  live() { this.follow = true; this.current = [...this.rows.keys()].at(-1) || 0; }
  finish() { this.finished = true; }
  get latest() { return [...this.rows.values()].at(-1); }
  get previousVerified() {
    return [...this.rows.values()].reverse().find((row) => row.verified && row !== this.latest);
  }
}
if (typeof module !== "undefined") module.exports = TraceState;
