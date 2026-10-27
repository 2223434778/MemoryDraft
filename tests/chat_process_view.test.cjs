const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const TraceState = require("../memorydraft/chat/static/trace-state.js");

// A small DOM boundary double, exercising the production process renderer itself.
class Element {
  constructor(tag = "div") {
    this.tagName = tag.toUpperCase(); this.children = []; this.dataset = {}; this.hidden = false;
    this.attributes = {}; this.classes = new Set();
    this.classList = {toggle: (name, value) => value ? this.classes.add(name) : this.classes.delete(name)};
  }
  setAttribute(key, value) { this.attributes[key] = value; }
  append(...children) { for (const child of children) { child.remove(); child.parent = this; this.children.push(child); } }
  replaceChildren(...children) { for (const child of [...this.children]) child.remove(); this.append(...children); }
  remove() { if (this.parent) this.parent.children.splice(this.parent.children.indexOf(this), 1); this.parent = null; }
  get lastElementChild() { return this.children.at(-1); }
}

function renderer() {
  const nodes = new Map();
  const get = id => { if (!nodes.has(id)) nodes.set(id, new Element()); return nodes.get(id); };
  get("process-panel").hidden = true;
  const context = {window: {}, TraceState, document: {getElementById: get,
    createElement: tag => new Element(tag), createElementNS: (_, tag) => new Element(tag)}};
  vm.runInNewContext(fs.readFileSync(path.join(__dirname, "../memorydraft/chat/static/process.js"), "utf8"), context);
  return {view: context.window.ProcessView, get, body: new Element()};
}

const draft = {version:1, phase:"draft", round:1, proposal:[{id:9,text:"例"}],tree:null,
  inline:{anchor:0,candidate:"例"}};
const verified = {version:1, phase:"verified", round:1, accepted:1,committed:[{id:9,text:"例"}],
  correction:[],relaxed:[],inline:{start:0,accepted_end:1,end:1,accepted:"例",target:"",rejected:""}};

for (const inline of [false,true]) for (const details of [false,true]) {
  test(`inline=${inline} and details=${details} remain independent as events arrive`, () => {
    const {view,get,body} = renderer();
    view.open(details); view.attach(body,inline);
    view.add(draft); view.add(verified);
    assert.equal(get("process-panel").hidden,!details);
    assert.equal(body.children.length,Number(inline));
    assert.equal(view.ranges(1).length,Number(inline));
    get("process-toggle").onclick();
    assert.equal(get("process-panel").hidden,details);
    assert.equal(body.children.length,Number(inline));
    view.setInlineVisible(!inline);
    assert.equal(get("process-panel").hidden,details);
    assert.equal(body.children.length,Number(!inline));
    view.finish();
    assert.equal(body.children.length,0);
    assert.equal(get("process-panel").hidden,details);
  });
}

test("a hidden detail view retains live rounds and can be opened later", () => {
  const {view,get,body} = renderer();
  view.attach(body,false); view.add(draft); view.add(verified);
  assert.equal(body.children.length,0);
  assert.equal(get("process-panel").hidden,true);
  get("process-toggle").onclick();
  assert.equal(get("stat-round").textContent,1);
  assert.equal(get("stat-accepted").textContent,1);
  assert.equal(get("draft-tokens").children.length,1);
  assert.equal(body.children.length,0);
});
