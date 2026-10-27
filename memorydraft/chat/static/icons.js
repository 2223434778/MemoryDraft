"use strict";
// Original line artwork; no third-party logo or icon assets.
window.Icons = {
  paths: {
    chevron: ["m6 9 6 6 6-6"],
    check: ["m5 12 4 4L19 5"],
    brand: ["M12 18V9m0 4-4-4m4 7 4-4", "M12 9C8 7 9 3 12 2c3 1 4 5 0 7Z", "M8 9C5 10 3 8 3 5c3-1 6 0 5 4Zm8 3c-1-3 2-5 5-4 0 3-2 5-5 4Z", "M3 20c3-3 6 3 9 0s6 3 9 0"],
    menu: ["M4 8h16M4 16h11"],
    add: ["M12 5v14M5 12h14"],
    draft: ["m5 17-1 4 4-1L20 8l-4-4L5 17Z", "m13 7 4 4"],
    code: ["m8 6-5 6 5 6m8-12 5 6-5 6m-3-14-2 16"],
    plan: ["M10 5h10M10 12h10M10 19h10m-16-14 1 1 2-2m-3 8 1 1 2-2m-3 8 1 1 2-2"],
    send: ["M12 20V4m-6 6 6-6 6 6"],
    stop: ["M6 6h12v12H6Z"],
    close: ["m6 6 12 12M18 6 6 18"],
    left: ["M20 12H4m5-5-5 5 5 5"],
    right: ["M4 12h16m-5-5 5 5-5 5"],
    trash: ["M4 6h16M9 6V3h6v3M6 6l1 15h10l1-15M10 10v7m4-7v7"]
  },
  make(name) {
    const ns = "http://www.w3.org/2000/svg", icon = document.createElementNS(ns, "svg");
    for (const [key, value] of Object.entries({viewBox: "0 0 24 24", fill: "none", stroke: "currentColor",
      "stroke-width": "1.65", "stroke-linecap": "round", "stroke-linejoin": "round", "aria-hidden": "true", class: "ui-icon"})) icon.setAttribute(key, value);
    for (const d of this.paths[name] || this.paths.brand) {
      const path = document.createElementNS(ns, "path"); path.setAttribute("d", d); icon.append(path);
    }
    return icon;
  }
};
for (const node of document.querySelectorAll("[data-icon]")) node.replaceChildren(Icons.make(node.dataset.icon));
