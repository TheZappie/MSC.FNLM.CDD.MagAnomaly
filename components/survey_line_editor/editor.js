const svg = document.querySelector("#map");
const background = document.querySelector("#background");
const linesGroup = document.querySelector("#lines");
const handlesGroup = document.querySelector("#handles");
const rotationsGroup = document.querySelector("#rotations");
const reset = document.querySelector("#reset");

let config;
let endpoints;
let drag;
let lastHeight;
const lines = [];
const lineTargets = [];
const handles = [];
const rotations = [];

function send(type, payload = {}) {
  window.parent.postMessage({ isStreamlitMessage: true, type, ...payload }, "*");
}

function commit() {
  send("streamlit:setComponentValue", { value: endpoints, dataType: "json" });
}

function setAttributes(element, attributes) {
  for (const [name, value] of Object.entries(attributes)) {
    element.setAttribute(name, value);
  }
}

function svgElement(tag, parent, attributes) {
  const element = document.createElementNS(svg.namespaceURI, tag);
  setAttributes(element, attributes);
  parent.append(element);
  return element;
}

function toSvg([x, y]) {
  const [left, top, width, height] = config.plot;
  const [min, max] = config.limits;
  return [left + width * (x - min) / (max - min),
          top + height * (max - y) / (max - min)];
}

function clamp(value) {
  const [min, max] = config.limits;
  return Math.max(min, Math.min(max, value));
}

function toMap(x, y) {
  const [left, top, width, height] = config.plot;
  const [min, max] = config.limits;
  return [min + (x - left) / width * (max - min),
          max - (y - top) / height * (max - min)];
}

function pointerPosition(event) {
  const { x, y } = new DOMPoint(event.clientX, event.clientY)
    .matrixTransform(svg.getScreenCTM().inverse());
  return toMap(x, y);
}

function midpoint([start, end]) {
  return start.map((value, i) => (value + end[i]) / 2);
}

function draw() {
  endpoints.forEach((pair, i) => {
    const visible = i < config.visible_line_count;
    for (const control of [lines[i], lineTargets[i], ...handles[i]]) {
      control.setAttribute("display", visible ? "inline" : "none");
    }
    const [[x1, y1], [x2, y2]] = pair.map(toSvg);
    setAttributes(lines[i], { x1, y1, x2, y2 });
    setAttributes(lineTargets[i], { x1, y1, x2, y2 });
    setAttributes(handles[i][0], { cx: x1, cy: y1 });
    setAttributes(handles[i][1], { cx: x2, cy: y2 });
    const dx = x2 - x1;
    const dy = y2 - y1;
    const length = Math.hypot(dx, dy);
    rotations[i].setAttribute("display", visible && length > 0 ? "inline" : "none");
    if (length > 0) {
      const offset = (i === 0 ? -32 : 32) / length;
      const x = (x1 + x2) / 2 - dy * offset;
      const y = (y1 + y2) / 2 + dx * offset;
      rotations[i].setAttribute("transform", `translate(${x} ${y})`);
    }
  });
}

function moveLine(pair, delta) {
  const [min, max] = config.limits;
  const [start, end] = pair;
  // Clamp the shared displacement, not the endpoints, to preserve the length.
  const shift = delta.map((d, i) =>
    Math.max(min - Math.min(start[i], end[i]), Math.min(max - Math.max(start[i], end[i]), d))
  );
  return pair.map(point => point.map((value, i) => clamp(value + shift[i])));
}

function rotateLine(pair, angle, fallback) {
  const [x, y] = midpoint(pair);
  const sin = Math.sin(angle), cos = Math.cos(angle);
  const next = pair.map(([px, py]) => [
    x + (px - x) * cos - (py - y) * sin,
    y + (px - x) * sin + (py - y) * cos,
  ]);
  const [min, max] = config.limits;
  // Ignore out-of-bounds poses rather than moving the pivot or shortening the line.
  if (next.some(point => point.some(value => value < min - 1e-9 || value > max + 1e-9))) {
    return fallback;
  }
  return next.map(point => point.map(clamp));
}

function moveEndpoint(pair, endpointIndex, delta) {
  return pair.map((point, i) => i === endpointIndex
    ? point.map((value, axis) => clamp(value + delta[axis])) : point
  );
}

function sameLine(a, b) {
  return a.every((point, i) => point.every((value, axis) => Math.abs(value - b[i][axis]) < 1e-9));
}

function bindControl(control, lineIndex, mode, endpointIndex) {
  control.addEventListener("pointerdown", (event) => {
    if (drag || event.button !== 0) return;
    event.preventDefault();
    control.focus({ preventScroll: true });
    drag = {
      id: event.pointerId,
      lineIndex,
      endpointIndex,
      mode,
      pair: endpoints[lineIndex],
      pointer: pointerPosition(event),
    };
    svg.setPointerCapture(event.pointerId);
  });

  const directions = {
    ArrowLeft: [-1, 0], ArrowRight: [1, 0],
    ArrowUp: [0, 1], ArrowDown: [0, -1],
  };
  control.addEventListener("keydown", (event) => {
    if (!Object.hasOwn(directions, event.key) || drag) return;
    if (mode === "rotate" && !["ArrowLeft", "ArrowRight"].includes(event.key)) return;
    event.preventDefault();
    const pair = endpoints[lineIndex];
    const direction = directions[event.key];
    const delta = direction.map(value => value * (event.shiftKey ? 1 : 0.1));
    const next = mode === "rotate"
      ? rotateLine(pair, -direction[0] * Math.PI / (event.shiftKey ? 12 : 180), pair)
      : mode === "move" ? moveLine(pair, delta) : moveEndpoint(pair, endpointIndex, delta);
    if (sameLine(pair, next)) return;
    endpoints[lineIndex] = next;
    draw();
    commit();
  });
}

window.addEventListener("pointermove", (event) => {
  if (drag?.id !== event.pointerId) return;
  const pointer = pointerPosition(event);
  const { mode, pair, lineIndex, endpointIndex } = drag;
  const delta = pointer.map((value, i) => value - drag.pointer[i]);
  if (mode === "rotate") {
    const [x, y] = midpoint(pair);
    if (Math.hypot(pointer[0] - x, pointer[1] - y) < 1e-9) return;
    const angle = ([px, py]) => Math.atan2(py - y, px - x);
    endpoints[lineIndex] = rotateLine(pair, angle(pointer) - angle(drag.pointer), endpoints[lineIndex]);
  } else {
    endpoints[lineIndex] = mode === "move"
      ? moveLine(pair, delta) : moveEndpoint(pair, endpointIndex, delta);
  }
  draw();
});

window.addEventListener("pointerup", (event) => {
  if (drag?.id !== event.pointerId) return;
  const { pair, lineIndex } = drag;
  drag = null;
  if (!sameLine(pair, endpoints[lineIndex])) commit();
});

function cancelDrag(event) {
  if (drag?.id !== event.pointerId) return;
  endpoints[drag.lineIndex] = drag.pair;
  drag = null;
  draw();
}
window.addEventListener("pointercancel", cancelDrag);
svg.addEventListener("lostpointercapture", cancelDrag);

reset.addEventListener("click", () => {
  if (drag) return;
  for (let i = 0; i < config.visible_line_count; i++) {
    endpoints[i] = structuredClone(config.defaults[i]);
  }
  draw();
  commit();
});

function updateHeight() {
  const height = Math.ceil(document.querySelector("main").getBoundingClientRect().height);
  if (height === lastHeight) return;
  lastHeight = height;
  send("streamlit:setFrameHeight", { height });
}

window.addEventListener("message", (event) => {
  if (event.source !== window.parent || event.data?.type !== "streamlit:render") return;
  config = event.data.args;
  if (drag && drag.lineIndex >= config.visible_line_count) {
    svg.releasePointerCapture(drag.id);
    cancelDrag({ pointerId: drag.id });
  }
  const [width, height] = config.size;
  setAttributes(svg, { viewBox: `0 0 ${width} ${height}`, width, height });
  if (background.getAttribute("href") !== config.image) {
    setAttributes(background, { href: config.image, width, height });
  }
  if (!endpoints) {
    // On later reruns, keep the current local endpoints rather than replaying old state.
    endpoints = structuredClone(config.endpoints);
    config.colors.forEach((color, i) => {
      lines.push(svgElement("line", linesGroup, { class: "survey-line", stroke: color }));
      const moveLabel = `Move Line ${i + 1}`;
      const target = svgElement("line", linesGroup, {
        class: "line-target", tabindex: 0, "aria-label": moveLabel,
      });
      svgElement("title", target, {}).textContent = moveLabel;
      lineTargets.push(target);
      bindControl(target, i, "move");
      handles.push(endpoints[i].map((_, j) => {
        const label = `Line ${i + 1} ${j === 0 ? "start" : "end"} endpoint`;
        const handle = svgElement("circle", handlesGroup, {
          class: "handle", r: 10, fill: color, tabindex: 0, "aria-label": label,
        });
        svgElement("title", handle, {}).textContent = label;
        bindControl(handle, i, "endpoint", j);
        return handle;
      }));
      const rotateLabel = `Rotate Line ${i + 1}`;
      const rotation = svgElement("g", rotationsGroup, {
        class: "rotation", tabindex: 0, "aria-label": rotateLabel,
      });
      svgElement("title", rotation, {}).textContent = rotateLabel;
      svgElement("circle", rotation, { r: 12, fill: color });
      svgElement("path", rotation, {
        d: "M 6 0 A 6 6 0 1 1 1 -6 M 1 -9 L 4 -6 L 1 -3",
      });
      rotations.push(rotation);
      bindControl(rotation, i, "rotate");
    });
    reset.disabled = false;
  }
  draw();
  updateHeight();
});

new ResizeObserver(updateHeight).observe(document.querySelector("main"));
send("streamlit:componentReady", { apiVersion: 1 });
