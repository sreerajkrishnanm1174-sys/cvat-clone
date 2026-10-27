/* Double-click a polygon or polyline edge to insert a point. */
(function () {
  var active = null;
  var gesture = null;
  var pointer = { x: 0, y: 0 };
  var latest = {};
  var knownOffset = {};

  function parsePoints(text) {
    var nums = [];
    String(text || "").trim().split(/[\s,]+/).forEach(function (part) {
      if (part !== "") nums.push(Number(part));
    });
    return nums;
  }

  function nearestEdge(points, x, y, maxDistance, closed) {
    var count = Math.floor(points.length / 2);
    if (count < 2) return null;
    var segments = closed ? count : count - 1;
    var best = null;
    for (var i = 0; i < segments; i++) {
      var j = (i + 1) % count;
      var ax = points[i * 2];
      var ay = points[i * 2 + 1];
      var bx = points[j * 2];
      var by = points[j * 2 + 1];
      var dx = bx - ax;
      var dy = by - ay;
      var length2 = dx * dx + dy * dy;
      if (length2 < 1) continue;
      var t = ((x - ax) * dx + (y - ay) * dy) / length2;
      t = Math.max(0, Math.min(1, t));
      if (t < 0.08 || t > 0.92) continue;
      var px = ax + t * dx;
      var py = ay + t * dy;
      var distance = Math.hypot(x - px, y - py);
      if (distance > maxDistance) continue;
      if (!best || distance < best.distance) best = { index: i, t: t, distance: distance };
    }
    return best;
  }

  function shapeFromEvent(event) {
    var target = event.target;
    if (!target || !target.tagName || target.tagName.toLowerCase() === "circle") return null;
    var shape = target.closest ? target.closest("polygon, polyline") : null;
    if (!shape || !shape.id || shape.id.indexOf("cvat_canvas_shape_") !== 0) return null;
    return shape;
  }

  function clientToShape(shape, clientX, clientY) {
    var svg = shape.ownerSVGElement;
    if (!svg) return null;
    var point = svg.createSVGPoint();
    point.x = clientX;
    point.y = clientY;
    var svgMatrix = svg.getScreenCTM();
    if (!svgMatrix) return null;
    var inSvg = point.matrixTransform(svgMatrix.inverse());
    var localMatrix = shape.getCTM();
    if (!localMatrix) return inSvg;
    return inSvg.matrixTransform(localMatrix.inverse());
  }

  function stillClick() {
    if (!gesture) return false;
    return Math.hypot(pointer.x - gesture.x, pointer.y - gesture.y) < 12 && Date.now() - gesture.time < 700;
  }

  function median(values) {
    var sorted = values.slice().sort(function (a, b) { return a - b; });
    var mid = Math.floor(sorted.length / 2);
    if (!sorted.length) return 0;
    return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
  }

  function consensusOffset(svgPoints, imagePoints) {
    if (!imagePoints || svgPoints.length !== imagePoints.length || svgPoints.length < 2) return null;
    var dxs = [];
    var dys = [];
    for (var i = 0; i < svgPoints.length; i += 2) {
      dxs.push(svgPoints[i] - imagePoints[i]);
      dys.push(svgPoints[i + 1] - imagePoints[i + 1]);
    }
    return { dx: median(dxs), dy: median(dys) };
  }

  function imageFromSvg(svgPoints, clientID, imagePoints) {
    var offset = consensusOffset(svgPoints, imagePoints) || knownOffset[clientID];
    if (!offset) return imagePoints ? imagePoints.slice() : null;
    knownOffset[clientID] = offset;
    var out = [];
    for (var i = 0; i < svgPoints.length; i += 2) out.push(svgPoints[i] - offset.dx, svgPoints[i + 1] - offset.dy);
    return out;
  }

  function attach(root) {
    if (!root || root.dataset.polygonInsert) return;
    root.dataset.polygonInsert = "1";
    root.addEventListener("canvas.clicked", function (event) {
      var state = event.detail && event.detail.state;
      if (state && (state.shapeType === "polygon" || state.shapeType === "polyline")) active = state;
    });
    root.addEventListener("canvas.edited", function (event) {
      var detail = event.detail || {};
      if (detail.source === "point-insert" || !gesture || !gesture.blockDrag) return;
      if (!detail.state || detail.state.clientID !== gesture.clientID) return;
      if (!stillClick() || !detail.points || !gesture.svg || detail.points.length !== gesture.svg.length) return;
      event.stopImmediatePropagation();
    }, true);
    root.addEventListener("canvas.edited", function (event) {
      var detail = event.detail || {};
      if (!detail.state || !detail.points) return;
      latest[detail.state.clientID] = detail.points.slice();
    });
  }

  function watch() {
    var shapes = document.querySelectorAll("polygon[id^='cvat_canvas_shape_'], polyline[id^='cvat_canvas_shape_']");
    for (var i = 0; i < shapes.length; i++) {
      var svg = shapes[i].ownerSVGElement;
      if (svg) attach(svg.parentElement);
    }
  }

  document.addEventListener("pointerdown", function (event) {
    pointer.x = event.clientX;
    pointer.y = event.clientY;
  }, true);
  document.addEventListener("pointermove", function (event) {
    pointer.x = event.clientX;
    pointer.y = event.clientY;
  }, true);

  document.addEventListener("mouseup", function (event) {
    if (!gesture || event.button !== 0) return;
    pointer.x = event.clientX;
    pointer.y = event.clientY;
    gesture.blockDrag = event.detail >= 2 && stillClick();
  }, true);

  document.addEventListener("mousedown", function (event) {
    if (event.button !== 0) return;
    pointer.x = event.clientX;
    pointer.y = event.clientY;
    var shape = shapeFromEvent(event);
    if (!shape || !active || !active.points) return;
    var clientID = Number(shape.id.slice("cvat_canvas_shape_".length));
    if (active.clientID !== clientID) return;
    if (event.detail >= 2) return;
    gesture = {
      clientID: clientID,
      svg: parsePoints(shape.getAttribute("points")),
      x: event.clientX,
      y: event.clientY,
      time: Date.now(),
      fresh: true,
    };
  }, true);

  document.addEventListener("dblclick", function (event) {
    var shape = shapeFromEvent(event);
    if (!shape) return;
    event.preventDefault();
    event.stopPropagation();
    pointer.x = event.clientX;
    pointer.y = event.clientY;
    var svg = shape.ownerSVGElement;
    var root = svg && svg.parentElement;
    if (!root || !active || !active.points) return;
    var clientID = Number(shape.id.slice("cvat_canvas_shape_".length));
    if (active.clientID !== clientID) return;
    var saved = (latest[clientID] || active.points).slice();
    var liveSvg = parsePoints(shape.getAttribute("points"));
    var svgForShape = liveSvg;
    if (gesture && gesture.fresh && gesture.clientID === clientID && stillClick() && gesture.svg && gesture.svg.length === liveSvg.length) {
      svgForShape = gesture.svg;
    }
    if (gesture) {
      gesture.fresh = false;
      gesture.blockDrag = false;
    }
    var base = imageFromSvg(svgForShape, clientID, saved);
    var local = clientToShape(shape, event.clientX, event.clientY);
    var scaleProbe = clientToShape(shape, event.clientX + 14, event.clientY);
    if (!base || !local || !scaleProbe) return;
    var maxDistance = Math.hypot(scaleProbe.x - local.x, scaleProbe.y - local.y);
    if (liveSvg.length < 4 || base.length < 4 || liveSvg.length !== base.length) return;
    var hit = nearestEdge(liveSvg, local.x, local.y, maxDistance, shape.tagName.toLowerCase() === "polygon");
    if (!hit) return;
    var count = Math.floor(base.length / 2);
    var next = (hit.index + 1) % count;
    var ax = base[hit.index * 2];
    var ay = base[hit.index * 2 + 1];
    var bx = base[next * 2];
    var by = base[next * 2 + 1];
    var imagePoints = base.slice();
    imagePoints.splice((hit.index + 1) * 2, 0, ax + hit.t * (bx - ax), ay + hit.t * (by - ay));
    latest[clientID] = imagePoints.slice();
    root.dispatchEvent(new CustomEvent("canvas.edited", {
      bubbles: false,
      cancelable: true,
      detail: {
        state: active,
        points: imagePoints,
        rotation: active.rotation,
        source: "point-insert",
      },
    }));
  }, true);

  setInterval(watch, 400);
})();
