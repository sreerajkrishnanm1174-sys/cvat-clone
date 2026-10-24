/* Double-click a polygon or polyline edge to insert a point. */
(function () {
  var active = null;

  function parsePoints(text) {
    var nums = [];
    String(text || "").trim().split(/[\s,]+/).forEach(function (part) {
      if (part !== "") nums.push(Number(part));
    });
    return nums;
  }

  function insertPoint(points, x, y, maxDistance, closed) {
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
      if (!best || distance < best.distance) best = { index: i, x: px, y: py, distance: distance };
    }
    if (!best) return null;
    var next = points.slice();
    next.splice((best.index + 1) * 2, 0, best.x, best.y);
    return next;
  }

  function attach(root) {
    if (!root || root.dataset.polygonInsert) return;
    root.dataset.polygonInsert = "1";
    root.addEventListener("canvas.clicked", function (event) {
      var state = event.detail && event.detail.state;
      if (state && (state.shapeType === "polygon" || state.shapeType === "polyline")) active = state;
    });
  }

  function watch() {
    var shapes = document.querySelectorAll("polygon[id^='cvat_canvas_shape_'], polyline[id^='cvat_canvas_shape_']");
    for (var i = 0; i < shapes.length; i++) {
      var svg = shapes[i].ownerSVGElement;
      if (svg) attach(svg.parentElement);
    }
  }

  document.addEventListener("dblclick", function (event) {
    var target = event.target;
    if (!target || !target.tagName || target.tagName.toLowerCase() === "circle") return;
    var shape = target.closest ? target.closest("polygon, polyline") : null;
    if (!shape || !shape.id || shape.id.indexOf("cvat_canvas_shape_") !== 0) return;
    var svg = shape.ownerSVGElement;
    var root = svg && svg.parentElement;
    if (!root || !active) return;
    var clientID = Number(shape.id.slice("cvat_canvas_shape_".length));
    if (active.clientID !== clientID || !active.points) return;
    var probe = svg.createSVGPoint();
    probe.x = event.clientX;
    probe.y = event.clientY;
    var matrix = shape.getScreenCTM();
    if (!matrix) return;
    var local = probe.matrixTransform(matrix.inverse());
    probe.x = event.clientX + 14;
    var shifted = probe.matrixTransform(matrix.inverse());
    var maxDistance = Math.hypot(shifted.x - local.x, shifted.y - local.y);
    var svgPoints = parsePoints(shape.getAttribute("points"));
    if (svgPoints.length < 4 || active.points.length < 4) return;
    var updated = insertPoint(svgPoints, local.x, local.y, maxDistance, shape.tagName.toLowerCase() === "polygon");
    if (!updated) return;
    var offsetX = svgPoints[0] - active.points[0];
    var offsetY = svgPoints[1] - active.points[1];
    var imagePoints = [];
    for (var i = 0; i < updated.length; i += 2) {
      imagePoints.push(updated[i] - offsetX, updated[i + 1] - offsetY);
    }
    event.preventDefault();
    event.stopPropagation();
    root.dispatchEvent(new CustomEvent("canvas.edited", {
      bubbles: false,
      cancelable: true,
      detail: { state: active, points: imagePoints, rotation: active.rotation },
    }));
  }, true);

  setInterval(watch, 400);
})();
