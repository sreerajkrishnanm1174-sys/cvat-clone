/* Requirement Analyzer panels on the CVAT dashboard, task menus, and annotation sidebar. */
(function () {
  var history = [];
  try { history = JSON.parse(sessionStorage.getItem("requirementAgentHistory") || "[]"); } catch (error) { history = []; }

  var pending = { taskId: null, projectName: "" };
  var view = { taskId: null, projectId: null, projectName: "", taskName: "" };

  function csrf() {
    var match = document.cookie.match(/(?:^|; )csrftoken=([^;]+)/);
    return match ? decodeURIComponent(match[1]) : "";
  }

  function token() {
    var saved = sessionStorage.getItem("analyzerToken");
    if (saved) return Promise.resolve(saved);
    return fetch("/api/auth/access_tokens", {
      method: "POST",
      credentials: "same-origin",
      headers: { "Content-Type": "application/json", "X-CSRFToken": csrf() },
      body: JSON.stringify({ name: "Requirement Analyzer", read_only: false }),
    }).then(function (response) {
      if (!response.ok) throw new Error("Sign in, then try again. The agent needs permission to create projects and tasks.");
      return response.json();
    }).then(function (created) {
      sessionStorage.setItem("analyzerToken", created.value);
      return created.value;
    });
  }

  function analyzer(path, options) {
    options = options || {};
    return token().then(function (value) {
      var headers = new Headers(options.headers || {});
      headers.set("Authorization", "Bearer " + value);
      if (options.json) {
        headers.set("Content-Type", "application/json");
        options = Object.assign({}, options, { body: JSON.stringify(options.json) });
      }
      return fetch("/analyzer" + path, {
        method: options.method || "GET",
        headers: headers,
        body: options.body || null,
      });
    }).then(function (response) {
      return response.text().then(function (raw) {
        var data = {};
        try { data = raw ? JSON.parse(raw) : {}; } catch (error) { data = { detail: raw }; }
        if (!response.ok) {
          var detail = data.detail || raw || "The agent could not complete that action.";
          if (typeof detail !== "string") detail = JSON.stringify(detail);
          throw new Error(detail);
        }
        return data;
      });
    });
  }

  function withOrg(path) {
    var slug = localStorage.getItem("currentOrganization") || "";
    var join = path.indexOf("?") === -1 ? "?" : "&";
    return path + join + "org=" + encodeURIComponent(slug);
  }

  function cvat(path) {
    return fetch(withOrg(path), { credentials: "same-origin" }).then(function (response) {
      if (!response.ok) throw new Error("CVAT could not load that item.");
      return response.json();
    });
  }

  function cvatSend(path, body) {
    return fetch(withOrg(path), {
      method: "POST",
      credentials: "same-origin",
      headers: { "Content-Type": "application/json", "X-CSRFToken": csrf() },
      body: JSON.stringify(body),
    }).then(function (response) {
      return response.text().then(function (raw) {
        var data = {};
        try { data = raw ? JSON.parse(raw) : {}; } catch (error) { data = { detail: raw }; }
        if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : "CVAT refused that request.");
        return data;
      });
    });
  }

  function escapeHtml(value) {
    return String(value || "").replace(/[&<>"]/g, function (char) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;" }[char];
    });
  }

  function saveHistory() {
    sessionStorage.setItem("requirementAgentHistory", JSON.stringify(history.slice(-20)));
  }

  function ensureStyle() {
    if (document.getElementById("ra-style")) return;
    var style = document.createElement("style");
    style.id = "ra-style";
    style.textContent = [
      "#ra-header-button{margin-left:8px;height:32px;padding:0 12px;border:1px solid #d9d9d9;border-radius:6px;background:#fff;color:#1f1f1f;font:500 13px/30px sans-serif;cursor:pointer}",
      "#ra-header-button:hover{border-color:#1677ff;color:#1677ff}",
      "#ra-drawer{position:fixed;top:0;right:0;bottom:0;width:420px;max-width:100vw;background:#fff;border-left:1px solid #f0f0f0;box-shadow:-8px 0 24px rgba(0,0,0,.08);z-index:10020;display:none;font:14px/1.5 -apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;color:#1f1f1f}",
      "#ra-drawer.is-open{display:flex;flex-direction:column}",
      "#ra-drawer header{display:flex;align-items:flex-start;justify-content:space-between;gap:12px;padding:16px 18px;border-bottom:1px solid #f0f0f0}",
      "#ra-drawer h2{margin:0;font-size:16px;font-weight:650}",
      "#ra-drawer header p{margin:4px 0 0;color:#667085;font-size:12px}",
      "#ra-close{border:0;background:#f5f5f5;width:32px;height:32px;border-radius:6px;font-size:20px;line-height:1;cursor:pointer;color:#1f1f1f;flex:none}",
      "#ra-body{overflow:auto;padding:16px 18px 24px}",
      ".ra-card{border:1px solid #f0f0f0;border-radius:10px;padding:14px;margin:0 0 12px;background:#fff}",
      ".ra-card h3{margin:0 0 6px;font-size:13px;font-weight:650}",
      ".ra-muted{color:#667085;font-size:12px;margin:0 0 10px}",
      ".ra-row{display:flex;flex-wrap:wrap;gap:8px}",
      ".ra-btn{border:1px solid #d9d9d9;background:#fff;border-radius:6px;padding:7px 10px;font-size:13px;cursor:pointer}",
      ".ra-btn.primary{background:#1677ff;border-color:#1677ff;color:#fff}",
      ".ra-btn:disabled{opacity:.55;cursor:default}",
      ".ra-chip{display:inline-block;margin:0 6px 6px 0;padding:2px 8px;border-radius:99px;background:#f5f5f5;font-size:12px}",
      ".ra-status{margin-top:8px;padding:8px 10px;border-radius:8px;background:#f6f8fa;font-size:12px;white-space:pre-wrap}",
      "#ra-log{max-height:180px;overflow:auto;background:#fafafa;border-radius:8px;padding:8px;margin-bottom:8px}",
      "#ra-log div{margin:0 0 8px;white-space:pre-wrap}",
      "#ra-text{width:100%;box-sizing:border-box;border:1px solid #d9d9d9;border-radius:6px;padding:8px;font:inherit;resize:vertical}",
      "#ra-dashboard,#ra-task-banner{margin:12px 16px 0;padding:14px 16px;border:1px solid #d6e4ff;border-radius:10px;background:linear-gradient(180deg,#f5f8ff,#fff)}",
      "#ra-dashboard h3,#ra-task-banner h3{margin:0 0 4px;font-size:15px}",
      "#ra-sidebar-button{width:32px;height:32px;margin:4px auto;display:flex;align-items:center;justify-content:center;border:1px solid #d6e4ff;border-radius:6px;background:#f5f8ff;color:#1677ff;cursor:pointer}",
      ".ra-menu-item{cursor:pointer}",
      "#ra-image-dialog{position:fixed;inset:0;z-index:10030;background:rgba(0,0,0,.35);display:flex;align-items:center;justify-content:center}",
      "#ra-image-dialog .ra-dialog-card{width:420px;max-width:92vw;background:#fff;border-radius:10px;padding:16px;box-shadow:0 12px 40px rgba(0,0,0,.2)}",
      "#ra-dialog-images{width:100%;box-sizing:border-box;margin:0 0 12px}",
      ".ra-ai-option{display:block;width:100%;margin-top:10px;text-align:left}",
      ".ra-switch{display:flex;align-items:center;gap:8px;font-size:13px}",
      ".ra-decision{border-top:1px solid #f0f0f0;padding:8px 0}",
      ".ra-decision p{margin:4px 0;font-size:12px}",
      ".ra-decision strong{font-size:13px}",
      ".ra-project{display:block;width:100%;text-align:left;border:1px solid #f0f0f0;background:#fff;border-radius:8px;padding:8px 10px;margin:0 0 6px;cursor:pointer;font:inherit;color:inherit}",
      ".ra-project.active{border-color:#1677ff;background:#f5f8ff}",
      ".ra-project strong{display:block;font-size:13px}",
      ".ra-project span{color:#667085;font-size:12px}",
      "#ra-new-name,#ra-task-name,#ra-task-choose,#ra-task-images{width:100%;box-sizing:border-box;border:1px solid #d9d9d9;border-radius:6px;padding:7px 8px;margin:0 0 8px;font:inherit;background:#fff}",
      "#ra-new{margin-top:8px}",
      "#ra-export-note{position:fixed;right:16px;bottom:16px;z-index:10001;max-width:320px;padding:10px 12px;border-radius:8px;background:#17324d;color:#fff;font-size:13px;box-shadow:0 8px 24px rgba(0,0,0,.18)}",
    ].join("");
    document.head.appendChild(style);
  }

  function drawer() {
    ensureStyle();
    var panel = document.getElementById("ra-drawer");
    if (panel) return panel;
    panel = document.createElement("aside");
    panel.id = "ra-drawer";
    panel.classList.remove("is-open");
    panel.innerHTML = [
      "<header><div><h2>Requirement agent</h2><p id='ra-context'>Learning stays inside one project.</p></div><button id='ra-close' type='button' aria-label='Close'>&times;</button></header>",
      "<div id='ra-body'>",
      "<section class='ra-card'><div class='ra-row' style='justify-content:space-between;align-items:center'><h3 style='margin:0'>Projects</h3><button class='ra-btn' id='ra-new-toggle' type='button'>New project</button></div><p class='ra-muted' id='ra-list-copy' style='margin-top:8px'>Loading your projects…</p><div id='ra-project-list'></div><div id='ra-new' hidden><input id='ra-new-name' type='text' placeholder='Project name'/><label class='ra-btn'>Requirement document<input id='ra-new-file' type='file' accept='.pdf,.docx,.txt,.md' hidden/></label><div class='ra-row' style='margin-top:8px'><button class='ra-btn primary' id='ra-new-create' type='button'>Create project</button></div><div class='ra-status' id='ra-new-status' hidden></div></div></section>",
      "<section class='ra-card' id='ra-detail'><h3>This project</h3><p class='ra-muted' id='ra-project-copy'>Select a project. The agent reads its requirement and explains the tool, the drawing method, and the decision for each label.</p><div id='ra-labels'></div><div id='ra-guide'></div><div id='ra-tasks'></div><p class='ra-muted' id='ra-learning'></p><p class='ra-muted'>Corrections are learned automatically when annotations in this project are saved.</p><label class='ra-switch'><input id='ra-learn' type='checkbox'/> Learn from this project's corrections</label></section>",
      "<section class='ra-card'><h3>Requirement</h3><p class='ra-muted'>After the document is read, the agent decides what this project needs: boxes, polygons, masks, polylines, points, ellipses, cuboids, tags, a human skeleton, or tracking across frames.</p><div class='ra-row'><label class='ra-btn'>Upload requirement<input id='ra-file' type='file' accept='.pdf,.docx,.txt,.md' hidden/></label><button class='ra-btn' id='ra-analyze' type='button'>Analyze requirement</button><button class='ra-btn' id='ra-apply' type='button'>Apply labels</button></div><div class='ra-status' id='ra-req-status' hidden></div></section>",
      "<section class='ra-card'><h3>Task in this project</h3><p class='ra-muted'>Choose the images, then create the task. Auto-annotation starts only after CVAT has imported those images into the task.</p><input id='ra-task-name' type='text' placeholder='Task name'/><input id='ra-task-images' type='file' accept='image/*,.zip,.jpg,.jpeg,.png,.webp,.bmp,.tif,.tiff' multiple/><p class='ra-muted' id='ra-task-file-note'>No images chosen.</p><div class='ra-row'><button class='ra-btn primary' id='ra-task-create' type='button'>Create task</button></div><div class='ra-status' id='ra-task-status' hidden></div><p class='ra-muted' style='margin-top:10px'>Choose a task in this project and open it in the annotation tool.</p><select id='ra-task-choose'><option value=''>No task selected</option></select><div class='ra-row'><button class='ra-btn primary' id='ra-open-tool' type='button'>Open in tool</button><button class='ra-btn' id='ra-annotate' type='button'>Annotate this task</button><button class='ra-btn' id='ra-sync' type='button'>Learn from my corrections</button></div><p class='ra-muted' style='margin-top:10px'>Add images to the selected task. Add keeps the current images. Replace removes them.</p><input id='ra-more-images' type='file' accept='image/*,.zip,.jpg,.jpeg,.png,.webp,.bmp,.tif,.tiff' multiple/><p class='ra-muted' id='ra-more-note'>No images chosen.</p><div class='ra-row'><button class='ra-btn primary' id='ra-images-add' type='button'>Add to current images</button><button class='ra-btn' id='ra-images-replace' type='button'>Replace current images</button></div></section>",
      "<section class='ra-card'><h3>Ask the agent</h3><p class='ra-muted'>Ask it to correct labels, the requirement, tasks, or images. It can create, change, and delete them in this project.</p><div id='ra-log'></div><input id='ra-files' type='file' multiple/><textarea id='ra-text' rows='3' placeholder='Remove the standing label and keep person. Add these images to the current task.'></textarea><div class='ra-row' style='margin-top:8px'><button class='ra-btn primary' id='ra-send' type='button'>Send</button><button class='ra-btn' id='ra-clear' type='button'>Clear chat</button></div></section>",
      "</div>",
    ].join("");
    document.body.appendChild(panel);
    panel.querySelector("#ra-close").onclick = function (event) {
      event.preventDefault();
      event.stopPropagation();
      panel.classList.remove("is-open");
    };
    panel.querySelector("#ra-file").onchange = onUpload;
    panel.querySelector("#ra-analyze").onclick = onAnalyze;
    panel.querySelector("#ra-apply").onclick = onApply;
    panel.querySelector("#ra-task-images").onchange = function (event) {
      var note = document.getElementById("ra-task-file-note");
      var files = event.target.files ? Array.prototype.slice.call(event.target.files) : [];
      if (!note) return;
      note.textContent = files.length
        ? files.length + " file(s) chosen: " + files.slice(0, 4).map(function (file) { return file.name; }).join(", ")
        : "No images chosen.";
    };
    panel.querySelector("#ra-task-create").onclick = createTaskInProject;
    panel.querySelector("#ra-task-choose").onchange = function (event) {
      var option = event.target.selectedOptions && event.target.selectedOptions[0];
      chooseTask(Number(event.target.value) || null, option ? option.textContent : "");
    };
    panel.querySelector("#ra-more-images").onchange = function (event) {
      var note = document.getElementById("ra-more-note");
      var files = event.target.files ? Array.prototype.slice.call(event.target.files) : [];
      if (!note) return;
      note.textContent = files.length
        ? files.length + " file(s) chosen: " + files.slice(0, 4).map(function (file) { return file.name; }).join(", ")
        : "No images chosen.";
    };
    panel.querySelector("#ra-images-add").onclick = function () { updateTaskImages("add"); };
    panel.querySelector("#ra-images-replace").onclick = function () { updateTaskImages("replace"); };
    panel.querySelector("#ra-open-tool").onclick = openTaskInTool;
    panel.querySelector("#ra-annotate").onclick = onAnnotate;
    panel.querySelector("#ra-sync").onclick = onSync;
    panel.querySelector("#ra-learn").onchange = onLearnToggle;
    panel.querySelector("#ra-send").onclick = onSend;
    panel.querySelector("#ra-clear").onclick = clearChat;
    panel.querySelector("#ra-new-toggle").onclick = function () {
      var box = document.getElementById("ra-new");
      box.hidden = !box.hidden;
    };
    panel.querySelector("#ra-new-create").onclick = createProject;
    paintLog();
    return panel;
  }

  function setStatus(id, text) {
    var node = document.getElementById(id);
    if (node) {
      node.hidden = !text;
      node.textContent = text || "";
    }
    if (id === "ra-task-status") exportNote(text || "");
  }

  function paintLog() {
    var log = document.getElementById("ra-log");
    if (!log) return;
    log.innerHTML = "";
    history.slice(-8).forEach(function (item) {
      var row = document.createElement("div");
      row.textContent = (item.role === "user" ? "You" : "Agent") + ": " + String(item.content || "").replace(/\*\*/g, "").replace(/`/g, "");
      log.appendChild(row);
    });
    log.scrollTop = log.scrollHeight;
  }

  function renderProject(spec, learning) {
    var copy = document.getElementById("ra-project-copy");
    var labels = document.getElementById("ra-labels");
    var learningNode = document.getElementById("ra-learning");
    var toggle = document.getElementById("ra-learn");
    if (!view.projectId) {
      copy.textContent = view.taskId
        ? "This task is not in a project. Move it into a project so the requirement and the learning stay together."
        : "Select a project above, or create one with a requirement document.";
      labels.innerHTML = "";
      document.getElementById("ra-guide").innerHTML = "";
      document.getElementById("ra-tasks").innerHTML = "";
      learningNode.textContent = "";
      toggle.checked = false;
      toggle.disabled = true;
      return;
    }
    var title = view.projectName ? view.projectName + " · project " + view.projectId : "Project " + view.projectId;
    if (view.taskId) title += " · task " + view.taskId + (view.taskName ? " " + view.taskName : "");
    copy.textContent = title;
    var items = ((spec && spec.spec && spec.spec.labels) || []);
    labels.innerHTML = items.length
      ? items.map(function (item) {
        return "<span class='ra-chip'>" + item.name + " · " + (item.type || "any") + "</span>";
      }).join("")
      : "<span class='ra-muted'>No requirement document is saved for this project yet.</span>";
    learningNode.textContent = learning
      ? (learning.summary || "") + (learning.mode_text ? " " + learning.mode_text : "")
      : "";
    renderGuide(spec, view.report);
    toggle.disabled = false;
    toggle.checked = !!(learning && learning.learn_enabled);
  }

  function renderGuide(spec, report) {
    var node = document.getElementById("ra-guide");
    if (!node) return;
    var guide = spec && spec.spec && spec.spec.guide;
    var decisions = (guide && guide.decisions) || [];
    var steps = (report && report.steps) || [];
    var annotations = (report && report.annotations) || [];
    var extras = (report && report.extra_labels) || [];
    var format = (report && report.output_format) || (spec && spec.spec && spec.spec.output_format);
    var formatLine = format && !/^cvat$/i.test(format)
      ? "<p class='ra-muted'>Export format from the requirement: " + escapeHtml(format) + ". It is added when a dataset export finishes.</p>"
      : "";
    var stepHtml = steps.length
      ? steps.map(function (item) {
        return "<article class='ra-decision'><strong>" + item.step + ". " + escapeHtml(item.title) + "</strong><p>" + escapeHtml(item.detail || "") + "</p></article>";
      }).join("")
      : "";
    var annotationHtml = "<p class='ra-muted'><b>Annotations in this project.</b></p>" + (annotations.length
      ? annotations.map(function (task) {
        return "<p class='ra-muted'>" + escapeHtml(task.name) + " · task " + task.id + " · " + (task.images || 0) + " images · " + (task.shapes || 0) + " shapes</p>";
      }).join("")
      : "<p class='ra-muted'>No tasks yet.</p>");
    var extraHtml = extras.length
      ? "<p class='ra-muted'>Labels that are not in the requirement: " + escapeHtml(extras.join(", ")) + ". Ask the agent to clear them.</p>"
      : "";
    if (!steps.length && !decisions.length) {
      node.innerHTML = formatLine + extraHtml + "<p class='ra-muted'>Choose Analyze requirement. The agent will list the steps and the annotations for this project.</p>";
      return;
    }
    if (steps.length) {
      node.innerHTML = formatLine + stepHtml + annotationHtml + extraHtml;
      return;
    }
    node.innerHTML = formatLine + "<p class='ra-muted'>" + escapeHtml(guide.summary || "") + "</p>" + decisions.map(function (item) {
      return "<article class='ra-decision'><strong>" + escapeHtml(item.name) + " · " + escapeHtml(item.type || "") + "</strong>"
        + "<p><b>Why this type.</b> " + escapeHtml(item.why_this_type || "") + "</p>"
        + "<p><b>How to annotate.</b> " + escapeHtml(item.how_to_annotate || "") + "</p>"
        + "<p><b>How to decide.</b> " + escapeHtml(item.how_to_decide || "") + "</p>"
        + (item.keypoints && item.keypoints.length ? "<p><b>Keypoints.</b> " + escapeHtml(item.keypoints.join(", ")) + "</p>" : "")
        + "</article>";
    }).join("")
      + (spec.spec.tracking ? "<p class='ra-muted'>Tracking is on for this project. Keep the same object id across frames.</p>" : "")
      + "<p class='ra-muted'>" + escapeHtml(guide.workflow || "") + "</p>";
  }

  function chooseTask(id, name) {
    view.taskId = id || null;
    view.taskName = name || "";
    var select = document.getElementById("ra-task-choose");
    if (select && id) select.value = String(id);
    var copy = document.getElementById("ra-project-copy");
    if (copy && view.projectId) {
      copy.textContent = (view.projectName || "Project " + view.projectId) + (id ? " · task " + id + " " + view.taskName : "");
    }
    Array.prototype.forEach.call(document.querySelectorAll(".ra-task"), function (item) {
      item.classList.toggle("active", Number(item.getAttribute("data-task")) === id);
    });
    if (id && !arguments[2]) setStatus("ra-task-status", "Task " + id + " is selected. Annotate it after its images are imported, or open it in the tool.");
  }

  function openTaskInTool() {
    if (!view.taskId) {
      setStatus("ra-task-status", "Choose a task first.");
      return;
    }
    setStatus("ra-task-status", "Opening task " + view.taskId + " in the annotation tool…");
    cvat("/api/jobs?task_id=" + view.taskId + "&page_size=20").then(function (page) {
      var job = (page.results || [])[0];
      if (!job) {
        setStatus("ra-task-status", "This task has no job yet. Wait until the images finish uploading, then open it again.");
        return;
      }
      window.location.assign("/tasks/" + view.taskId + "/jobs/" + job.id);
    }).catch(function (error) {
      setStatus("ra-task-status", error.message);
    });
  }

  function renderTasks(tasks) {
    var node = document.getElementById("ra-tasks");
    var select = document.getElementById("ra-task-choose");
    if (select) {
      var current = view.taskId ? String(view.taskId) : "";
      select.innerHTML = tasks.length
        ? "<option value=''>Choose a task</option>" + tasks.map(function (task) {
          return "<option value='" + task.id + "'>" + escapeHtml(task.name) + " · #" + task.id + " · " + (task.size || 0) + " images</option>";
        }).join("")
        : "<option value=''>No tasks in this project</option>";
      if (current && tasks.some(function (task) { return String(task.id) === current; })) select.value = current;
    }
    if (!node) return;
    if (!tasks.length) {
      node.innerHTML = "<p class='ra-muted'>This project has no tasks yet.</p>";
      return;
    }
    node.innerHTML = "<p class='ra-muted'>Tasks</p>" + tasks.map(function (task) {
      var active = view.taskId === task.id ? " active" : "";
      return "<button type='button' class='ra-project ra-task" + active + "' data-task='" + task.id + "'><strong>" + escapeHtml(task.name) + "</strong><span>Task " + task.id + " · " + (task.size || 0) + " images</span></button>";
    }).join("");
    Array.prototype.forEach.call(node.querySelectorAll(".ra-task"), function (button) {
      button.onclick = function () {
        chooseTask(Number(button.getAttribute("data-task")), button.querySelector("strong").textContent);
      };
    });
  }

  function renderProjectList(rows) {
    var list = document.getElementById("ra-project-list");
    var copy = document.getElementById("ra-list-copy");
    var form = document.getElementById("ra-new");
    if (!list) return;
    if (!rows.length) {
      copy.textContent = "You have no projects in this workspace yet. Add one with its requirement document.";
      list.innerHTML = "";
      if (form) form.hidden = false;
      return;
    }
    copy.textContent = "Select a project to see its requirement, labels, tasks, and learning.";
    list.innerHTML = rows.map(function (row) {
      var active = view.projectId === row.id ? " active" : "";
      return "<button type='button' class='ra-project" + active + "' data-id='" + row.id + "'><strong>" + escapeHtml(row.name) + "</strong><span>Project " + row.id + "</span></button>";
    }).join("");
    Array.prototype.forEach.call(list.querySelectorAll(".ra-project"), function (button) {
      button.onclick = function () {
        var id = Number(button.getAttribute("data-id"));
        var found = rows.filter(function (row) { return row.id === id; })[0];
        selectProject(id, found ? found.name : "");
      };
    });
  }

  function loadProjectList() {
    return cvat("/api/projects?page_size=100").then(function (page) {
      renderProjectList(page.results || []);
    }).catch(function () {
      var copy = document.getElementById("ra-list-copy");
      if (copy) copy.textContent = "The project list could not be loaded. Refresh CVAT and open the agent again.";
    });
  }

  function selectProject(id, name) {
    view.projectId = id;
    view.projectName = name || "";
    view.taskId = null;
    view.taskName = "";
    var context = document.getElementById("ra-context");
    if (context) context.textContent = "Learning is stored for this project only, from its requirement document and your corrections.";
    Array.prototype.forEach.call(document.querySelectorAll("#ra-project-list .ra-project"), function (button) {
      button.classList.toggle("active", Number(button.getAttribute("data-id")) === id);
    });
    setStatus("ra-req-status", "");
    setStatus("ra-task-status", "");
    return refresh();
  }

  function createProject() {
    var nameInput = document.getElementById("ra-new-name");
    var fileInput = document.getElementById("ra-new-file");
    var name = nameInput.value.trim();
    var file = fileInput.files && fileInput.files[0];
    if (!name) {
      setStatus("ra-new-status", "Enter a project name.");
      return;
    }
    var button = document.getElementById("ra-new-create");
    button.disabled = true;
    setStatus("ra-new-status", "Creating " + name + "…");
    cvatSend("/api/projects", { name: name }).then(function (created) {
      view.projectId = created.id;
      view.projectName = created.name || name;
      if (!file) return null;
      var body = new FormData();
      body.append("file", file);
      setStatus("ra-new-status", "Reading the requirement…");
      return analyzer("/projects/" + created.id + "/requirements", { method: "POST", body: body }).then(function () {
        return analyzer("/projects/" + created.id + "/labels/apply", { method: "POST", json: { confirm_type_changes: false } });
      });
    }).then(function () {
      nameInput.value = "";
      fileInput.value = "";
      setStatus("ra-new-status", "Project created" + (file ? " and its requirement was applied." : ". Upload a requirement when you have the document."));
      return loadProjectList().then(function () { return selectProject(view.projectId, view.projectName); });
    }).catch(function (error) {
      setStatus("ra-new-status", error.message);
    }).then(function () { button.disabled = false; });
  }

  function refresh() {
    if (!view.projectId) {
      renderProject(null, null);
      return Promise.resolve();
    }
    return Promise.all([
      analyzer("/projects/" + view.projectId + "/spec").catch(function () { return null; }),
      analyzer("/projects/" + view.projectId + "/learning").catch(function () { return null; }),
      cvat("/api/tasks?project_id=" + view.projectId + "&page_size=100").catch(function () { return { results: [] }; }),
      analyzer("/projects/" + view.projectId + "/brief").catch(function () { return null; }),
    ]).then(function (result) {
      view.report = result[3];
      renderProject(result[0], result[1]);
      renderTasks((result[2] && result[2].results) || []);
    });
  }

  function openPanel(next) {
    view.taskId = next.taskId || null;
    view.projectId = next.projectId || null;
    view.projectName = next.projectName || "";
    view.taskName = next.taskName || "";
    var panel = drawer();
    panel.classList.add("is-open");
    document.getElementById("ra-context").textContent = view.projectId
      ? "Learning is stored for this project only, from its requirement document and your corrections."
      : "Create or open a project. Each project keeps its own requirement and learning.";
    setStatus("ra-req-status", "");
    setStatus("ra-task-status", "");
    var chain = Promise.resolve();
    if (view.taskId && !view.projectId) {
      chain = cvat("/api/tasks/" + view.taskId).then(function (task) {
        view.projectId = task.project_id || null;
        view.taskName = task.name || view.taskName;
      }).catch(function () {});
    } else if (!view.taskId && !view.projectId && pending.projectName) {
      chain = cvat("/api/projects?page_size=100").then(function (page) {
        var rows = page.results || [];
        var found = rows.filter(function (row) { return row.name === pending.projectName; })[0];
        if (found) {
          view.projectId = found.id;
          view.projectName = found.name;
        }
      }).catch(function () {});
    }
    chain.then(function () {
      return loadProjectList();
    }).then(refresh);
  }

  function contextFromLocation() {
    var task = location.pathname.match(/\/tasks\/(\d+)/);
    var project = location.pathname.match(/\/projects\/(\d+)/);
    return {
      taskId: task ? Number(task[1]) : null,
      projectId: project ? Number(project[1]) : null,
    };
  }

  function onUpload(event) {
    var file = event.target.files && event.target.files[0];
    event.target.value = "";
    if (!file || !view.projectId) {
      setStatus("ra-req-status", view.projectId ? "" : "Open a project before uploading its requirement.");
      return;
    }
    var body = new FormData();
    body.append("file", file);
    setStatus("ra-req-status", "Reading " + file.name + "…");
    analyzer("/projects/" + view.projectId + "/requirements", { method: "POST", body: body }).then(function (result) {
      var labels = (result.spec && result.spec.labels) || [];
      var names = labels.map(function (item) { return item.name + " (" + (item.type || "any") + ")"; }).join(", ");
      setStatus("ra-req-status", labels.length
        ? "The project is set up from the requirement. " + labels.length + " label(s) applied: " + names + "."
        : "The document was read, but it did not describe any annotation class. Analyze requirement if the skeleton properties were not picked up.");
      return refresh();
    }).catch(function (error) { setStatus("ra-req-status", error.message); });
  }

  function onAnalyze() {
    if (!view.projectId) {
      setStatus("ra-req-status", "Select a project that has a requirement document.");
      return;
    }
    var button = document.getElementById("ra-analyze");
    button.disabled = true;
    setStatus("ra-req-status", "Reading the requirement. Deciding the annotation type, how to draw it, and how to decide each object…");
    analyzer("/projects/" + view.projectId + "/requirements/analyze", { method: "POST" }).then(function (result) {
      var labels = (result.spec && result.spec.labels) || [];
      var names = labels.map(function (item) { return item.name + " (" + (item.type || "any") + ")"; }).join(", ");
      setStatus("ra-req-status", labels.length
        ? "Project set up automatically. " + (result.applied || 0) + " label change(s): " + names + "."
        : "The requirement still has no annotation class to add.");
      return refresh();
    }).catch(function (error) {
      setStatus("ra-req-status", error.message);
    }).then(function () { button.disabled = false; });
  }

  function cvatPost(path, options) {
    options = options || {};
    var headers = { "X-CSRFToken": csrf() };
    Object.keys(options.headers || {}).forEach(function (key) { headers[key] = options.headers[key]; });
    if (options.json) headers["Content-Type"] = "application/json";
    return fetch(withOrg(path), {
      method: "POST",
      credentials: "same-origin",
      headers: headers,
      body: options.json ? JSON.stringify(options.json) : (options.body || null),
    }).then(function (response) {
      return response.text().then(function (raw) {
        var data = {};
        try { data = raw ? JSON.parse(raw) : {}; } catch (error) { data = { detail: raw }; }
        if (!response.ok) {
          var detail = data.detail || data.message || raw || "CVAT could not upload the images.";
          if (typeof detail !== "string") detail = "CVAT could not upload the images.";
          throw new Error(detail);
        }
        return data;
      });
    });
  }

  function waitForImport(rqId) {
    var tries = 0;
    function poll() {
      return cvat("/api/requests/" + encodeURIComponent(rqId)).then(function (data) {
        if (data.status === "finished") return data;
        if (data.status === "failed") throw new Error(data.message || "CVAT could not import the images.");
        tries += 1;
        if (tries > 150) throw new Error("CVAT is still importing the images. Open the task after it finishes, then annotate.");
        var percent = Math.round(Number(data.progress || 0) * 100);
        setStatus("ra-task-status", "CVAT is importing the images" + (percent ? " (" + percent + "%)" : "") + "…");
        return new Promise(function (resolve) { setTimeout(resolve, 2000); }).then(poll);
      });
    }
    return poll();
  }

  function uploadTaskImages(taskId, images) {
    var path = "/api/tasks/" + taskId + "/data";
    var batches = [];
    var current = [];
    var size = 0;
    var limit = 80 * 1024 * 1024;
    images.forEach(function (file) {
      if (current.length && size + file.size > limit) {
        batches.push(current);
        current = [];
        size = 0;
      }
      current.push(file);
      size += file.size;
    });
    if (current.length) batches.push(current);
    return cvatPost(path, { headers: { "Upload-Start": "true" } }).then(function () {
      var chain = Promise.resolve();
      batches.forEach(function (batch, batchIndex) {
        chain = chain.then(function () {
          setStatus("ra-task-status", "Uploading images (" + (batchIndex + 1) + " of " + batches.length + ") into task " + taskId + "…");
          var form = new FormData();
          form.append("image_quality", "70");
          batch.forEach(function (file, index) { form.append("client_files[" + index + "]", file, file.name); });
          return cvatPost(path, { body: form, headers: { "Upload-Multiple": "true" } });
        });
      });
      return chain;
    }).then(function () {
      setStatus("ra-task-status", "CVAT is importing the images into task " + taskId + "…");
      return cvatPost(path, { json: { image_quality: 70 }, headers: { "Upload-Finish": "true" } });
    }).then(function (started) {
      if (!started.rq_id) throw new Error("CVAT did not start the image import.");
      return waitForImport(started.rq_id);
    }).then(function () {
      return cvat("/api/tasks/" + taskId);
    }).then(function (task) {
      if (!task.size) throw new Error("The images were not attached to the task. Choose the files again and create the task.");
      return task;
    });
  }

  function uniqueFileName(name, used) {
    var clean = String(name || "image.jpg").split(/[/\\]/).pop() || "image.jpg";
    if (!used[clean]) {
      used[clean] = true;
      return clean;
    }
    var dot = clean.lastIndexOf(".");
    var stem = dot > 0 ? clean.slice(0, dot) : clean;
    var ext = dot > 0 ? clean.slice(dot) : "";
    var index = 2;
    var next = stem + "_" + index + ext;
    while (used[next]) {
      index += 1;
      next = stem + "_" + index + ext;
    }
    used[next] = true;
    return next;
  }

  function stripAnnotationIds(value) {
    if (Array.isArray(value)) return value.map(stripAnnotationIds);
    if (!value || typeof value !== "object") return value;
    var copy = {};
    Object.keys(value).forEach(function (key) {
      if (key === "id") return;
      copy[key] = stripAnnotationIds(value[key]);
    });
    return copy;
  }

  function deleteCvatTask(taskId) {
    return fetch(withOrg("/api/tasks/" + taskId), {
      method: "DELETE",
      credentials: "same-origin",
      headers: { "X-CSRFToken": csrf() },
    }).then(function (response) {
      if (!response.ok && response.status !== 404) throw new Error("The previous task could not be removed.");
    });
  }

  function readCurrentImages(task) {
    return cvat("/api/tasks/" + task.id + "/data/meta").then(function (meta) {
      var frames = meta.frames || [];
      var count = Number(task.size || frames.length || 0);
      var files = [];
      var names = [];
      var used = {};
      function next(index) {
        if (index >= count) return { files: files, names: names };
        setStatus("ra-task-status", "Reading current image " + (index + 1) + " of " + count + "…");
        return fetch(withOrg("/api/tasks/" + task.id + "/data?type=frame&quality=original&number=" + index), {
          credentials: "same-origin",
        }).then(function (response) {
          if (!response.ok) throw new Error("Could not read image " + (index + 1) + " from this task.");
          return response.blob().then(function (blob) {
            var frame = frames[index] || {};
            var name = uniqueFileName(frame.name || ("frame_" + index + ".jpg"), used);
            names[index] = name;
            files.push(new File([blob], name, { type: blob.type || "image/jpeg" }));
            return next(index + 1);
          });
        });
      }
      return next(0);
    });
  }

  function remapAnnotations(annotations, oldNames, newFrames) {
    var byName = {};
    (newFrames || []).forEach(function (frame, index) {
      var name = String((frame && frame.name) || "").split(/[/\\]/).pop();
      if (name) byName[name] = index;
    });
    var payload = stripAnnotationIds(annotations || {});
    function mapFrame(frame) {
      var name = oldNames[Number(frame)];
      if (!name || byName[name] == null) return frame;
      return byName[name];
    }
    (payload.shapes || []).forEach(function (shape) { shape.frame = mapFrame(shape.frame); });
    (payload.tags || []).forEach(function (tag) { tag.frame = mapFrame(tag.frame); });
    (payload.tracks || []).forEach(function (track) {
      (track.shapes || []).forEach(function (shape) { shape.frame = mapFrame(shape.frame); });
    });
    return payload;
  }

  function putTaskAnnotations(taskId, annotations, oldNames) {
    return cvat("/api/tasks/" + taskId + "/data/meta").then(function (meta) {
      var payload = oldNames ? remapAnnotations(annotations, oldNames, meta.frames || []) : stripAnnotationIds(annotations || {});
      payload.version = 0;
    var has = (payload.shapes || []).length || (payload.tracks || []).length || (payload.tags || []).length;
    if (!has) return Promise.resolve();
    return fetch(withOrg("/api/tasks/" + taskId + "/annotations"), {
      method: "PUT",
      credentials: "same-origin",
      headers: { "Content-Type": "application/json", "X-CSRFToken": csrf() },
      body: JSON.stringify(payload),
    }).then(function (response) {
      if (!response.ok) throw new Error("The images were added, but the annotations on the current images could not be copied.");
    });
    });
  }

  function updateTaskImages(mode, chosenFiles, taskId) {
    drawer();
    if (taskId) view.taskId = taskId;
    var input = document.getElementById("ra-more-images");
    var incoming = chosenFiles && chosenFiles.length
      ? chosenFiles.slice()
      : (input && input.files ? Array.prototype.slice.call(input.files) : []);
    if (!view.taskId) {
      setStatus("ra-task-status", "Choose a task in this project first.");
      return;
    }
    if (!incoming.length) {
      setStatus("ra-task-status", "Choose the new image files first.");
      return;
    }
    var adding = mode === "add";
    var addButton = document.getElementById("ra-images-add");
    var replaceButton = document.getElementById("ra-images-replace");
    var oldId = view.taskId;
    var createdId = null;
    var committed = false;
    if (addButton) addButton.disabled = true;
    if (replaceButton) replaceButton.disabled = true;
    setStatus("ra-task-status", "Checking this task…");
    cvat("/api/tasks/" + oldId).then(function (task) {
      if (task.mode === "interpolation") {
        throw new Error("Add and replace work on image tasks. This task is a video.");
      }
      var currentCount = Number(task.size || 0);
      var question = adding
        ? "Add " + incoming.length + " file(s) after the " + currentCount + " image(s) already in this task? Annotations on the current images stay."
        : "Replace the images in this task with " + incoming.length + " file(s)? The current images and their annotations will be removed.";
      if (!window.confirm(question)) {
        setStatus("ra-task-status", "No images were changed.");
        return null;
      }
      if (!currentCount) {
        setStatus("ra-task-status", "This task has no images yet. Uploading " + incoming.length + " file(s)…");
        return uploadTaskImages(oldId, incoming).then(function (updated) {
          return { task: updated, removed: false };
        });
      }
      var savedAnnotations = null;
      var savedNames = [];
      var prepare = adding
        ? cvat("/api/tasks/" + oldId + "/annotations").then(function (annotations) {
          savedAnnotations = annotations;
          return readCurrentImages(task);
        })
        : Promise.resolve({ files: [], names: [] });
      return prepare.then(function (current) {
        var currentFiles = current.files || [];
        savedNames = current.names || [];
        var used = {};
        savedNames.forEach(function (name) { used[name] = true; });
        var files = currentFiles.slice();
        incoming.forEach(function (file) {
          files.push(new File([file], uniqueFileName(file.name, used), { type: file.type }));
        });
        if (!task.project_id && !view.projectId) throw new Error("This task is not inside a project.");
        view.projectId = task.project_id || view.projectId;
        var taskName = task.name || view.taskName || "task";
        setStatus("ra-task-status", "Creating the updated image set for " + taskName + "…");
        return cvatSend("/api/tasks", { name: taskName, project_id: view.projectId }).then(function (created) {
          createdId = created.id;
          setStatus("ra-task-status", "Uploading " + files.length + " image file(s)…");
          return uploadTaskImages(created.id, files).then(function (updated) {
            var restore = adding ? putTaskAnnotations(created.id, savedAnnotations, savedNames) : Promise.resolve();
            return restore.then(function () { return updated; });
          });
        }).then(function (updated) {
          setStatus("ra-task-status", "Removing the previous copy of this task…");
          return deleteCvatTask(oldId).then(function () {
            committed = true;
            return { task: updated, removed: true, previousId: oldId };
          });
        });
      });
    }).then(function (done) {
      if (!done) return null;
      if (input) input.value = "";
      var noteNode = document.getElementById("ra-more-note");
      if (noteNode) noteNode.textContent = "No images chosen.";
      view.taskId = done.task.id;
      view.taskName = done.task.name || view.taskName;
      var note = adding
        ? "Task " + done.task.id + " now has " + done.task.size + " image(s). The images you added are after the ones that were already there."
        : "Task " + done.task.id + " now has " + done.task.size + " replacement image(s).";
      if (done.removed && done.previousId !== done.task.id) {
        note += " CVAT kept this as task " + done.task.id + " in the same project.";
      }
      return refresh().then(function () {
        chooseTask(done.task.id, view.taskName, true);
        setStatus("ra-task-status", note + " Press Annotate this task when you want the agent to draw the new images.");
        if (done.removed && location.pathname.indexOf("/tasks/" + done.previousId) !== -1) openTaskInTool();
      });
    }).catch(function (error) {
      var cleanup = createdId && !committed ? deleteCvatTask(createdId) : Promise.resolve();
      return cleanup.then(function () {
        setStatus("ra-task-status", error.message || "The images could not be updated.");
      });
    }).then(function () {
      if (addButton) addButton.disabled = false;
      if (replaceButton) replaceButton.disabled = false;
    });
  }

  function createTaskInProject() {
    if (!view.projectId) {
      setStatus("ra-task-status", "Select a project first.");
      return;
    }
    var nameInput = document.getElementById("ra-task-name");
    var imageInput = document.getElementById("ra-task-images");
    var name = nameInput.value.trim();
    var images = imageInput.files ? Array.prototype.slice.call(imageInput.files) : [];
    if (!name) {
      setStatus("ra-task-status", "Enter a task name.");
      return;
    }
    if (!images.length) {
      setStatus("ra-task-status", "Choose the images in the file box first. The task is created with those files.");
      return;
    }
    var button = document.getElementById("ra-task-create");
    var createdId = null;
    var imported = false;
    button.disabled = true;
    setStatus("ra-task-status", "Setting up the project from its requirement…");
    analyzer("/projects/" + view.projectId + "/spec").then(function (saved) {
      var labels = (saved.spec && saved.spec.labels) || [];
      if (!labels.length) throw new Error("Upload the requirement and analyze it first. The task uses the labels from that document.");
      return analyzer("/projects/" + view.projectId + "/labels/apply", { method: "POST", json: { confirm_type_changes: false } });
    }).then(function (applied) {
      setStatus("ra-task-status", "Project labels are ready (" + (applied.applied || 0) + " change(s)). Creating the task…");
      return cvatSend("/api/tasks", { name: name, project_id: view.projectId });
    }).then(function (created) {
      createdId = created.id;
      view.taskId = created.id;
      view.taskName = created.name || name;
      setStatus("ra-task-status", "Task " + created.id + " is created. Uploading " + images.length + " image file(s)…");
      return uploadTaskImages(created.id, images);
    }).then(function (task) {
      imported = true;
      nameInput.value = "";
      imageInput.value = "";
      var noteNode = document.getElementById("ra-task-file-note");
      if (noteNode) noteNode.textContent = "No images chosen.";
      chooseTask(task.id, view.taskName, true);
      var note = "Task " + task.id + " is created with " + task.size + " image(s).";
      return refresh().then(function () {
        if (!window.confirm(note + " Auto-annotate them now?")) {
          setStatus("ra-task-status", note + " Press Annotate this task when you want the agent to draw.");
          return null;
        }
        setStatus("ra-task-status", note + " Annotating from the requirement. This can take several minutes.");
        return analyzer("/projects/" + view.projectId + "/annotate", {
          method: "POST",
          json: { task_id: task.id, mode: "auto_first" },
        });
      });
    }).then(function (done) {
      if (!done) return null;
      setStatus("ra-task-status", done.message || "Annotation finished. Correct the shapes in Standard, then learn from those corrections.");
      return refresh();
    }).catch(function (error) {
      if (createdId && !imported) {
        setStatus("ra-task-status", error.message + " The task was removed because the images were not imported.");
        view.taskId = null;
        return fetch(withOrg("/api/tasks/" + createdId), {
          method: "DELETE",
          credentials: "same-origin",
          headers: { "X-CSRFToken": csrf() },
        }).then(refresh);
      }
      setStatus("ra-task-status", error.message);
      return null;
    }).then(function () { button.disabled = false; });
  }

  function onApply() {
    if (!view.projectId) return;
    setStatus("ra-req-status", "Applying labels…");
    analyzer("/projects/" + view.projectId + "/labels/apply", { method: "POST", json: { confirm_type_changes: false } })
      .then(function (result) {
        setStatus("ra-req-status", result.applied
          ? "Applied " + result.applied + " label change(s) from the requirement."
          : "The project already matches the requirement. No label changes were needed.");
      }).catch(function (error) { setStatus("ra-req-status", error.message); });
  }

  function onAnnotate() {
    if (!view.projectId || !view.taskId) {
      setStatus("ra-task-status", "Open a task that belongs to a project, then annotate it.");
      return;
    }
    var button = document.getElementById("ra-annotate");
    button.disabled = true;
    setStatus("ra-task-status", "Checking that the images are in the task…");
    cvat("/api/tasks/" + view.taskId).then(function (task) {
      if (!task.size) throw new Error("This task has no images. Choose the image files, create the task, and wait until the import finishes.");
      setStatus("ra-task-status", "Looking at " + task.size + " image(s). This can take several minutes. Correct the shapes in Standard when it finishes.");
      return analyzer("/projects/" + view.projectId + "/annotate", {
        method: "POST",
        json: { task_id: view.taskId, mode: "auto_first" },
      });
    }).then(function (result) {
      setStatus("ra-task-status", result.message || "Annotation finished.");
    }).catch(function (error) {
      setStatus("ra-task-status", error.message);
    }).then(function () { button.disabled = false; });
  }

  function onSync() {
    if (!view.projectId) return;
    setStatus("ra-task-status", "Learning from the corrections in this project…");
    analyzer("/projects/" + view.projectId + "/sync", { method: "POST" }).then(function (result) {
      setStatus("ra-task-status", "Learned " + (result.samples || 0) + " frame(s) for this project. The next task in the same project follows those corrections.");
      return refresh();
    }).catch(function (error) { setStatus("ra-task-status", error.message); });
  }

  function onLearnToggle(event) {
    if (!view.projectId) return;
    analyzer("/projects/" + view.projectId + "/learning/settings", {
      method: "POST",
      json: { learn_enabled: event.target.checked, annotation_mode: "mixed" },
    }).then(refresh).catch(function (error) { setStatus("ra-task-status", error.message); });
  }

  function clearChat() {
    history = [];
    sessionStorage.removeItem("requirementAgentHistory");
    var text = document.getElementById("ra-text");
    var files = document.getElementById("ra-files");
    if (text) text.value = "";
    if (files) files.value = "";
    paintLog();
  }

  function onSend() {
    var text = document.getElementById("ra-text");
    var files = document.getElementById("ra-files");
    var message = text.value.trim() || (files.files.length ? "Use the attached files." : "");
    if (!message) return;
    history.push({ role: "user", content: message });
    saveHistory();
    paintLog();
    text.value = "";
    var body = new FormData();
    body.append("message", message);
    body.append("history", JSON.stringify(history.slice(0, -1)));
    if (view.projectId) body.append("project_id", String(view.projectId));
    if (view.taskId) body.append("task_id", String(view.taskId));
    Array.prototype.forEach.call(files.files, function (file) { body.append("files", file); });
    files.value = "";
    analyzer("/chat", { method: "POST", body: body }).then(function (result) {
      history.push({ role: "assistant", content: result.reply });
      saveHistory();
      paintLog();
      return refresh();
    }).catch(function (error) {
      history.push({ role: "assistant", content: error.message });
      saveHistory();
      paintLog();
    });
  }

  function menuItem(label, className, sample) {
    var item = document.createElement(sample && sample.tagName ? sample.tagName : "li");
    item.className = "ant-dropdown-menu-item ant-dropdown-menu-item-only-child " + className;
    item.setAttribute("role", "menuitem");
    item.innerHTML = "<span class='ant-dropdown-menu-title-content'>" + label + "</span>";
    return item;
  }

  function openImageDialog(mode, taskId) {
    var id = taskId || pending.taskId || contextFromLocation().taskId;
    if (!id) {
      exportNote("Open a task Actions menu, then choose Add images or Replace images.");
      return;
    }
    var existing = document.getElementById("ra-image-dialog");
    if (existing) existing.remove();
    ensureStyle();
    var adding = mode === "add";
    var box = document.createElement("div");
    box.id = "ra-image-dialog";
    box.innerHTML = "<div class='ra-dialog-card'><h3>" + (adding ? "Add images" : "Replace images") + "</h3><p class='ra-muted'>Task " + id + ". " + (adding ? "The current images stay, and these files are added with them." : "The current images and their annotations are removed.") + "</p><input id='ra-dialog-images' type='file' accept='image/*,.zip,.jpg,.jpeg,.png,.webp,.bmp,.tif,.tiff' multiple/><div class='ra-row'><button class='ra-btn primary' id='ra-dialog-go' type='button'>" + (adding ? "Add to current images" : "Replace current images") + "</button><button class='ra-btn' id='ra-dialog-cancel' type='button'>Cancel</button></div></div>";
    document.body.appendChild(box);
    box.querySelector("#ra-dialog-cancel").onclick = function () { box.remove(); };
    box.querySelector("#ra-dialog-go").onclick = function () {
      var fileInput = document.getElementById("ra-dialog-images");
      var files = fileInput && fileInput.files ? Array.prototype.slice.call(fileInput.files) : [];
      if (!files.length) {
        exportNote("Choose the image files first.");
        return;
      }
      box.remove();
      updateTaskImages(mode, files, id);
    };
  }

  function imageMenuItems(sample, taskId) {
    var add = menuItem("Add images", "ra-menu-item ra-images-add", sample);
    var replace = menuItem("Replace images", "ra-menu-item ra-images-replace", sample);
    add.onclick = function (event) {
      event.preventDefault();
      event.stopPropagation();
      openImageDialog("add", contextFromLocation().taskId || pending.taskId || taskId);
    };
    replace.onclick = function (event) {
      event.preventDefault();
      event.stopPropagation();
      openImageDialog("replace", contextFromLocation().taskId || pending.taskId || taskId);
    };
    return [add, replace];
  }

  function insertAfterAutomaticAnnotation() {
    var items = document.querySelectorAll(".ant-dropdown-menu-item, .ant-menu-item");
    Array.prototype.forEach.call(items, function (item) {
      if (item.classList.contains("ra-menu-item")) return;
      var text = (item.textContent || "").replace(/\s+/g, " ").trim();
      if (text.indexOf("Automatic annotation") === -1) return;
      var taskId = pending.taskId || contextFromLocation().taskId;
      var anchor = item.nextElementSibling && item.nextElementSibling.classList.contains("ra-task-menu")
        ? item.nextElementSibling
        : null;
      if (!anchor) {
        anchor = menuItem("Requirement agent", "ra-menu-item ra-task-menu", item);
        anchor.onclick = function (event) {
          event.preventDefault();
          event.stopPropagation();
          openPanel({ taskId: pending.taskId || contextFromLocation().taskId });
        };
        item.insertAdjacentElement("afterend", anchor);
      }
      if (anchor.nextElementSibling && anchor.nextElementSibling.classList.contains("ra-images-add")) return;
      var pair = imageMenuItems(item, taskId);
      anchor.insertAdjacentElement("afterend", pair[0]);
      pair[0].insertAdjacentElement("afterend", pair[1]);
    });
  }

  function insertAnnotationImageActions() {
    var menu = document.querySelector(".cvat-annotation-menu");
    if (!menu || menu.querySelector(".ra-images-add")) return;
    var anchor = null;
    Array.prototype.forEach.call(menu.querySelectorAll(".ant-dropdown-menu-item"), function (item) {
      var text = (item.textContent || "").replace(/\s+/g, " ").trim();
      if (text === "Open the task") anchor = item;
    });
    if (!anchor) return;
    var pair = imageMenuItems(anchor, contextFromLocation().taskId);
    anchor.insertAdjacentElement("afterend", pair[0]);
    pair[0].insertAdjacentElement("afterend", pair[1]);
  }

  function insertProjectMenu() {
    var items = document.querySelectorAll(".ant-dropdown-menu-item, .ant-menu-item");
    Array.prototype.forEach.call(items, function (item) {
      var text = (item.textContent || "").replace(/\s+/g, " ").trim();
      if (text.indexOf("Backup Project") === -1) return;
      if (item.nextElementSibling && item.nextElementSibling.classList.contains("ra-project-menu")) return;
      var extra = menuItem("Requirement agent", "ra-menu-item ra-project-menu", item);
      extra.onclick = function (event) {
        event.preventDefault();
        event.stopPropagation();
        var fromUrl = contextFromLocation();
        openPanel({ projectId: fromUrl.projectId, projectName: pending.projectName });
      };
      item.insertAdjacentElement("afterend", extra);
    });
  }

  function insertAiOption() {
    var popover = document.querySelector(".cvat-tools-control-popover-content");
    if (!popover || popover.querySelector(".ra-ai-option")) return;
    var button = document.createElement("button");
    button.type = "button";
    button.className = "ra-btn ra-ai-option";
    button.textContent = "Requirement agent";
    button.onclick = function (event) {
      event.preventDefault();
      event.stopPropagation();
      var fromUrl = contextFromLocation();
      openPanel({ taskId: fromUrl.taskId, projectId: fromUrl.projectId });
    };
    popover.appendChild(button);
  }

  function sidebarButton() {
    var sider = document.querySelector(".cvat-canvas-controls-sidebar");
    var button = document.getElementById("ra-sidebar-button");
    if (!sider) {
      if (button) button.remove();
      return;
    }
    if (!button) {
      button = document.createElement("button");
      button.id = "ra-sidebar-button";
      button.type = "button";
      button.title = "Requirement agent";
      button.innerHTML = "<svg width='16' height='16' viewBox='0 0 16 16' fill='none'><path d='M3 2.5h7l3 3V13.5H3v-11z' stroke='currentColor' stroke-width='1.4'/><path d='M10 2.5V6h3M5 8h6M5 10.5h4' stroke='currentColor' stroke-width='1.4'/></svg>";
      button.onclick = function () {
        var fromUrl = contextFromLocation();
        openPanel({ taskId: fromUrl.taskId, projectId: fromUrl.projectId });
      };
    }
    var anchor = sider.querySelector(".cvat-tools-control");
    var host = anchor && anchor.parentElement && anchor.parentElement !== sider ? anchor.parentElement : anchor;
    if (host) {
      if (button.previousElementSibling === host) return;
      host.insertAdjacentElement("afterend", button);
    } else if (button.parentElement !== sider) {
      sider.appendChild(button);
    }
  }

  function dashboardCard() {
    var page = document.querySelector(".cvat-projects-page");
    var existing = document.getElementById("ra-dashboard");
    if (!page) {
      if (existing) existing.remove();
      return;
    }
    if (existing) return;
    ensureStyle();
    var card = document.createElement("section");
    card.id = "ra-dashboard";
    card.innerHTML = "<h3>Requirement agent</h3><p class='ra-muted'>Each project keeps its own requirement document. The agent creates labels from that document, annotates the project's tasks, and learns only from the corrections in that project.</p><div class='ra-row'><button class='ra-btn primary' id='ra-dashboard-open' type='button'>Open agent</button></div>";
    var top = page.querySelector(".cvat-projects-page-top-bar");
    if (top) top.insertAdjacentElement("afterend", card);
    else page.insertBefore(card, page.firstChild);
    card.querySelector("#ra-dashboard-open").onclick = function () { openPanel({}); };
  }

  function taskBanner() {
    var page = document.querySelector(".cvat-task-page");
    var existing = document.getElementById("ra-task-banner");
    if (!page) {
      if (existing) existing.remove();
      return;
    }
    if (existing) return;
    ensureStyle();
    var fromUrl = contextFromLocation();
    var card = document.createElement("section");
    card.id = "ra-task-banner";
    card.innerHTML = "<h3>Requirement agent</h3><p class='ra-muted'>Listed in Actions, directly after Automatic annotation. It uses this task's project requirement and keeps learning on that project.</p><div class='ra-row'><button class='ra-btn primary' id='ra-task-open' type='button'>Open for this task</button></div>";
    page.insertBefore(card, page.firstChild);
    card.querySelector("#ra-task-open").onclick = function () { openPanel({ taskId: fromUrl.taskId }); };
  }

  function headerButton() {
    if (document.getElementById("ra-header-button")) return;
    var header = document.querySelector(".cvat-header");
    if (!header) return;
    ensureStyle();
    var button = document.createElement("button");
    button.id = "ra-header-button";
    button.type = "button";
    button.textContent = "Requirement";
    button.onclick = function () {
      var fromUrl = contextFromLocation();
      openPanel({ taskId: fromUrl.taskId, projectId: fromUrl.projectId });
    };
    var user = header.querySelector(".cvat-header-menu-user-dropdown");
    if (user) user.parentElement.insertBefore(button, user);
    else header.appendChild(button);
  }

  document.addEventListener("click", function (event) {
    var taskButton = event.target.closest && event.target.closest(".cvat-task-item-actions-button, .cvat-task-page-actions-button");
    if (taskButton) {
      var row = taskButton.closest(".cvat-tasks-list-item");
      var idNode = row && row.querySelector(".cvat-item-task-id");
      var match = idNode && (idNode.textContent || "").match(/#(\d+)/);
      pending.taskId = match ? Number(match[1]) : contextFromLocation().taskId;
    }
    var projectButton = event.target.closest && event.target.closest(".cvat-project-details-button");
    if (projectButton) {
      var card = projectButton.closest(".cvat-projects-project-item-card");
      var title = card && card.querySelector(".cvat-projects-project-item-title");
      pending.projectName = title ? title.textContent.trim() : "";
    }
  }, true);

  var oldPanel = document.getElementById("requirement-agent-panel");
  var oldButton = document.getElementById("requirement-agent-button");
  if (oldPanel) oldPanel.remove();
  if (oldButton) oldButton.remove();

  function exportNote(text) {
    var node = document.getElementById("ra-export-note");
    if (!text) {
      if (node) node.remove();
      return;
    }
    if (!node) {
      node = document.createElement("div");
      node.id = "ra-export-note";
      document.body.appendChild(node);
    }
    node.textContent = text;
  }

  function waitRequest(rqId) {
    var tries = 0;
    function poll() {
      return cvat("/api/requests/" + encodeURIComponent(rqId)).then(function (data) {
        if (data.status === "finished") return data;
        if (data.status === "failed") throw new Error(data.message || "CVAT export failed.");
        tries += 1;
        if (tries > 150) throw new Error("CVAT export is still running.");
        return new Promise(function (resolve) { setTimeout(resolve, 2000); }).then(poll);
      });
    }
    return poll();
  }

  var learnTimers = {};
  function scheduleLearn(jobId) {
    clearTimeout(learnTimers[jobId]);
    learnTimers[jobId] = setTimeout(function () { learnSavedJob(jobId); }, 1500);
  }

  function learnSavedJob(jobId) {
    cvat("/api/jobs/" + jobId).then(function (job) {
      if (!job.project_id) return null;
      return analyzer("/projects/" + job.project_id + "/jobs/" + jobId + "/learn", { method: "POST" });
    }).then(function (result) {
      var node = document.getElementById("ra-learning");
      if (!node || !result || result.skipped) return;
      node.textContent = "Learned " + (result.samples || 0) + " frame(s) from your edits. " + (result.rules || 0) + " correction(s) will be used on the next task.";
    }).catch(function (error) {
      var node = document.getElementById("ra-learning");
      if (node) node.textContent = error.message || "The correction could not be learned.";
    });
  }

  function downloadRequirementExport(kind, id) {
    var lookup = kind === "projects" ? Promise.resolve({ project_id: id }) : cvat("/api/" + kind + "/" + id);
    return lookup.then(function (resource) {
      var projectId = kind === "projects" ? id : resource.project_id;
      if (!projectId) return null;
      var query = kind === "tasks" ? "?task_id=" + id : kind === "jobs" ? "?job_id=" + id : "";
      exportNote("Preparing the format named in this project's requirement…");
      return token().then(function (value) {
        return fetch("/analyzer/projects/" + projectId + "/export" + query, {
          headers: { Authorization: "Bearer " + value },
        });
      });
    }).then(function (response) {
      if (!response) return;
      var type = response.headers.get("content-type") || "";
      if (!response.ok) {
        exportNote("The requirement format could not be created.");
        setTimeout(function () { exportNote(""); }, 4000);
        return;
      }
      if (type.indexOf("zip") === -1) {
        exportNote("");
        return;
      }
      var header = response.headers.get("content-disposition") || "";
      var match = /filename="?([^";]+)"?/.exec(header);
      var name = match ? match[1] : "requirement-format.zip";
      return response.blob().then(function (blob) {
        var link = document.createElement("a");
        link.href = URL.createObjectURL(blob);
        link.download = name;
        document.body.appendChild(link);
        link.click();
        link.remove();
        exportNote("Downloaded the requirement format: " + name);
        setTimeout(function () { exportNote(""); }, 5000);
      });
    }).catch(function () {
      exportNote("The requirement format could not be created.");
      setTimeout(function () { exportNote(""); }, 4000);
    });
  }

  function finishExport(kind, id, rqId) {
    var wait = rqId ? waitRequest(rqId) : Promise.resolve();
    wait.then(function () { return downloadRequirementExport(kind, id); }).catch(function () {});
  }

  function watchAnnotationTraffic() {
    if (window.__raTraffic) return;
    window.__raTraffic = true;
    function handle(method, url, responseText, ok) {
      if (!ok) return;
      var verb = String(method || "GET").toUpperCase();
      var address = String(url || "");
      var saved = address.match(/\/api\/jobs\/(\d+)\/annotations(?:\?|$)/);
      if (saved && (verb === "PUT" || verb === "PATCH")) scheduleLearn(saved[1]);
      var exported = address.match(/\/api\/(projects|tasks|jobs)\/(\d+)\/dataset\/export(?:\?|$)/);
      if (exported && verb === "POST") {
        var body = {};
        try { body = JSON.parse(responseText || "{}"); } catch (error) { body = {}; }
        finishExport(exported[1], Number(exported[2]), body.rq_id);
      }
    }
    var nativeFetch = window.fetch;
    window.fetch = function (input, init) {
      var result = nativeFetch.apply(this, arguments);
      try {
        var address = typeof input === "string" ? input : (input && input.url) || "";
        var method = (init && init.method) || (input && input.method) || "GET";
        result.then(function (response) {
          if (!response || !response.clone) return;
          response.clone().text().then(function (raw) {
            handle(method, address, raw, response.ok);
          }).catch(function () {});
        }).catch(function () {});
      } catch (error) {}
      return result;
    };
    var nativeOpen = XMLHttpRequest.prototype.open;
    var nativeSend = XMLHttpRequest.prototype.send;
    XMLHttpRequest.prototype.open = function (method, url) {
      this.__raMethod = method;
      this.__raUrl = url;
      return nativeOpen.apply(this, arguments);
    };
    XMLHttpRequest.prototype.send = function () {
      this.addEventListener("load", function () {
        handle(this.__raMethod, this.__raUrl, this.responseText, this.status >= 200 && this.status < 300);
      });
      return nativeSend.apply(this, arguments);
    };
  }

  watchAnnotationTraffic();
  var openJob = location.pathname.match(/\/jobs\/(\d+)/);
  if (openJob) learnSavedJob(openJob[1]);

  setInterval(function () {
    headerButton();
    dashboardCard();
    taskBanner();
    sidebarButton();
    insertAfterAutomaticAnnotation();
    insertAnnotationImageActions();
    insertProjectMenu();
    insertAiOption();
  }, 400);
})();
