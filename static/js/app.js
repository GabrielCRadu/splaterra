// Single source of truth for stage-name/status Romanian labels - previously
// duplicated separately in index.html and job.html, which let index.html's
// copy silently drift out of sync (it was missing "training"/"export",
// leaking the raw English stage name into an otherwise all-Romanian pill).
const STAGE_LABELS = {
  ingestion: "Ingestie",
  frame_extraction: "Extragere cadre",
  alignment: "Alignment",
  recommendation: "Recomandare",
  training: "Antrenare",
  export: "Export",
};

const STATUS_LABELS = {
  running: "in curs",
  done: "gata",
  failed: "esuat",
};

async function _errorMessage(response, method, path) {
  // FastAPI's HTTPException body is {"detail": "..."} - surface that real
  // message (e.g. "Frame extraction is already running for this job")
  // instead of a bare status code, so a caught error is actually useful to
  // show the user rather than just "POST ... failed: 409".
  try {
    const data = await response.json();
    if (data && typeof data.detail === "string") return data.detail;
  } catch (err) {
    // response body wasn't JSON - fall through to the generic message
  }
  return `${method} ${path} failed: ${response.status}`;
}

async function apiGet(path) {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(await _errorMessage(response, "GET", path));
  }
  return response.json();
}

async function apiPost(path, body) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  if (!response.ok) {
    throw new Error(await _errorMessage(response, "POST", path));
  }
  return response.json();
}

async function apiDelete(path) {
  const response = await fetch(path, { method: "DELETE" });
  if (!response.ok) {
    throw new Error(await _errorMessage(response, "DELETE", path));
  }
}

async function runAction(button, fn) {
  // Shared by every stage-action button (extract/align/train/export/cancel/
  // import/upload): prevents a double-tap or impatient double-click from
  // firing the request twice before the first response/re-render lands, and
  // surfaces a failure (e.g. the backend's 409 "already running" guard)
  // instead of silently swallowing it - previously an unhandled rejection
  // left the button clickable with zero feedback.
  if (button.disabled) return;
  button.disabled = true;
  try {
    await fn();
  } catch (err) {
    alert(err.message || String(err));
  } finally {
    button.disabled = false;
  }
}

function jobIdFromQuery() {
  return new URLSearchParams(window.location.search).get("id");
}

function escapeHtml(value) {
  const div = document.createElement("div");
  div.textContent = String(value);
  return div.innerHTML;
}

function pollStages(jobId, onUpdate, intervalMs) {
  intervalMs = intervalMs || 2000;
  let stopped = false;

  async function tick() {
    if (stopped) return;
    try {
      const stages = await apiGet(`/jobs/${jobId}/stages`);
      onUpdate(stages);
    } catch (err) {
      console.error("poll failed", err);
    }
    if (!stopped) {
      setTimeout(tick, intervalMs);
    }
  }

  tick();
  return function stop() {
    stopped = true;
  };
}

const UPLOAD_CHUNK_SIZE = 8 * 1024 * 1024;

async function uploadFileToJob(jobId, file, onProgress) {
  const totalChunks = Math.ceil(file.size / UPLOAD_CHUNK_SIZE);
  const { upload_id: uploadId } = await apiPost(`/jobs/${jobId}/uploads`, {
    filename: file.name,
    total_chunks: totalChunks,
  });

  for (let index = 0; index < totalChunks; index += 1) {
    const start = index * UPLOAD_CHUNK_SIZE;
    const chunk = file.slice(start, start + UPLOAD_CHUNK_SIZE);
    const response = await fetch(
      `/jobs/${jobId}/uploads/${uploadId}/chunks/${index}`,
      { method: "PUT", body: chunk }
    );
    if (!response.ok) {
      throw new Error(`Chunk ${index} failed: ${response.status}`);
    }
    if (onProgress) {
      onProgress(index + 1, totalChunks);
    }
  }

  await apiPost(`/jobs/${jobId}/uploads/${uploadId}/complete`, {});
}

function initCustomSelect(select) {
  // Native <select> popups are rendered by the OS/browser and can't be
  // reskinned reliably across browsers - this builds a themed replacement UI
  // while keeping the original <select> (hidden) as the source of truth, so
  // any existing code reading/writing select.value keeps working unchanged.
  select.style.display = "none";

  const wrapper = document.createElement("div");
  wrapper.className = "custom-select";

  const trigger = document.createElement("button");
  trigger.type = "button";
  trigger.className = "custom-select-trigger";
  trigger.innerHTML = '<span class="custom-select-value"></span><span class="custom-select-arrow">&#9662;</span>';

  const panel = document.createElement("div");
  panel.className = "custom-select-panel";
  // Positioned as a fixed-position portal on document.body (not a child of
  // wrapper) so it escapes any ancestor with overflow:hidden - e.g. the
  // story-view carousel's .story-viewport, which needs overflow:hidden for
  // its own paging mechanism and would otherwise clip the open dropdown.

  function labelFor(value) {
    const match = Array.from(select.options).find((opt) => opt.value === value);
    return match ? match.textContent : "";
  }

  function positionPanel() {
    const rect = trigger.getBoundingClientRect();
    panel.style.left = `${rect.left}px`;
    panel.style.top = `${rect.bottom + 6}px`;
    panel.style.width = `${rect.width}px`;
  }

  function close() {
    panel.classList.remove("open");
    trigger.classList.remove("open");
  }

  function renderOptions() {
    panel.innerHTML = "";
    Array.from(select.options).forEach((opt) => {
      const item = document.createElement("button");
      item.type = "button";
      item.className = "custom-select-option";
      if (opt.value === select.value) {
        item.classList.add("selected");
      }
      item.textContent = opt.textContent;
      item.addEventListener("click", () => {
        select.value = opt.value;
        select.dispatchEvent(new Event("change"));
        sync();
        close();
      });
      panel.appendChild(item);
    });
  }

  function open() {
    renderOptions();
    positionPanel();
    panel.classList.add("open");
    trigger.classList.add("open");
  }

  function sync() {
    trigger.querySelector(".custom-select-value").textContent = labelFor(select.value);
  }

  trigger.addEventListener("click", (event) => {
    event.stopPropagation();
    if (panel.classList.contains("open")) {
      close();
    } else {
      open();
    }
  });

  document.addEventListener("click", (event) => {
    if (!wrapper.contains(event.target) && !panel.contains(event.target)) {
      close();
    }
  });

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      close();
    }
  });

  window.addEventListener(
    "scroll",
    () => {
      if (panel.classList.contains("open")) positionPanel();
    },
    true
  );
  window.addEventListener("resize", () => {
    if (panel.classList.contains("open")) positionPanel();
  });

  sync();
  wrapper.appendChild(trigger);
  select.insertAdjacentElement("afterend", wrapper);
  document.body.appendChild(panel);

  return { sync };
}

function pollUsage(onUpdate, intervalMs) {
  intervalMs = intervalMs || 2000;
  let stopped = false;

  async function tick() {
    if (stopped) return;
    try {
      const usage = await apiGet("/hardware/usage");
      onUpdate(usage);
    } catch (err) {
      console.error("usage poll failed", err);
    }
    if (!stopped) {
      setTimeout(tick, intervalMs);
    }
  }

  tick();
  return function stop() {
    stopped = true;
  };
}
