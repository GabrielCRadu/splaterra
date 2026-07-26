async function apiGet(path) {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`GET ${path} failed: ${response.status}`);
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
    throw new Error(`POST ${path} failed: ${response.status}`);
  }
  return response.json();
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
