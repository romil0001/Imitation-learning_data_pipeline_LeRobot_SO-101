const stateEl = document.getElementById("state");
const timerEl = document.getElementById("timer");
const omenEl = document.getElementById("omen-pill");
const hwEl = document.getElementById("hw");
const logEl = document.getElementById("log");
const camsEl = document.getElementById("cams");
const camMeta = document.getElementById("cam-meta");
const toastEl = document.getElementById("toast");
const jointsEl = document.getElementById("joints");
const jointsMeta = document.getElementById("joints-meta");
const jointsPanel = document.querySelector(".panel.joints");
const liveMain = document.getElementById("live-main");
const browseMain = document.getElementById("browse-main");
const livePills = document.getElementById("live-pills");
const datasetSelect = document.getElementById("dataset-select");
const tagFilterEl = document.getElementById("tag-filter");
const tagChipsEl = document.getElementById("tag-chips");
const episodeTbody = document.getElementById("episode-tbody");
const compareToggle = document.getElementById("compare-toggle");
const sharedScrub = document.getElementById("shared-scrub");
const compareMeta = document.getElementById("compare-meta");
const paneA = document.getElementById("pane-a");
const paneB = document.getElementById("pane-b");

let toastTimer = 0;
let jointRows = null;
let streamKey = "";
let mode = "live";
let datasetsCache = [];
let datasetDetail = null;
let selectedIndex = null;
let compareA = null;
let compareB = null;
const panes = { a: null, b: null };

function toast(msg) {
  toastEl.hidden = false;
  toastEl.textContent = msg;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    toastEl.hidden = true;
  }, 2800);
}

function fmtTime(s) {
  const t = Math.max(0, Number(s) || 0);
  const m = Math.floor(t / 60);
  const r = t - m * 60;
  return `${String(m).padStart(2, "0")}:${r.toFixed(1).padStart(4, "0")}`;
}

function clsFor(state) {
  const x = (state || "IDLE").toLowerCase();
  if (x === "recording") return "recording";
  if (x === "teleop") return "teleop";
  if (x === "disconnected") return "disconnected";
  return "idle";
}

function row(k, v, ok) {
  const li = document.createElement("li");
  const vk = ok === true ? "ok" : ok === false ? "bad" : "";
  li.innerHTML = `<span class="k">${k}</span><span class="v ${vk}">${v}</span>`;
  return li;
}

function normPct(v) {
  const x = Number(v) || 0;
  const pct = ((x + 100) / 200) * 100;
  return Math.max(0, Math.min(100, pct));
}

function ensureJointRows(names) {
  if (jointRows && jointRows.length === names.length) return;
  jointsEl.replaceChildren();
  jointRows = names.map((name) => {
    const rowEl = document.createElement("div");
    rowEl.className = "joint-row";
    rowEl.innerHTML = `
      <span class="name">${name}</span>
      <div class="joint-bars">
        <div class="bar-track"><div class="bar-fill state"></div></div>
        <div class="bar-track"><div class="bar-fill action"></div></div>
      </div>
      <span class="vals">—</span>`;
    jointsEl.appendChild(rowEl);
    return {
      stateFill: rowEl.querySelector(".bar-fill.state"),
      actionFill: rowEl.querySelector(".bar-fill.action"),
      vals: rowEl.querySelector(".vals"),
    };
  });
}

function renderStreams(streams) {
  const key = (streams || []).map((s) => s.id).join("|");
  if (key === streamKey) {
    (streams || []).forEach((s) => {
      const slot = camsEl.querySelector(`[data-stream="${CSS.escape(s.id)}"]`);
      if (!slot) return;
      slot.classList.toggle("live", !!s.live);
      slot.classList.toggle("waiting", !s.live);
      const badge = slot.querySelector(".live-badge");
      if (badge) badge.textContent = s.live ? "live" : "offline";
    });
    return;
  }
  streamKey = key;
  camsEl.replaceChildren();
  (streams || []).forEach((s) => {
    const slot = document.createElement("div");
    slot.className = `cam-slot ${s.kind || "rgb"} ${s.live ? "live" : "waiting"}`;
    slot.dataset.stream = s.id;
    slot.innerHTML = `
      <div class="panel-h">
        <h1>${s.label || s.id}</h1>
        <span class="live-badge muted">${s.live ? "live" : "offline"}</span>
      </div>
      <div class="stage">
        <img src="${s.url}?t=${Date.now()}" alt="${s.label || s.id}"/>
      </div>`;
    camsEl.appendChild(slot);
  });
  if (!(streams || []).length) {
    const empty = document.createElement("div");
    empty.className = "cam-slot waiting";
    empty.innerHTML = `
      <div class="panel-h"><h1>no streams</h1></div>
      <div class="stage empty-hint">Set CAMERA_NAMES in camera.env</div>`;
    camsEl.appendChild(empty);
  }
}

async function refreshJoints() {
  if (mode !== "live") return;
  const r = await fetch("/api/joints");
  const j = await r.json();
  const names = j.names || [];
  if (!names.length) {
    jointsMeta.textContent = "waiting for record/eval…";
    jointsPanel.classList.add("stale");
    return;
  }
  ensureJointRows(names);
  const state = j.state || [];
  const action = j.action || [];
  jointRows.forEach((rowEl, i) => {
    const s = state[i];
    const a = action[i];
    rowEl.stateFill.style.width = `${normPct(s)}%`;
    rowEl.actionFill.style.width = `${normPct(a)}%`;
    rowEl.vals.textContent = `${Number(s).toFixed(1)} / ${Number(a).toFixed(1)}`;
  });
  if (j.fresh) {
    jointsMeta.textContent = "live";
    jointsPanel.classList.remove("stale");
  } else {
    jointsMeta.textContent = j.age_s != null ? `stale ${j.age_s}s` : "stale";
    jointsPanel.classList.add("stale");
  }
}

async function refresh() {
  if (mode !== "live") return;
  const r = await fetch("/api/status");
  const s = await r.json();
  const st = s.state || "IDLE";
  stateEl.textContent = st;
  stateEl.className = `pill ${clsFor(st)}`;
  timerEl.textContent = fmtTime(s.elapsed_s);
  const cam = s.camera || {};
  const streams = s.streams || [];
  renderStreams(streams);

  if (camMeta) {
    const nLive = (cam.live_ids || []).length;
    camMeta.textContent = cam.backend === "zmq"
      ? `${cam.host}:${cam.port} · ${nLive}/${streams.length || "?"} live · ${cam.fps ?? "—"} fps`
      : `opencv · ${cam.fps ?? "—"} fps`;
  }
  omenEl.textContent = cam.reachable ? "OMEN UP" : "OMEN DOWN";
  omenEl.className = `pill ${cam.reachable ? "ok" : "bad"}`;

  const arms = s.arms || {};
  const age = cam.last_frame_age_s == null ? "—" : `${cam.last_frame_age_s}s`;
  const depthNames = (cam.depth_names || []).join(", ") || "—";
  const depthLive = (cam.depth_live || []).map((id) => id.replace(/_depth$/, "")).join(", ");
  const rows = [
    row("Omen", cam.reachable ? `reachable :${cam.port || 5555}` : "not reachable", cam.reachable),
    row("Streams", `${(cam.live_ids || []).length}/${streams.length || 0} live`, (cam.live_ids || []).length > 0),
    row("Frame age", age, cam.last_frame_age_s != null && cam.last_frame_age_s < 2),
    row("FPS", cam.fps ?? "—", cam.fps > 1),
    row("Depth", cam.depth_ok ? (depthLive || depthNames) : "no depth yet", cam.depth_ok),
    row(
      "Follower",
      arms.follower_ok ? arms.follower_port : (arms.follower_port || "missing"),
      !!arms.follower_ok,
    ),
    row(
      "Leader",
      arms.leader_ok ? arms.leader_port : (arms.leader_port || "missing"),
      !!arms.leader_ok,
    ),
    row("Last session", s.last_banner || "—"),
  ];
  hwEl.replaceChildren(...rows);
}

function bindLogs() {
  const src = new EventSource("/api/log/stream");
  src.onmessage = (ev) => {
    let payload;
    try {
      payload = JSON.parse(ev.data);
    } catch {
      return;
    }
    if (payload.reset) {
      logEl.textContent = payload.chunk || "";
    } else {
      logEl.textContent += payload.chunk || "";
      if (logEl.textContent.length > 120000) {
        logEl.textContent = logEl.textContent.slice(-80000);
      }
    }
    logEl.scrollTop = logEl.scrollHeight;
  };
  src.onerror = () => {
    /* EventSource retries */
  };
}

const OP_KEYS = new Set(["n", "r", "q", "y", "p", "c", "s", "f", " "]);

async function sendOpKey(raw) {
  const key = raw === " " || raw === "Space" || raw === "space" ? " " : String(raw).toLowerCase();
  if (!OP_KEYS.has(key)) return;
  try {
    const r = await fetch("/api/key", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ key }),
    });
    const j = await r.json().catch(() => ({}));
    if (!r.ok) {
      toast(j.detail || `Key ${key} rejected`);
      return;
    }
    const label = key === " " ? "space" : key;
    if (j.listening) {
      toast(`Sent ${label}`);
    } else {
      toast(`Queued ${label} — start record/eval/dagger`);
    }
  } catch {
    toast("Could not send key");
  }
}

function handleDashboardKey(raw) {
  if (mode !== "live") return;
  const k = raw === " " ? " " : String(raw).toLowerCase();
  if (k === "d") {
    Promise.all([refresh(), refreshJoints()])
      .then(() => toast("Status refreshed (D)"))
      .catch(() => toast("Status check failed"));
    return;
  }
  if (k === "a") {
    toast("Start recording in a terminal — bash scripts/08_record.sh");
    return;
  }
  if (OP_KEYS.has(k)) {
    sendOpKey(k);
  }
}

function setMode(next) {
  mode = next;
  document.querySelectorAll(".mode-tab").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.mode === next);
  });
  liveMain.hidden = next !== "live";
  browseMain.hidden = next !== "browse";
  livePills.hidden = next !== "live";
  if (next === "browse") {
    loadDatasets().catch((e) => toast(String(e.message || e)));
  }
}

function filterTokens() {
  return (tagFilterEl.value || "")
    .split(/\s+/)
    .map((t) => t.trim().toLowerCase())
    .filter(Boolean);
}

function episodeMatches(ep, tokens) {
  if (!tokens.length) return true;
  const tags = (ep.tags || []).map((t) => String(t).toLowerCase());
  return tokens.every((tok) => tags.some((t) => t.includes(tok)));
}

function allTags(episodes) {
  const set = new Set();
  (episodes || []).forEach((ep) => (ep.tags || []).forEach((t) => set.add(t)));
  return [...set].sort();
}

function renderTagChips(episodes) {
  tagChipsEl.replaceChildren();
  allTags(episodes).forEach((tag) => {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "tag-chip";
    chip.textContent = tag;
    chip.addEventListener("click", () => {
      const cur = tagFilterEl.value.trim();
      tagFilterEl.value = cur ? `${cur} ${tag}` : tag;
      renderEpisodeTable();
    });
    tagChipsEl.appendChild(chip);
  });
}

function renderEpisodeTable() {
  if (!datasetDetail) {
    episodeTbody.replaceChildren();
    return;
  }
  const tokens = filterTokens();
  const rows = (datasetDetail.episodes || []).filter((ep) => episodeMatches(ep, tokens));
  episodeTbody.replaceChildren();
  rows.forEach((ep) => {
    const tr = document.createElement("tr");
    const active = selectedIndex === ep.index || compareA === ep.index || compareB === ep.index;
    if (active) tr.classList.add("selected");
    const mark =
      compareA === ep.index ? "A" : compareB === ep.index ? "B" : selectedIndex === ep.index ? "•" : "";
    tr.innerHTML = `
      <td class="mark-cell">${mark}</td>
      <td>${String(ep.index).padStart(6, "0")}</td>
      <td>${ep.num_frames}</td>
      <td>${Number(ep.duration_s).toFixed(1)}s</td>
      <td>${ep.source}${ep.intervention ? " *" : ""}</td>
      <td class="tags-cell">${(ep.tags || []).map((t) => `<span class="tag">${t}</span>`).join(" ")}</td>`;
    tr.addEventListener("click", () => onEpisodeClick(ep.index));
    episodeTbody.appendChild(tr);
  });
}

function updateCompareMeta() {
  if (!compareToggle.checked) {
    compareMeta.textContent = "Single episode view";
    return;
  }
  compareMeta.textContent = `A=${compareA == null ? "—" : String(compareA).padStart(6, "0")}  B=${compareB == null ? "—" : String(compareB).padStart(6, "0")}`;
}

async function onEpisodeClick(index) {
  if (compareToggle.checked) {
    if (compareA == null || (compareA != null && compareB != null)) {
      compareA = index;
      compareB = null;
    } else if (compareB == null && index !== compareA) {
      compareB = index;
    } else if (index === compareA) {
      compareA = compareB;
      compareB = null;
    }
    selectedIndex = compareA;
    updateCompareMeta();
    renderEpisodeTable();
    await loadComparePanes();
    return;
  }
  selectedIndex = index;
  compareA = index;
  compareB = null;
  updateCompareMeta();
  renderEpisodeTable();
  await loadPane("a", index);
  paneB.hidden = true;
  panes.b = null;
  paneB.replaceChildren();
}

async function loadDatasets() {
  const r = await fetch("/api/datasets");
  const j = await r.json();
  datasetsCache = j.datasets || [];
  const prev = datasetSelect.value;
  datasetSelect.replaceChildren();
  if (!datasetsCache.length) {
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = "(no datasets found)";
    datasetSelect.appendChild(opt);
    datasetDetail = null;
    renderEpisodeTable();
    paneA.innerHTML = `<div class="empty-browse">No datasets under data/datasets/</div>`;
    return;
  }
  datasetsCache.forEach((ds) => {
    const opt = document.createElement("option");
    opt.value = ds.path_name || ds.name;
    opt.textContent = `${ds.path_name || ds.name} · ${ds.episode_count} eps · ${ds.task || ""}`;
    datasetSelect.appendChild(opt);
  });
  if (prev && [...datasetSelect.options].some((o) => o.value === prev)) {
    datasetSelect.value = prev;
  }
  await loadDataset(datasetSelect.value);
}

async function loadDataset(name) {
  if (!name) return;
  const r = await fetch(`/api/datasets/${encodeURIComponent(name)}`);
  if (!r.ok) {
    const err = await r.json().catch(() => ({}));
    throw new Error(err.detail || `Failed to load ${name}`);
  }
  datasetDetail = await r.json();
  selectedIndex = null;
  compareA = null;
  compareB = null;
  panes.a = null;
  panes.b = null;
  paneA.replaceChildren();
  paneB.replaceChildren();
  paneB.hidden = true;
  renderTagChips(datasetDetail.episodes || []);
  renderEpisodeTable();
  updateCompareMeta();
  const first = (datasetDetail.episodes || [])[0];
  if (first) await onEpisodeClick(first.index);
}

function drawJointChart(canvas, joints, frame, fps) {
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#0a0c0f";
  ctx.fillRect(0, 0, w, h);
  if (!joints || !joints.state || !joints.state.length) return;
  const names = joints.joint_names || [];
  const nJ = (joints.state[0] || []).length;
  const rows = Math.max(1, nJ);
  const rowH = h / rows;
  let ymin = Infinity;
  let ymax = -Infinity;
  joints.state.forEach((s, i) => {
    const a = joints.action[i] || [];
    for (let j = 0; j < nJ; j += 1) {
      ymin = Math.min(ymin, s[j], a[j] ?? s[j]);
      ymax = Math.max(ymax, s[j], a[j] ?? s[j]);
    }
  });
  if (!(ymax > ymin)) {
    ymin -= 1;
    ymax += 1;
  }
  const pad = (ymax - ymin) * 0.08;
  ymin -= pad;
  ymax += pad;
  const n = joints.state.length;
  const indices = joints.indices || joints.state.map((_, i) => i);
  const xAt = (i) => (i / Math.max(1, n - 1)) * (w - 8) + 4;
  const yAt = (row, v) => row * rowH + 4 + (1 - (v - ymin) / (ymax - ymin)) * (rowH - 8);

  for (let j = 0; j < nJ; j += 1) {
    ctx.strokeStyle = "#1d232c";
    ctx.beginPath();
    ctx.moveTo(0, (j + 1) * rowH);
    ctx.lineTo(w, (j + 1) * rowH);
    ctx.stroke();
    ctx.fillStyle = "#8b93a2";
    ctx.font = "10px ui-monospace, Menlo, monospace";
    ctx.fillText(names[j] || `j${j}`, 6, j * rowH + 12);

    ctx.strokeStyle = "#7aa2ff";
    ctx.beginPath();
    joints.state.forEach((s, i) => {
      const x = xAt(i);
      const y = yAt(j, s[j]);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    ctx.strokeStyle = "#e0b45c";
    ctx.beginPath();
    joints.action.forEach((a, i) => {
      const x = xAt(i);
      const y = yAt(j, a[j]);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }

  if (frame != null && joints.num_frames > 0) {
    let best = 0;
    let bestDist = Infinity;
    indices.forEach((idx, i) => {
      const d = Math.abs(idx - frame);
      if (d < bestDist) {
        bestDist = d;
        best = i;
      }
    });
    const x = xAt(best);
    ctx.strokeStyle = "#ee5b5b";
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, h);
    ctx.stroke();
  }
}

function buildPaneShell(label, detail) {
  const root = document.createElement("div");
  root.className = "pane-inner";
  root.innerHTML = `
    <div class="panel-h">
      <h1>${label} · episode ${String(detail.index).padStart(6, "0")}</h1>
      <span class="muted">${detail.num_frames} frames · ${detail.fps} fps</span>
    </div>
    <div class="tag-row">${(detail.tags || []).map((t) => `<span class="tag">${t}</span>`).join("")}</div>
    <div class="file-chips"></div>
    <div class="browse-cams"></div>
    <div class="scrub-row">
      <button type="button" class="btn play-btn">Play</button>
      <input type="range" class="scrub" min="0" max="${Math.max(0, detail.num_frames - 1)}" value="0" />
      <span class="frame-label muted">0 / ${detail.num_frames}</span>
    </div>
    <canvas class="joint-canvas" width="900" height="360"></canvas>
    <div class="joint-legend browse-legend">
      <span class="lg state">state</span>
      <span class="lg action">action</span>
    </div>`;
  return root;
}

async function loadPane(which, index) {
  const name = datasetSelect.value;
  const paneEl = which === "a" ? paneA : paneB;
  const [detailR, jointsR] = await Promise.all([
    fetch(`/api/datasets/${encodeURIComponent(name)}/episodes/${index}`),
    fetch(`/api/datasets/${encodeURIComponent(name)}/episodes/${index}/joints`),
  ]);
  if (!detailR.ok || !jointsR.ok) {
    toast(`Failed to load episode ${index}`);
    return;
  }
  const detail = await detailR.json();
  const joints = await jointsR.json();
  paneEl.hidden = false;
  paneEl.replaceChildren();
  const shell = buildPaneShell(which.toUpperCase(), detail);
  paneEl.appendChild(shell);

  const chips = shell.querySelector(".file-chips");
  (detail.files || []).forEach((f) => {
    const span = document.createElement("span");
    span.className = "file-chip";
    const kb = Math.round((f.size || 0) / 1024);
    span.textContent = `${f.name} (${kb} KB)`;
    chips.appendChild(span);
  });

  const cams = shell.querySelector(".browse-cams");
  const videos = [];
  (detail.camera_names || []).forEach((cam) => {
    if (!(detail.video_urls || {})[cam]) return;
    const slot = document.createElement("div");
    slot.className = "browse-cam";
    slot.innerHTML = `<div class="panel-h"><h1>${cam}</h1></div><video src="${detail.video_urls[cam]}" preload="metadata" playsinline></video>`;
    cams.appendChild(slot);
    videos.push(slot.querySelector("video"));
  });
  const depthImgs = {};
  (detail.depth_cameras || []).forEach((cam) => {
    const slot = document.createElement("div");
    slot.className = "browse-cam depth";
    slot.innerHTML = `<div class="panel-h"><h1>${cam} depth</h1></div><img alt="${cam} depth"/>`;
    cams.appendChild(slot);
    depthImgs[cam] = slot.querySelector("img");
  });

  const scrub = shell.querySelector(".scrub");
  const frameLabel = shell.querySelector(".frame-label");
  const playBtn = shell.querySelector(".play-btn");
  const canvas = shell.querySelector(".joint-canvas");
  let frame = 0;
  let playing = false;
  let raf = 0;
  let syncing = false;

  const state = {
    which,
    detail,
    joints,
    videos,
    depthImgs,
    scrub,
    frameLabel,
    playBtn,
    canvas,
    get frame() {
      return frame;
    },
    setFrame,
    play,
    pause,
  };

  function setDepth(t) {
    Object.entries(depthImgs).forEach(([cam, img]) => {
      img.src = `/api/datasets/${encodeURIComponent(name)}/episodes/${index}/depth/${encodeURIComponent(cam)}/frame/${t}?v=${Date.now() % 100000}`;
    });
  }

  function setFrame(t, { fromVideo = false, fromShared = false } = {}) {
    const maxF = Math.max(0, detail.num_frames - 1);
    frame = Math.max(0, Math.min(maxF, Math.round(t)));
    scrub.value = String(frame);
    frameLabel.textContent = `${frame} / ${detail.num_frames}`;
    drawJointChart(canvas, joints, frame, detail.fps);
    setDepth(frame);
    if (!fromVideo && videos.length) {
      syncing = true;
      const tSec = frame / Math.max(1, detail.fps);
      videos.forEach((v) => {
        if (Math.abs((v.currentTime || 0) - tSec) > 1 / Math.max(1, detail.fps)) {
          try {
            v.currentTime = tSec;
          } catch {
            /* ignore seek errors while metadata loads */
          }
        }
      });
      syncing = false;
    }
    if (sharedScrub.checked && compareToggle.checked && !fromShared) {
      const other = which === "a" ? panes.b : panes.a;
      if (other && other.frame !== frame) other.setFrame(frame, { fromShared: true });
    }
  }

  function tick() {
    if (!playing) return;
    const v = videos[0];
    if (v && !v.paused && !v.ended) {
      const t = Math.round((v.currentTime || 0) * detail.fps);
      if (!syncing) setFrame(t, { fromVideo: true });
      raf = requestAnimationFrame(tick);
      return;
    }
    playing = false;
    playBtn.textContent = "Play";
  }

  function play() {
    playing = true;
    playBtn.textContent = "Pause";
    videos.forEach((v) => v.play().catch(() => {}));
    raf = requestAnimationFrame(tick);
  }

  function pause() {
    playing = false;
    playBtn.textContent = "Play";
    cancelAnimationFrame(raf);
    videos.forEach((v) => v.pause());
  }

  scrub.addEventListener("input", () => {
    pause();
    setFrame(Number(scrub.value));
  });
  playBtn.addEventListener("click", () => {
    if (playing) pause();
    else play();
  });
  videos.forEach((v) => {
    v.addEventListener("timeupdate", () => {
      if (syncing || !playing) return;
      setFrame(Math.round((v.currentTime || 0) * detail.fps), { fromVideo: true });
    });
  });

  panes[which] = state;
  setFrame(0);
}

async function loadComparePanes() {
  if (compareA == null) return;
  await loadPane("a", compareA);
  if (compareB != null) {
    paneB.hidden = false;
    await loadPane("b", compareB);
  } else {
    paneB.hidden = true;
    panes.b = null;
    paneB.replaceChildren();
  }
}

document.querySelectorAll(".mode-tab").forEach((btn) => {
  btn.addEventListener("click", () => setMode(btn.dataset.mode || "live"));
});

document.getElementById("browse-refresh").addEventListener("click", () => {
  loadDatasets().catch((e) => toast(String(e.message || e)));
});

datasetSelect.addEventListener("change", () => {
  loadDataset(datasetSelect.value).catch((e) => toast(String(e.message || e)));
});

tagFilterEl.addEventListener("input", () => renderEpisodeTable());

compareToggle.addEventListener("change", () => {
  updateCompareMeta();
  if (!compareToggle.checked) {
    compareB = null;
    paneB.hidden = true;
    panes.b = null;
    paneB.replaceChildren();
    if (selectedIndex != null) {
      loadPane("a", selectedIndex);
    }
  }
  renderEpisodeTable();
});

document.addEventListener("keydown", (e) => {
  const tag = (e.target && e.target.tagName) || "";
  if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  const raw = e.key === " " ? " " : e.key;
  const k = raw === " " ? " " : raw.toLowerCase();
  if (k === "d" || k === "a" || OP_KEYS.has(k)) {
    e.preventDefault();
    handleDashboardKey(raw);
  }
});

document.querySelectorAll("kbd[data-key]").forEach((el) => {
  el.addEventListener("click", () => {
    handleDashboardKey(el.getAttribute("data-key") || "");
  });
  el.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      e.stopPropagation();
      handleDashboardKey(el.getAttribute("data-key") || "");
    }
  });
});

bindLogs();
refresh();
refreshJoints().catch(() => {});
setInterval(() => {
  refresh().catch(() => {});
}, 1000);
setInterval(() => {
  refreshJoints().catch(() => {});
}, 100);
