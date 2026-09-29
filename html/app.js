"use strict";

const state = {
  scenarios: [],
  data: null,
  cutoff: 0,
  frame: 0,
  playing: false,
  playbackSpeed: 1,
  timer: null,
  roadMotion: [],
  motionFrame: null,
  lastMotionTime: null,
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const velocityColors = ["#d84a3a", "#efb949", "#a9c85f", "#38a877"];
const densityColors = ["#38a877", "#b4ca55", "#efb949", "#d84a3a"];
const FLOW_STEP = 31;
const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

function mix(a, b, amount) {
  const parse = (hex) => [1, 3, 5].map((i) => Number.parseInt(hex.slice(i, i + 2), 16));
  const left = parse(a);
  const right = parse(b);
  return `rgb(${left.map((value, i) => Math.round(value + (right[i] - value) * amount)).join(",")})`;
}

function colorFor(value, range, metric) {
  const stops = metric === "velocity" ? velocityColors : densityColors;
  const normalized = Math.max(0, Math.min(0.9999, (value - range[0]) / (range[1] - range[0] || 1)));
  const scaled = normalized * (stops.length - 1);
  const index = Math.floor(scaled);
  return mix(stops[index], stops[index + 1] ?? stops[index], scaled - index);
}

function formatClock(step) {
  const { metadata } = state.data;
  const totalSeconds = Math.round(metadata.startHour * 3600 + step * metadata.timeStepSeconds);
  const hour = Math.floor(totalSeconds / 3600) % 24;
  const minute = Math.floor(totalSeconds / 60) % 60;
  const second = totalSeconds % 60;
  return `${String(hour).padStart(2, "0")}:${String(minute).padStart(2, "0")}:${String(second).padStart(2, "0")}`;
}

function displayedRow(metric, time) {
  const source = time < state.cutoff ? state.data : state.data.baseline;
  return source[metric][time];
}

function showError(message) {
  $("#error-message").textContent = message;
  $("#error-panel").hidden = false;
  $("#loading-panel").hidden = true;
}

function setLoading(loading) {
  $("#loading-panel").hidden = !loading;
  if (loading) $("#error-panel").hidden = true;
}

function stopPlayback() {
  state.playing = false;
  if (state.timer) window.clearInterval(state.timer);
  state.timer = null;
  $("#play-button").textContent = "▶";
  $("#play-button").setAttribute("aria-label", "Play optimization sequence");
}

function startPlayback() {
  if (!state.data) return;
  stopPlayback();
  state.playing = true;
  $("#play-button").textContent = "Ⅱ";
  $("#play-button").setAttribute("aria-label", "Pause optimization sequence");
  state.timer = window.setInterval(() => {
    setCutoff(state.cutoff >= state.data.metadata.timeSteps ? 0 : state.cutoff + 1);
  }, Math.max(24, 180 / state.playbackSpeed));
}

function setCutoff(cutoff) {
  if (!state.data) return;
  const steps = state.data.metadata.timeSteps;
  state.cutoff = Math.max(0, Math.min(steps, cutoff));
  state.frame = state.cutoff === 0 ? 0 : Math.min(steps - 1, state.cutoff - 1);
  $("#timeline").value = state.cutoff;
  $("#road-time").textContent = formatClock(state.frame);
  $("#time-code").textContent = formatClock(state.cutoff);
  $("#step-code").textContent = state.cutoff === 0
    ? "BASELINE"
    : state.cutoff === steps
      ? "FULL OPTIMAL"
      : `OPTIMAL ${state.cutoff} / ${steps}`;
  updateRoad();
  updateStats();
  drawHeatmaps();
}

function updateRoad() {
  const { frame } = state;
  const velocities = displayedRow("velocity", frame);
  const densities = displayedRow("density", frame);
  $$(".road-segment").forEach((segment, index) => {
    const velocity = velocities[index];
    const density = densities[index];
    const motion = state.roadMotion[index];
    if (motion) {
      motion.targetVelocity = velocity;
      motion.targetDensity = density;
    }
    segment.title = `Segment ${index + 1}: ${velocity.toFixed(1)} km/h, ${density.toFixed(1)} veh/km/lane`;
  });
  const meanSpeed = velocities.reduce((sum, value) => sum + value, 0) / velocities.length;
  const meanDensity = densities.reduce((sum, value) => sum + value, 0) / densities.length;
  $("#current-speed").textContent = meanSpeed.toFixed(1);
  $("#current-density").textContent = meanDensity.toFixed(1);
}

function updateStats() {
  const { data, cutoff } = state;
  const { timeSteps, segments, lanes, segmentLengthKm, timeStepSeconds } = data.metadata;
  let velocitySum = 0;
  let weightedDensitySum = 0;
  let queueSum = 0;
  let peakDensity = 0;
  let peakQueue = 0;

  for (let time = 0; time < timeSteps; time += 1) {
    const source = time < cutoff ? data : data.baseline;
    const queue = source.queue[time];
    queueSum += queue;
    peakQueue = Math.max(peakQueue, queue);
    for (let segment = 0; segment < segments; segment += 1) {
      const velocity = source.velocity[time][segment];
      const density = source.density[time][segment];
      velocitySum += velocity;
      weightedDensitySum += density * lanes[segment];
      peakDensity = Math.max(peakDensity, density);
    }
  }

  const travelTime = timeStepSeconds / 3600 * (
    segmentLengthKm * weightedDensitySum + queueSum
  );
  $("#travel-time").textContent = travelTime.toFixed(2);
  $("#mean-velocity").textContent = (velocitySum / (timeSteps * segments)).toFixed(1);
  $("#peak-density").textContent = peakDensity.toFixed(1);
  $("#peak-queue").textContent = peakQueue.toFixed(1);
}

function animateRoad(timestamp) {
  const elapsed = state.lastMotionTime === null ? 0 : Math.min((timestamp - state.lastMotionTime) / 1000, 0.1);
  state.lastMotionTime = timestamp;
  const smoothing = 1 - Math.exp(-elapsed / 0.45);
  const velocityRange = state.data?.scales.velocity;
  const densityRange = state.data?.scales.density;

  if (velocityRange && densityRange) {
    state.roadMotion.forEach((motion) => {
      motion.velocity += (motion.targetVelocity - motion.velocity) * smoothing;
      motion.density += (motion.targetDensity - motion.density) * smoothing;
      if (!prefersReducedMotion) {
        motion.phase = (motion.phase + Math.max(0, motion.velocity) * 0.52 * elapsed) % FLOW_STEP;
      }

      const densityLevel = Math.max(0, Math.min(1,
        (motion.density - densityRange[0]) / (densityRange[1] - densityRange[0] || 1)
      ));
      motion.element.style.setProperty("--flow-offset", `${motion.phase}px`);
      motion.element.style.setProperty("--flow-color", colorFor(motion.velocity, velocityRange, "velocity"));
      motion.element.style.setProperty("--density-color", colorFor(motion.density, densityRange, "density"));
      motion.element.style.setProperty("--density-level", `${densityLevel * 100}%`);
    });
  }

  state.motionFrame = window.requestAnimationFrame(animateRoad);
}

function ensureRoadAnimation() {
  if (state.motionFrame === null) state.motionFrame = window.requestAnimationFrame(animateRoad);
}

function buildRoad() {
  const road = $("#road-surface");
  road.replaceChildren();
  state.roadMotion = [];
  for (let segment = 0; segment < state.data.metadata.segments; segment += 1) {
    const element = document.createElement("div");
    element.className = "road-segment";

    const densityFill = document.createElement("div");
    densityFill.className = "density-fill";
    element.append(densityFill);

    const pattern = document.createElement("div");
    pattern.className = "flow-pattern";
    const track = document.createElement("div");
    track.className = "flow-track";
    for (let marker = 0; marker < 48; marker += 1) {
      const chevron = document.createElement("i");
      chevron.className = "flow-chevron";
      track.append(chevron);
    }
    pattern.append(track);
    element.append(pattern);

    const label = document.createElement("span");
    label.className = "segment-number";
    label.textContent = segment + 1;
    element.append(label);
    road.append(element);

    const velocity = displayedRow("velocity", state.frame)[segment];
    const density = displayedRow("density", state.frame)[segment];
    state.roadMotion.push({
      element,
      velocity,
      targetVelocity: velocity,
      density,
      targetDensity: density,
      phase: 0,
    });
  }
  ensureRoadAnimation();
}

function drawHeatmap(canvas, optimized, baseline, metric, range) {
  if (!optimized.length) return;
  const rect = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  const height = 270;
  canvas.width = Math.max(1, Math.floor(rect.width * ratio));
  canvas.height = Math.floor(height * ratio);
  const context = canvas.getContext("2d");
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, rect.width, height);

  const margin = { left: 47, right: 14, top: 12, bottom: 34 };
  const plotWidth = rect.width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const timeCount = optimized.length;
  const segmentCount = optimized[0].length;
  const cellWidth = plotWidth / timeCount;
  const cellHeight = plotHeight / segmentCount;

  for (let time = 0; time < timeCount; time += 1) {
    const row = time < state.cutoff ? optimized[time] : baseline[time];
    for (let segment = 0; segment < segmentCount; segment += 1) {
      context.fillStyle = colorFor(row[segment], range, metric);
      const y = margin.top + (segmentCount - segment - 1) * cellHeight;
      context.fillRect(margin.left + time * cellWidth, y, Math.ceil(cellWidth) + 0.5, Math.ceil(cellHeight) + 0.5);
    }
  }

  if (state.cutoff > 0 && state.cutoff < timeCount) {
    context.strokeStyle = "#000";
    context.lineWidth = 4;
    const dividerX = margin.left + state.cutoff / timeCount * plotWidth;
    context.beginPath();
    context.moveTo(dividerX, margin.top);
    context.lineTo(dividerX, margin.top + plotHeight);
    context.stroke();
  }

  context.fillStyle = "#81939b";
  context.font = "11px ui-monospace, SFMono-Regular, Menlo, monospace";
  context.textAlign = "center";
  for (let tick = 0; tick <= 4; tick += 1) {
    const x = margin.left + plotWidth * tick / 4;
    const minutes = timeCount * state.data.metadata.timeStepSeconds / 60 * tick / 4;
    context.fillText(`${minutes.toFixed(0)}m`, x, height - 12);
  }
  context.textAlign = "right";
  for (let tick = 0; tick <= 4; tick += 1) {
    const y = margin.top + plotHeight - plotHeight * tick / 4 + 4;
    const distance = segmentCount * state.data.metadata.segmentLengthKm * tick / 4;
    context.fillText(distance.toFixed(1), margin.left - 8, y);
  }
  context.save();
  context.translate(12, margin.top + plotHeight / 2);
  context.rotate(-Math.PI / 2);
  context.textAlign = "center";
  context.fillText("DISTANCE · KM", 0, 0);
  context.restore();
}

function drawHeatmaps() {
  if (!state.data) return;
  drawHeatmap($("#velocity-heatmap"), state.data.velocity, state.data.baseline.velocity, "velocity", state.data.scales.velocity);
  drawHeatmap($("#density-heatmap"), state.data.density, state.data.baseline.density, "density", state.data.scales.density);
}

function cutoffFromPointer(canvas, clientX) {
  const rect = canvas.getBoundingClientRect();
  const left = 47;
  const right = 14;
  const x = Math.max(0, Math.min(rect.width - left - right, clientX - rect.left - left));
  return Math.round(x / (rect.width - left - right) * state.data.metadata.timeSteps);
}

function renderScenario() {
  const { data } = state;
  state.cutoff = 0;
  state.frame = 0;
  buildRoad();
  $("#visualization").hidden = false;
  $("#road-end").textContent = `Segment ${String(data.metadata.segments).padStart(2, "0")} · ${(data.metadata.segments * data.metadata.segmentLengthKm).toFixed(1)} km`;
  $("#timeline").max = data.metadata.timeSteps;
  $("#velocity-max").textContent = `${data.scales.velocity[1].toFixed(0)} km/h`;
  $("#density-max").textContent = `${data.scales.density[1].toFixed(0)} veh/km/lane`;
  $("#source-path").textContent = `Source · results/${data.scenario.id}`;
  $("#resolution").textContent = `${data.metadata.segments} segments · ${data.metadata.timeStepSeconds.toFixed(0)}-second resolution`;
  setCutoff(0);
}

async function loadScenario(id) {
  stopPlayback();
  setLoading(true);
  try {
    const response = await fetch(`/api/scenario?id=${encodeURIComponent(id)}`);
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "This scenario could not be simulated.");
    state.data = payload;
    renderScenario();
    setLoading(false);
  } catch (error) {
    showError(error.message);
  }
}

async function initialize() {
  try {
    const response = await fetch("/api/scenarios");
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "The scenario reader is unavailable.");
    state.scenarios = payload.scenarios;
    if (!state.scenarios.length) throw new Error("No compatible result scenarios were found.");

    const select = $("#scenario-select");
    select.replaceChildren(...state.scenarios.map((scenario) => {
      const option = document.createElement("option");
      option.value = scenario.id;
      option.textContent = scenario.label;
      return option;
    }));
    select.disabled = false;
    updateScenarioMeta(state.scenarios[0]);
    await loadScenario(state.scenarios[0].id);
  } catch (error) {
    showError(error.message);
  }
}

function updateScenarioMeta(scenario) {
  const container = $("#scenario-meta");
  container.replaceChildren(...[scenario.date, scenario.calibration, scenario.variant || "standard run"].map((value) => {
    const span = document.createElement("span");
    span.textContent = value;
    return span;
  }));
}

$("#scenario-select").addEventListener("change", (event) => {
  const scenario = state.scenarios.find((item) => item.id === event.target.value);
  if (scenario) updateScenarioMeta(scenario);
  loadScenario(event.target.value);
});

$("#play-button").addEventListener("click", () => state.playing ? stopPlayback() : startPlayback());
$("#timeline").addEventListener("input", (event) => setCutoff(Number(event.target.value)));
$("#playback-speed").addEventListener("change", (event) => {
  state.playbackSpeed = Number(event.target.value);
  if (state.playing) startPlayback();
});

for (const id of ["velocity-heatmap", "density-heatmap"]) {
  const canvas = $(`#${id}`);
  canvas.addEventListener("click", (event) => setCutoff(cutoffFromPointer(canvas, event.clientX)));
  canvas.addEventListener("keydown", (event) => {
    if (event.key === "ArrowLeft") setCutoff(state.cutoff - 1);
    if (event.key === "ArrowRight") setCutoff(state.cutoff + 1);
  });
}

new ResizeObserver(drawHeatmaps).observe($(".diagrams"));
initialize();
