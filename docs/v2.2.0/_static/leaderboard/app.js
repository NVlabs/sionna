// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import { NVIDIA_PALETTE } from "./color_palette.js";

const PALETTE_GROUPS = [
  "Red Values",
  "Blue Values",
  "Green Values",
  "Gray Values",
  "Yellow and Orange Values",
  "Purple Values",
  "Magenta Values",
  "Teal Values",
];

const PALETTE_GROUP_BY_NAME = Object.fromEntries(
  NVIDIA_PALETTE.map((group) => [group.group, group])
);

const METRIC_LABELS = {
  nmse: "NMSE",
  ber: "BER",
  bler: "BLER",
};

const SOURCE_INFO = {
  builtin: {
    order: 0,
    title: "Builtin",
    short: "Builtin",
    reference: "Sionna built-in channel estimators.",
  },
  balevi2020: {
    order: 10,
    title: "Untrained Neural Network",
    short: "Balevi 2020",
    reference: `E. Balevi, A. Doshi, and J. G. Andrews, "Massive MIMO Channel Estimation With an Untrained Deep Neural Network," IEEE Transactions on Wireless Communications, vol. 19, no. 3, pp. 2079-2090, March 2020.`,
  },
  baur2022: {
    order: 9,
    title: "Variational Auto-Encoder",
    short: "Baur 2022",
    reference: `M. Baur, B. Fesl, M. Koller, and W. Utschick, "Variational Autoencoder Leveraged MMSE Channel Estimation," 2022 56th Asilomar Conference on Signals, Systems, and Computers.`,
  },
  coleri2002: {
    order: 1,
    title: "Interpolation",
    short: "Coleri 2002",
    reference: `S. Coleri, M. Ergen, A. Puri, and A. Bahai, "Channel Estimation Techniques Based on Pilot Arrangement in OFDM Systems," IEEE Transactions on Broadcasting, vol. 48, no. 3, pp. 223-229, 2002.`,
  },
  delfeld2025: {
    order: 4,
    title: "RKHS Regularization",
    short: "Delfeld 2025",
    reference: `J. Delfeld, G. Marti, and C. Dick, "Sparse MIMO-OFDM Channel Estimation via RKHS Regularization," arXiv:2511.20082v1, 2025.`,
  },
  edfors1996: {
    order: 2,
    title: "Delay Truncation",
    short: "Edfors 1996",
    reference: `O. Edfors, M. Sandell, J.-J. van de Beek, S. K. Wilson, and P. O. Börjesson, "Analysis of DFT-based channel estimators for OFDM," Research Report, 1996.`,
  },
  li2020: {
    order: 8,
    title: "Residual Neural Network",
    short: "Li 2020",
    reference: `L. Li, H. Chen, H.-H. Chang, and L. Liu, "Deep Residual Learning Meets OFDM Channel Estimation," IEEE Wireless Communications Letters, vol. 9, no. 5, pp. 615-618, 2020.`,
  },
  melgar2022: {
    order: 7,
    title: "Deep Neural Networks",
    short: "Melgar 2022",
    reference: `A. Melgar, A. de la Fuente, L. Carro-Calvo, O. Barquero-Perez, and E. Morgado, "Deep Neural Network: An Alternative to Traditional Channel Estimators in Massive MIMO Systems," IEEE Transactions on Cognitive Communications and Networking, vol. 8, no. 2, pp. 657-671, 2022.`,
  },
  wei2017: {
    order: 6,
    title: "Sparse Bayesian Learning",
    short: "Wei 2017",
    reference: `C. Wei, Z. Zhang, and H. Liu, "Expectation-propagation based low-complexity channel estimation for massive MIMO systems," IEEE International Conference on Communications (ICC), 2017.`,
  },
  koller2022: {
    order: 3,
    title: "Gaussian Mixture Models",
    short: "Koller 2022",
    reference: `M. Koller, B. Fesl, N. Turan, and W. Utschick, "An Asymptotically MSE-Optimal Estimator Based on Gaussian Mixture Models," IEEE Transactions on Signal Processing, vol. 70, pp. 4109-4123, 2022.`,
  },
  lyu2026: {
    order: 5,
    title: "Sparse/Diffuse SDP",
    short: "Lyu 2026",
    reference: `L. Lyu, M. Ferreira Da Costa, and U. Mitra, "Atomic Hybrid Sparse/Diffuse Channel Estimation and Cramer-Rao Bounds Analysis," arXiv:2605.01825v1, 2026`,
  },
};

const SELECTION_PRESETS = [
  { id: "all", label: "Select all" },
  { id: "interpolators", label: "Select all interpolators" },
  { id: "linear", label: "Select all linear estimators" },
  { id: "sparsity", label: "Select all sparsity-based estimators" },
  { id: "latent", label: "Select all latent variable-based estimators" },
  { id: "neural", label: "Select all neural network-based estimators" },
];

const FALLBACK_COLOR = "#4B4B4B";

const state = {
  data: null,
  scenarioId: null,
  metric: "nmse",
  rankSnr: null,
  selectedOrder: [],
  chartInstance: null,
};

let ChartJs = null;
let chartLoadError = null;

const mount = document.querySelector("#leaderboardApp");

function renderLeaderboardShell() {
  if (mount === null) return;

  const standaloneNote = mount.dataset.context === "standalone"
    ? `
      <aside class="leaderboard-note" aria-label="Note">
        <div class="leaderboard-note-title">Disclaimer</div>
        <p>
        Methods from prior literature included on this leaderboard were not originally developed for this benchmark and,
        in some cases, required adaptation to align with its design. While we have made every effort to implement and
        evaluate these methods faithfully, implementation errors or unintended deviations cannot be entirely ruled out.
        Accordingly, a method's relative performance on this scoreboard should not be interpreted as a definitive
        assessment of its underlying quality, general applicability, or the expertise of its authors.
        </p>
      </aside>
    `
    : "";

  mount.innerHTML = `
    <div class="leaderboard">
      <div class="page-layout">
        <aside class="available-panel" aria-label="Available estimators">
          <div class="panel-heading">
            <h1>Available Estimators</h1>
          </div>
          <div id="availablePresetList" class="selection-preset-list"></div>
          <div id="availableEstimatorList" class="estimator-list available-list"></div>
        </aside>

        <main class="app-shell">
          <div id="metaLine" class="meta-line">Loading results...</div>

          <section class="chart-panel">
            <div class="chart-header">
              <div class="chart-controls">
                <fieldset class="field metric-field">
                  <span class="metric-title">Metric</span>
                  <div class="segmented" role="radiogroup" aria-label="Metric">
                    <label><input type="radio" name="metric" value="nmse" checked><span>NMSE</span></label>
                    <label><input type="radio" name="metric" value="ber"><span>BER</span></label>
                    <label><input type="radio" name="metric" value="bler"><span>BLER</span></label>
                  </div>
                </fieldset>
              </div>
            </div>
            <div class="chart-frame">
              <canvas id="metricChart" role="img" aria-label="Selected estimator metric curves"></canvas>
              <div id="chartMessage" class="chart-message" hidden></div>
            </div>
          </section>

          <section class="ranking-panel" aria-label="Ranking SNR">
            <label class="field rank-field">
              <span>Eb/N0</span>
              <div class="rank-slider-control">
                <input id="rankSnrSlider" type="range" min="0" max="0" step="1" value="0">
                <output id="rankSnrValue" for="rankSnrSlider"></output>
              </div>
            </label>
          </section>

          ${standaloneNote}
        </main>

        <aside class="selector-panel" aria-label="Estimator ranking">
          <div class="panel-heading">
            <h1>Ranking</h1>
          </div>
          <div class="table-frame ranking-table-frame">
            <table class="ranking-table">
              <thead>
                <tr>
                  <th>Rank</th>
                  <th>Estimator</th>
                  <th id="rankingMetricHeader">NMSE</th>
                </tr>
              </thead>
              <tbody id="leaderboardRows"></tbody>
            </table>
          </div>
        </aside>
      </div>
    </div>
  `;
}

renderLeaderboardShell();


function placeLeaderboardPanels() {
  const pageLayout = mount?.querySelector(".page-layout");

  // app.js is loaded on every documentation page. On other pages,
  // there is no leaderboard and nothing should happen.
  if (pageLayout === null || pageLayout === undefined) {
    return;
  }

  const selectedPanel = pageLayout.querySelector(".selector-panel");
  const appShell = pageLayout.querySelector(".app-shell");
  const rightSidebar = document.querySelector(
    "#pst-secondary-sidebar .sidebar-secondary-items"
  );

  // The standalone page keeps all three panels in its own grid.
  if (rightSidebar === null) {
    return;
  }

  document.body.classList.add("leaderboard-page");

  if (selectedPanel === null || appShell === null) return;

  const note = mount.nextElementSibling;
  if (note?.matches(".admonition.note")) {
    appShell.querySelector(".ranking-panel")?.after(note);
  }

  const wideLayout = window.matchMedia("(min-width: 1550px)");
  const updateRankingPlacement = () => {
    if (wideLayout.matches) {
      rightSidebar.prepend(selectedPanel);
    } else {
      appShell.append(selectedPanel);
    }
  };
  updateRankingPlacement();
  wideLayout.addEventListener("change", updateRankingPlacement);
}

placeLeaderboardPanels();


const els = {
  metaLine: document.querySelector("#metaLine"),
  rankSnrSlider: document.querySelector("#rankSnrSlider"),
  rankSnrValue: document.querySelector("#rankSnrValue"),
  availablePresetList: document.querySelector("#availablePresetList"),
  availableEstimatorList: document.querySelector("#availableEstimatorList"),
  rankingMetricHeader: document.querySelector("#rankingMetricHeader"),
  chart: document.querySelector("#metricChart"),
  chartMessage: document.querySelector("#chartMessage"),
  rows: document.querySelector("#leaderboardRows"),
};

function currentScenario() {
  return state.data.scenarios.find((scenario) => scenario.id === state.scenarioId);
}

function runById(scenario, runId) {
  return scenario.runs.find((run) => run.id === runId);
}

function sourceLabel(run) {
  return run.source === "builtin" ? "builtin" : (run.paper || run.source || "unknown");
}

function sourceInfo(source) {
  if (SOURCE_INFO[source]) return SOURCE_INFO[source];
  const spaced = source.replace(/([a-zA-Z]+)(\d{4})$/, "$1 $2");
  return {
    order: 1000,
    title: spaced,
    short: spaced,
    reference: spaced,
  };
}

function compareSourceKeys(a, b) {
  const infoA = sourceInfo(a);
  const infoB = sourceInfo(b);
  if (infoA.order !== infoB.order) return infoA.order - infoB.order;
  return infoA.title.localeCompare(infoB.title);
}

function orderedScenarioRuns(scenario) {
  return scenario.runs
    .map((run, index) => ({ run, index }))
    .sort((a, b) => {
      const sourceCompare = compareSourceKeys(sourceLabel(a.run), sourceLabel(b.run));
      if (sourceCompare !== 0) return sourceCompare;
      return a.index - b.index;
    })
    .map((item) => item.run);
}

function selectionPresetRunIds(presetId, scenario) {
  const runs = orderedScenarioRuns(scenario);
  if (presetId === "all") return runs.map((run) => run.id);

  return runs
    .filter((run) => {
      const source = sourceLabel(run);
      if (presetId === "interpolators") {
        return source === "coleri2002";
      }
      if (presetId === "linear") {
        return ["builtin", "edfors1996"].includes(source);
      }
      if (presetId === "sparsity") {
        return [
          "edfors1996",
          "delfeld2025",
          "wei2017",
          "lyu2026",
        ].includes(source);
      }
      if (presetId === "latent") {
        return ["koller2022", "baur2022"].includes(source);
      }
      if (presetId === "neural") {
        return [
          "melgar2022",
          "li2020",
          "baur2022",
          "balevi2020",
        ].includes(source);
      }
      return false;
    })
    .map((run) => run.id);
}

function sourceRunIds(source, scenario) {
  return orderedScenarioRuns(scenario)
    .filter((run) => sourceLabel(run) === source)
    .map((run) => run.id);
}

function sortRunIdsForScenario(runIds, scenario) {
  const selectedIds = new Set(runIds);
  return orderedScenarioRuns(scenario)
    .filter((run) => selectedIds.has(run.id))
    .map((run) => run.id);
}

function selectedSourceKeys(scenario) {
  const keys = [];
  state.selectedOrder.forEach((runId) => {
    const run = runById(scenario, runId);
    if (!run) return;
    const key = sourceLabel(run);
    if (!keys.includes(key)) keys.push(key);
  });
  return keys;
}

function paletteGroupForSelectedSource(source, scenario) {
  const sourceIndex = selectedSourceKeys(scenario).indexOf(source);
  const index = sourceIndex >= 0 ? sourceIndex : 0;
  const groupName = PALETTE_GROUPS[index % PALETTE_GROUPS.length];
  return PALETTE_GROUP_BY_NAME[groupName];
}

function relativeLuminance(hex) {
  const channels = hex.match(/[0-9a-f]{2}/gi).map((value) => {
    const channel = Number.parseInt(value, 16) / 255;
    return channel <= 0.04045
      ? channel / 12.92
      : ((channel + 0.055) / 1.055) ** 2.4;
  });
  return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2];
}

function orderedShades(group) {
  if (!group || !Array.isArray(group.colors)) return [];
  const darkToLight = [...group.colors]
    .sort((a, b) => relativeLuminance(a.hex) - relativeLuminance(b.hex));
  if (darkToLight.length !== 5) return darkToLight;
  return [
    darkToLight[0],
    darkToLight[2],
    darkToLight[1],
    darkToLight[4],
    darkToLight[3],
  ];
}

function runColor(run, scenario) {
  const source = sourceLabel(run);
  const sourceRuns = state.selectedOrder
    .map((runId) => runById(scenario, runId))
    .filter((candidate) => candidate && sourceLabel(candidate) === source);
  const index = Math.max(0, sourceRuns.findIndex((candidate) => candidate.id === run.id));
  const shades = orderedShades(paletteGroupForSelectedSource(source, scenario));
  if (!shades.length) return FALLBACK_COLOR;
  return shades[index % shades.length].hex || FALLBACK_COLOR;
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (char) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    "\"": "&quot;",
    "'": "&#39;",
  }[char]));
}

function softWrapEstimatorName(name) {
  return escapeHtml(name)
    .replace(/([a-z0-9])([A-Z])/g, "$1<wbr>$2")
    .replace(/([A-Z])([A-Z][a-z])/g, "$1<wbr>$2");
}

function formatDate(value) {
  if (!value) return "";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString();
}

function formatSnr(value) {
  return Number(value).toFixed(Number.isInteger(value) ? 0 : 1);
}

function formatMetric(value) {
  if (!Number.isFinite(value)) return "n/a";
  if (value === 0) return "0";
  return value.toExponential(2);
}

function formatYTick(value) {
  const exponent =  Math.round(log10(value));
  const SUPERSCRIPT = ["⁰", "¹", "²", "³", "⁴", "⁵", "⁶", "⁷", "⁸", "⁹"];
  if (exponent < 0 && exponent > -10) {
    return "10\u200A⁻\u200A".concat(SUPERSCRIPT[-exponent]);
  }
  else if (exponent >= 0 && exponent <= 10) {
    return "10\u200A".concat(SUPERSCRIPT[exponent]);
  } else {
    return value.toExponential(2);
  }
}

function formatLogTick(value) {
  if (!Number.isFinite(value) || value <= 0) return "";
  const exponent = log10(value);
  if (Math.abs(exponent - Math.round(exponent)) > 1e-6) return "";
  return formatYTick(value);
}

function logTickValues(minValue, maxValue) {
  const minPower = Math.floor(log10(minValue));
  const maxPower = Math.ceil(log10(maxValue));
  const ticks = [];

  for (let power = minPower; power <= maxPower; power += 1) {
    const decade = 10 ** power;
    for (let multiplier = 1; multiplier <= 9; multiplier += 1) {
      const value = multiplier * decade;
      if (value >= minValue && value <= maxValue) {
        ticks.push(value);
      }
    }
  }

  return ticks;
}

function sourceBadge(source, className = "paper") {
  const info = sourceInfo(source);
  return `
    <span
      class="source-pill ${escapeHtml(className)}"
      tabindex="0"
      data-reference="${escapeHtml(info.reference)}"
      title="${escapeHtml(info.reference)}"
    >${escapeHtml(info.short)}</span>
  `;
}

function metricAtSnr(run, metric, snr) {
  let index = 0;
  let bestDistance = Infinity;
  run.ebnoDbs.forEach((value, candidateIndex) => {
    const distance = Math.abs(value - snr);
    if (distance < bestDistance) {
      index = candidateIndex;
      bestDistance = distance;
    }
  });
  const value = run[metric][index];
  return Number.isFinite(value) ? value : null;
}

function rankSnrIndex(scenario) {
  const index = scenario.snrDb.findIndex((snr) => snr === state.rankSnr);
  return index >= 0 ? index : Math.max(0, scenario.snrDb.length - 1);
}

function updateRankSnrControl() {
  const scenario = currentScenario();
  const index = rankSnrIndex(scenario);
  const snr = scenario.snrDb[index] ?? null;
  const label = snr === null ? "" : `${formatSnr(snr)} dB`;
  const snrMin = Math.min(...scenario.snrDb);
  const snrMax = Math.max(...scenario.snrDb);
  const position = snrMax > snrMin && snr !== null
    ? (snr - snrMin) / (snrMax - snrMin)
    : 0.5;

  els.rankSnrSlider.min = String(snrMin);
  els.rankSnrSlider.max = String(snrMax);
  els.rankSnrSlider.step = "any";
  els.rankSnrSlider.value = String(snr ?? snrMin);
  els.rankSnrSlider.disabled = scenario.snrDb.length <= 1;
  els.rankSnrSlider.setAttribute("aria-valuetext", label);
  els.rankSnrValue.textContent = label;
  els.rankSnrValue.parentElement.style.setProperty(
    "--rank-position",
    `${position * 100}%`
  );
}

function defaultSelectedRunIds(scenario) {
  return orderedScenarioRuns(scenario).map((run) => run.id);
}

function initScenario(scenarioId) {
  const scenario = state.data.scenarios.find((item) => item.id === scenarioId);
  state.scenarioId = scenario.id;
  state.rankSnr = scenario.snrDb[scenario.snrDb.length - 1] ?? null;
  state.selectedOrder = defaultSelectedRunIds(scenario);
  updateRankSnrControl();
  render();
}


function renderSelectionPresets(scenario, selectedIds) {
  els.availablePresetList.innerHTML = SELECTION_PRESETS
    .map((preset) => {
      const targetIds = selectionPresetRunIds(preset.id, scenario);
      const selectedCount = targetIds.filter((runId) => selectedIds.has(runId)).length;
      const checked = targetIds.length > 0 && selectedCount === targetIds.length
        ? " checked"
        : "";
      const disabled = targetIds.length === 0 ? " disabled" : "";
      return `
        <label class="selection-preset-option">
          <input type="checkbox" data-selection-preset="${preset.id}"${checked}${disabled}>
          <span class="selection-preset-label">${escapeHtml(preset.label)}</span>
        </label>
      `;
    })
    .join("");

  els.availablePresetList
    .querySelectorAll("input[data-selection-preset]")
    .forEach((input) => {
      const targetIds = selectionPresetRunIds(input.dataset.selectionPreset, scenario);
      const selectedCount = targetIds.filter((runId) => selectedIds.has(runId)).length;
      input.indeterminate = selectedCount > 0 && selectedCount < targetIds.length;
    });
}

function renderEstimatorList() {
  const scenario = currentScenario();
  const selectedIds = new Set(state.selectedOrder);
  const groups = [];

  renderSelectionPresets(scenario, selectedIds);

  orderedScenarioRuns(scenario).forEach((run) => {
    const key = sourceLabel(run);
    let group = groups.find((candidate) => candidate.key === key);
    if (!group) {
      group = { key, runs: [] };
      groups.push(group);
    }
    group.runs.push(run);
  });

  groups.sort((a, b) => compareSourceKeys(a.key, b.key));

  els.availableEstimatorList.innerHTML = groups
    .map((group) => {
      const info = sourceInfo(group.key);
      const groupIds = group.runs.map((run) => run.id);
      const groupSelectedCount = groupIds.filter((runId) => selectedIds.has(runId)).length;
      const groupChecked = groupIds.length > 0 && groupSelectedCount === groupIds.length
        ? " checked"
        : "";
      const methodRows = group.runs.map((run) => {
        const checked = selectedIds.has(run.id) ? " checked" : "";
        return `
          <label class="estimator-option available-option">
            <input type="checkbox" data-run-id="${escapeHtml(run.id)}"${checked}>
            <span class="estimator-name">${escapeHtml(run.estimator)}</span>
          </label>
        `;
      }).join("");

      return `
        <section class="estimator-group">
          <div class="estimator-group-header">
            <label class="estimator-group-select">
              <input
                type="checkbox"
                data-source-key="${escapeHtml(group.key)}"
                ${groupChecked}
              >
              <span class="estimator-group-title">${escapeHtml(info.title)}</span>
            </label>
            <div class="estimator-group-source">
              ${sourceBadge(group.key, group.key === "builtin" ? "builtin" : "paper")}
            </div>
          </div>
          <div class="estimator-methods">
            ${methodRows}
          </div>
        </section>
      `;
    })
    .join("");

  els.availableEstimatorList
    .querySelectorAll("input[data-source-key]")
    .forEach((input) => {
      const targetIds = sourceRunIds(input.dataset.sourceKey, scenario);
      const selectedCount = targetIds.filter((runId) => selectedIds.has(runId)).length;
      input.indeterminate = selectedCount > 0 && selectedCount < targetIds.length;
    });
}

function selectedRuns() {
  const scenario = currentScenario();
  return state.selectedOrder
    .map((runId) => runById(scenario, runId))
    .filter(Boolean);
}

function log10(value) {
  return Math.log(value) / Math.LN10;
}

function alignChartMessageToPlot(chart) {
  if (!els.chartMessage || !chart?.chartArea || chart.width <= 0 || chart.height <= 0) {
    return;
  }

  const frameRect = els.chartMessage.parentElement.getBoundingClientRect();
  const canvasRect = chart.canvas.getBoundingClientRect();
  const scaleX = canvasRect.width / chart.width;
  const scaleY = canvasRect.height / chart.height;
  const left = canvasRect.left - frameRect.left + chart.chartArea.left * scaleX;
  const top = canvasRect.top - frameRect.top + chart.chartArea.top * scaleY;
  const right = frameRect.right - canvasRect.left - chart.chartArea.right * scaleX;
  const bottom = frameRect.bottom - canvasRect.top - chart.chartArea.bottom * scaleY;
  els.chartMessage.style.inset = `${top}px ${right}px ${bottom}px ${left}px`;
}

function showChartMessage(message) {
  if (!els.chartMessage) return;
  els.chartMessage.textContent = message;
  els.chartMessage.hidden = false;
  if (state.chartInstance) alignChartMessageToPlot(state.chartInstance);
}

function hideChartMessage() {
  if (!els.chartMessage) return;
  els.chartMessage.hidden = true;
  els.chartMessage.textContent = "";
}

function chartFont(size, weight = "700") {
  return {
    family: "Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, \"Segoe UI\", sans-serif",
    size,
    weight,
  };
}

function cssVar(name, fallback) {
  const value = getComputedStyle(mount ?? document.documentElement).getPropertyValue(name).trim();
  return value || fallback;
}

const horizontalYAxisTitlePlugin = {
  id: "horizontalYAxisTitle",
  afterDraw(chart, _args, options) {
    if (!options?.text) return;

    const { ctx, chartArea } = chart;
    const font = options.font || chartFont(15, "700");
    ctx.save();
    ctx.fillStyle = options.color;
    ctx.font = `${font.weight} ${font.size}px ${font.family}`;
    ctx.textAlign = "center";
    ctx.textBaseline = "bottom";
    const textWidth = ctx.measureText(options.text).width;
    const textHeight = font.size * 1.2;
    ctx.fillStyle = options.backgroundColor;
    ctx.fillRect(
      chartArea.left - textWidth / 2 - 4,
      chartArea.top - 16 - textHeight - 2,
      textWidth + 8,
      textHeight + 4
    );
    ctx.fillStyle = options.color;
    ctx.fillText(options.text, chartArea.left, chartArea.top - 16);
    ctx.restore();
  },
};

const axisTitleBackdropPlugin = {
  id: "axisTitleBackdrop",
  afterDraw(chart, _args, options) {
    const scale = chart.scales.x;
    const title = scale?.options?.title;
    if (!scale || !title?.display || !title.text) return;

    const font = title.font || chartFont(15, "700");
    const fontSize = Number(font.size) || 15;
    const lineHeight = Number(font.lineHeight) || fontSize * 1.2;
    const titlePadding = title.padding;
    const paddingBottom = typeof titlePadding === "number"
      ? titlePadding
      : Number(titlePadding?.bottom) || 4;
    const text = String(title.text);
    const titleX = (scale.left + scale.right) / 2;
    const titleY = scale.bottom - paddingBottom - lineHeight / 2;
    const { ctx } = chart;

    ctx.save();
    ctx.font = `${font.weight} ${fontSize}px ${font.family}`;
    ctx.textAlign = "center";
    ctx.textBaseline = "alphabetic";
    const textMetrics = ctx.measureText(text);
    const textAscent = textMetrics.actualBoundingBoxAscent || fontSize * 0.8;
    const textDescent = textMetrics.actualBoundingBoxDescent || fontSize * 0.2;
    const textBaseline = titleY + (textAscent - textDescent) / 2;
    ctx.fillStyle = options.backgroundColor;
    ctx.fillRect(
      titleX - textMetrics.width / 2 - 4,
      textBaseline - textAscent - 5,
      textMetrics.width + 8,
      textAscent + textDescent + 10
    );
    ctx.fillStyle = options.color;
    ctx.fillText(text, titleX, textBaseline);
    ctx.restore();
  },
};

const rankNeedlePlugin = {
  id: "rankNeedle",
  beforeDraw(chart, _args, options) {
    const snr = Number(state.rankSnr);
    const scale = chart.scales.x;
    if (!Number.isFinite(snr) || !scale || !chart.chartArea) return;

    const x = scale.getPixelForValue(snr);
    const { ctx, chartArea } = chart;
    ctx.save();
    ctx.beginPath();
    ctx.moveTo(x, chartArea.bottom - 10);
    ctx.lineTo(x, chart.height);
    ctx.strokeStyle = options.color;
    ctx.lineWidth = 2;
    ctx.globalAlpha = 0.5;
    ctx.stroke();
    ctx.restore();
  },
};

function alignRankSliderToChart(chart) {
  const field = els.rankSnrSlider.closest(".rank-field");
  if (!field || !chart.chartArea || chart.width <= 0) return;

  const fieldRect = field.getBoundingClientRect();
  const canvasRect = chart.canvas.getBoundingClientRect();
  if (fieldRect.width <= 0 || canvasRect.width <= 0) return;

  const canvasScaleX = canvasRect.width / chart.width;
  const axisLeft = canvasRect.left + chart.chartArea.left * canvasScaleX;
  const axisRight = canvasRect.left + chart.chartArea.right * canvasScaleX;
  field.style.setProperty(
    "--rank-track-left",
    `${Math.max(0, axisLeft - fieldRect.left)}px`
  );
  field.style.setProperty(
    "--rank-track-right",
    `${Math.max(0, fieldRect.right - axisRight)}px`
  );
}

const rankSliderAlignmentPlugin = {
  id: "rankSliderAlignment",
  afterLayout(chart) {
    alignRankSliderToChart(chart);
  },
};

const chartMessageAlignmentPlugin = {
  id: "chartMessageAlignment",
  afterLayout(chart) {
    alignChartMessageToPlot(chart);
  },
};

function chartPointData(run, metric) {
  const count = Math.min(run.ebnoDbs.length, run[metric].length);
  return Array.from({ length: count }, (_, index) => ({
    x: run.ebnoDbs[index],
    y: run[metric][index],
  })).filter((point) =>
    Number.isFinite(point.x) && Number.isFinite(point.y) && point.y > 0
  );
}

function renderChart() {
  if (!ChartJs) {
    showChartMessage(chartLoadError ?? "Loading Chart.js...");
    return;
  }

  const scenario = currentScenario();
  const runs = selectedRuns();
  const metric = state.metric;

  if (!runs.length) {
    showChartMessage("Select at least one estimator");
    if (state.chartInstance) {
      state.chartInstance.data.datasets = [];
      state.chartInstance.update();
    }
    return;
  }

  const allX = runs.flatMap((run) => run.ebnoDbs.filter((value) => Number.isFinite(value)));
  const allY = runs.flatMap((run) =>
    run[metric].filter((value) => Number.isFinite(value) && value > 0)
  );
  if (!allX.length || !allY.length) {
    showChartMessage("No positive values to plot on a log scale");
    if (state.chartInstance) {
      state.chartInstance.data.datasets = [];
      state.chartInstance.update();
    }
    return;
  }
  hideChartMessage();

  const xMin = Math.min(...allX);
  const xMax = Math.max(...allX);
  const yMinRaw = Math.min(...allY);
  const yMaxRaw = Math.max(...allY);
  let yMin = 10 ** Math.floor(log10(yMinRaw));
  let yMax = 10 ** Math.ceil(log10(yMaxRaw));
  if (yMin === yMax) {
    yMin /= 10;
    yMax *= 10;
  }

  const datasets = runs.map((run) => {
    const color = runColor(run, scenario);
    return {
      label: run.estimator,
      data: chartPointData(run, metric),
      parsing: false,
      borderColor: color,
      backgroundColor: color,
      borderWidth: 2.5,
      pointRadius: 3,
      pointHoverRadius: 5,
      tension: 0,
    };
  }).filter((dataset) => dataset.data.length);

  const gridColor = cssVar("--bench-chart-grid-color", "#c5c0c0");
  const mutedColor = cssVar("--pst-color-text-muted", "#444444");
  const backgroundColor = cssVar("--pst-color-background", "#ffffff");
  const primaryColor = cssVar("--pst-color-primary", "#AA3333");
  const options = {
    responsive: true,
    maintainAspectRatio: false,
    animation: false,
    layout: {
      padding: { top: 36, right: 12, bottom: 18, left: 8 },
    },
    plugins: {
      legend: { display: false },
      title: { display: false },
      horizontalYAxisTitle: {
        text: METRIC_LABELS[metric],
        color: mutedColor,
        backgroundColor,
        font: chartFont(15, "700"),
      },
      axisTitleBackdrop: {
        backgroundColor,
        color: mutedColor,
      },
      rankNeedle: { color: primaryColor },
      tooltip: {
        callbacks: {
          title: (items) => `${formatSnr(items[0].parsed.x)} dB`,
          label: (item) => `${item.dataset.label}: ${formatMetric(item.parsed.y)}`,
        },
      },
    },
    scales: {
      x: {
        type: "linear",
        min: xMin === xMax ? xMin - 1 : xMin,
        max: xMin === xMax ? xMax + 1 : xMax,
        border: {
          color: mutedColor,
          width: 2,
        },
        title: {
          display: true,
          text: "Eb/N0 [dB]",
          color: "rgba(0, 0, 0, 0)",
          font: chartFont(15, "700"),
        },
        grid: { color: gridColor },
        ticks: {
          showLabelBackdrop: true,
          backdropColor: backgroundColor,
          backdropPadding: 2,
          color: mutedColor,
          font: chartFont(15, "700"),
          callback: (value) => formatSnr(Number(value)),
        },
      },
      y: {
        type: "logarithmic",
        min: yMin,
        max: yMax,
        border: {
          color: mutedColor,
          width: 2,
        },
        afterBuildTicks: (axis) => {
          axis.ticks = logTickValues(yMin, yMax).map((value) => ({
            value,
            major: Math.abs(log10(value) - Math.round(log10(value))) < 1e-6,
          }));
        },
        grid: {
          color: gridColor,
          tickColor: (context) =>
            context.tick?.major ? gridColor : "rgba(0, 0, 0, 0)",
        },
        ticks: {
          autoSkip: false,
          showLabelBackdrop: true,
          backdropColor: backgroundColor,
          backdropPadding: 2,
          color: mutedColor,
          font: chartFont(15, "700"),
          callback: (value) => formatLogTick(Number(value)),
        },
      },
    },
  };

  if (!state.chartInstance) {
    state.chartInstance = new ChartJs(els.chart, {
      type: "line",
      data: { datasets },
      options,
      plugins: [
        horizontalYAxisTitlePlugin,
        axisTitleBackdropPlugin,
        rankNeedlePlugin,
        rankSliderAlignmentPlugin,
        chartMessageAlignmentPlugin,
      ],
    });
  } else {
    state.chartInstance.data.datasets = datasets;
    state.chartInstance.options = options;
    state.chartInstance.update();
  }
}

function renderRows() {
  const scenario = currentScenario();
  const runs = selectedRuns();
  const selectedIds = new Set(state.selectedOrder);
  const snr = Number(state.rankSnr);
  els.rankingMetricHeader.textContent = METRIC_LABELS[state.metric];
  const compareByMetric = (a, b) => {
    const aValid = Number.isFinite(a.rankValue);
    const bValid = Number.isFinite(b.rankValue);
    if (aValid && bValid) return a.rankValue - b.rankValue;
    if (aValid) return -1;
    if (bValid) return 1;
    return a.run.estimator.localeCompare(b.run.estimator);
  };
  const selectedRanked = runs
    .map((run) => ({
      run,
      rankValue: metricAtSnr(run, state.metric, snr),
    }))
    .sort(compareByMetric);
  const unselectedRanked = orderedScenarioRuns(scenario)
    .filter((run) => !selectedIds.has(run.id))
    .map((run) => ({
      run,
      rankValue: metricAtSnr(run, state.metric, snr),
    }))
    .sort(compareByMetric);

  const selectedRows = selectedRanked
    .map((item, index) => {
      const rowClasses = index === 0
        ? "ranking-selected best-row"
        : "ranking-selected";
      const color = runColor(item.run, scenario);
      return `
        <tr class="${rowClasses}" style="--ranking-color:${color}">
          <td>${index + 1}</td>
          <td class="ranking-estimator">${softWrapEstimatorName(item.run.estimator)}</td>
          <td>${formatMetric(item.rankValue)}</td>
        </tr>
      `;
    })
    .join("");
  const unselectedRows = unselectedRanked
    .map((item) => `
      <tr class="ranking-unselected">
        <td aria-label="Unselected"></td>
        <td>${escapeHtml(item.run.estimator)}</td>
        <td>${formatMetric(item.rankValue)}</td>
      </tr>
    `)
    .join("");
  const separator = selectedRanked.length && unselectedRanked.length
    ? `<tr class="ranking-separator" aria-hidden="true"><td colspan="3"></td></tr>`
    : "";

  els.rows.innerHTML = `${selectedRows}${separator}${unselectedRows}`;
}

function render() {
  renderEstimatorList();
  renderChart();
  renderRows();
}

function setRunSelected(runId, selected) {
  const selectedIds = new Set(state.selectedOrder);
  if (selected) {
    selectedIds.add(runId);
  } else {
    selectedIds.delete(runId);
  }
  state.selectedOrder = sortRunIdsForScenario(selectedIds, currentScenario());
}

function setSelectionPreset(presetId, selected) {
  const scenario = currentScenario();
  const selectedIds = new Set(state.selectedOrder);
  const targetIds = selectionPresetRunIds(presetId, scenario);
  if (presetId === "all" && !selected) {
    selectedIds.clear();
  } else {
    targetIds.forEach((runId) => {
      if (selected) {
        selectedIds.add(runId);
      } else {
        selectedIds.delete(runId);
      }
    });
  }
  state.selectedOrder = sortRunIdsForScenario(selectedIds, scenario);
}

function setSourceSelected(source, selected) {
  const scenario = currentScenario();
  const selectedIds = new Set(state.selectedOrder);
  sourceRunIds(source, scenario).forEach((runId) => {
    if (selected) {
      selectedIds.add(runId);
    } else {
      selectedIds.delete(runId);
    }
  });
  state.selectedOrder = sortRunIdsForScenario(selectedIds, scenario);
}

function bindEvents() {

  els.rankSnrSlider.addEventListener("input", (event) => {
    const scenario = currentScenario();
    const requestedSnr = Number(event.target.value);
    state.rankSnr = scenario.snrDb.reduce((closest, candidate) =>
      Math.abs(candidate - requestedSnr) < Math.abs(closest - requestedSnr)
        ? candidate
        : closest
    );
    updateRankSnrControl();
    state.chartInstance?.draw();
    renderRows();
  });

  mount.querySelectorAll("input[name='metric']").forEach((input) => {
    input.addEventListener("change", (event) => {
      state.metric = event.target.value;
      renderChart();
      renderRows();
    });
  });

  els.availableEstimatorList.addEventListener("change", (event) => {
    const source = event.target.dataset.sourceKey;
    if (source) {
      setSourceSelected(source, event.target.checked);
      render();
      return;
    }

    const runId = event.target.dataset.runId;
    if (!runId) return;
    setRunSelected(runId, event.target.checked);
    render();
  });

  els.availablePresetList.addEventListener("change", (event) => {
    const presetId = event.target.dataset.selectionPreset;
    if (!presetId) return;
    setSelectionPreset(presetId, event.target.checked);
    render();
  });
}

async function init() {
  bindEvents();
  try {
    const response = await fetch(new URL("./data.json", import.meta.url), { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    state.data = await response.json();
    if (!state.data.scenarios.length) {
      throw new Error("No benchmark result files found");
    }
    els.metaLine.textContent = `Generated ${formatDate(state.data.generatedAt)}`;
    const scenario = state.data.scenarios.find(
      (candidate) => candidate.id === "3gpp-umi"
    );

    if (scenario === undefined) {
      throw new Error("Required scenario '3gpp-umi' is unavailable");
    }

    initScenario(scenario.id);
  } catch (error) {
    els.metaLine.innerHTML = `<span class="status-error">Could not load data.json: ${error.message}</span>`;
    showChartMessage("Run leaderboard/mimo_ofdm_chest/generate_data.py first");
  }
}

async function loadChartJs() {
  try {
    const chartModule = await import(
      "https://cdn.jsdelivr.net/npm/chart.js@4.5.1/auto/auto.js/+esm"
    );
    ChartJs = chartModule.default;
  } catch (error) {
    chartLoadError = `Could not load Chart.js: ${error.message}`;
  }
  if (state.scenarioId !== null) renderChart();
}

if (els.chart) {
  init();
  loadChartJs();
}
