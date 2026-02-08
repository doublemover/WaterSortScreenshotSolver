// main.js
import { buildColorLabels } from './colors.js';
import { contentsToStacksTopBottom, stacksToContentsTopBottom, solveBfs, applyMove } from './solver.js';

// --- Defaults (UI is intentionally minimal) ---
const DEFAULT_CV_VER = '4.9.0';
const DEFAULT_DBSCAN_EPS = 16.0;
const DEFAULT_MAX_STATES = 200000;
const DEFAULT_USE_TESSERACT = true;

const elFile = document.getElementById('file');
const btnAnalyze = document.getElementById('btnAnalyze');
const btnSolve = document.getElementById('btnSolve');
const statusLine = document.getElementById('statusLine');
const legend = document.getElementById('legend');
const legendTitle = document.getElementById('legendTitle');
const boardDetectedEl = document.getElementById('boardDetected');
const boardFinalEl = document.getElementById('boardFinal');
const solverTitleEl = document.getElementById('solverTitle');
const solverMetaEl = document.getElementById('solverMeta');
const movesEl = document.getElementById('moves');

const canvas = document.getElementById('canvas');
const overlay = document.getElementById('overlay');
const ctx = canvas.getContext('2d', { willReadFrequently: true });
const octx = overlay.getContext('2d');

let imageBitmap = null;
let lastImageData = null; // ImageData on base canvas
let parseResult = null; // result from vision worker, later augmented with OCR
let colorLabels = null;

let visionWorker = null;
let visionWorkerCvVer = null;

let isAnalyzing = false;

function resetAnalysisLogs() {
  // UI intentionally hides debug logs; keep this as a no-op hook.
}

function pushAnalysisLog(line) {
  if (isAnalyzing && statusLine) statusLine.textContent = line;
}

function setStatus(msg) {
  if (statusLine) statusLine.textContent = msg;
}

function ensureVisionWorker(cvVer) {
  if (visionWorker && visionWorkerCvVer === cvVer) return visionWorker;

  if (visionWorker) {
    visionWorker.terminate();
    visionWorker = null;
  }

  visionWorkerCvVer = cvVer;

  visionWorker = new Worker('./vision.worker.js');

  // If the worker throws (e.g., importScripts fails, CORS blocked, syntax error),
  // it will NOT send us a postMessage. It will land here.
  visionWorker.onerror = (err) => {
    console.error('[vision worker error event]', err);
    setStatus(`Vision worker crashed: ${err?.message || err}`);
    isAnalyzing = false;
    btnAnalyze.disabled = false;
    btnSolve.disabled = true;
  };

  visionWorker.onmessageerror = (err) => {
    console.error('[vision worker message error]', err);
    setStatus('Vision worker message error (structured clone failed).');
    isAnalyzing = false;
    btnAnalyze.disabled = false;
    btnSolve.disabled = true;
  };

  visionWorker.onmessage = (e) => {
    const { type } = e.data || {};
    if (type === 'log') {
      console.log('[vision]', e.data.message);
      pushAnalysisLog(String(e.data.message || ''));
      return;
    }
    if (type === 'error') {
      console.error(e.data.error);
      setStatus(`Vision worker error: ${e.data.error}`);
      isAnalyzing = false;
      // allow retry without reloading the image
      btnAnalyze.disabled = false;
      btnSolve.disabled = true;
      return;
    }
    if (type === 'result') {
      onVisionResult(e.data.result, e.data.debug);
      return;
    }
  };
  return visionWorker;
}

function clearOverlay() {
  octx.clearRect(0, 0, overlay.width, overlay.height);
}

function drawOverlay(debug) {
  clearOverlay();
  if (!debug) return;

  // Make the overlay visually obvious by default.
  octx.lineWidth = 4;

  // bottle boxes
  for (const b of debug.bottles) {
    const { x, y, w, h, idx, rock } = b;
    octx.strokeStyle = rock ? 'rgba(255, 200, 0, 0.95)' : 'rgba(0, 255, 0, 0.9)';
    octx.strokeRect(x, y, w, h);

    octx.fillStyle = 'rgba(0,0,0,0.55)';
    octx.fillRect(x, Math.max(0, y - 20), 78, 20);

    octx.fillStyle = 'rgba(255,255,255,0.95)';
    octx.font = '14px ui-monospace, monospace';
    octx.fillText(`${idx + 1}${rock ? ' R' : ''}`, x + 6, Math.max(14, y - 6));
  }

  // rock boxes (detected rock piles)
  if (debug.rockBoxes && debug.rockBoxes.length) {
    octx.save();
    octx.strokeStyle = 'rgba(255, 200, 0, 0.95)';
    if (typeof octx.setLineDash === 'function') octx.setLineDash([6, 3]);

    for (const rb of debug.rockBoxes) {
      if (!rb) continue;
      const { x, y, w, h, idx } = rb;
      octx.strokeRect(x, y, w, h);

      // label
      octx.fillStyle = 'rgba(0,0,0,0.55)';
      octx.fillRect(x, Math.min(overlay.height - 20, y + h + 2), 78, 20);

      octx.fillStyle = 'rgba(255,255,255,0.95)';
      octx.font = '14px ui-monospace, monospace';
      octx.fillText(`ROCK${typeof idx === 'number' ? ` b${idx + 1}` : ''}`, x + 6, Math.min(overlay.height - 6, y + h + 16));
    }

    octx.restore();
  }

  // slot sample points
  for (const p of debug.samplePoints) {
    octx.beginPath();
    octx.fillStyle = p.colorHex ? p.colorHex : 'rgba(180,180,180,0.8)';
    octx.arc(p.x, p.y, 7, 0, Math.PI * 2);
    octx.fill();

    // subtle outline so light colors still pop
    octx.strokeStyle = 'rgba(0,0,0,0.6)';
    octx.lineWidth = 2;
    octx.stroke();
  }

  // Restore defaults for subsequent strokes
  octx.lineWidth = 4;

  // badge boxes
  for (let i = 0; i < (debug.badgeBoxes || []).length; i++) {
    const [x, y, w, h] = debug.badgeBoxes[i];
    octx.strokeStyle = 'rgba(0, 255, 255, 0.95)';
    octx.strokeRect(x, y, w, h);

    octx.fillStyle = 'rgba(0,0,0,0.55)';
    octx.fillRect(x, y - 16, 70, 16);

    octx.fillStyle = 'rgba(255,255,255,0.95)';
    octx.font = '13px ui-monospace, monospace';
    octx.fillText(`badge ${i + 1}`, x + 4, y - 4);
  }
}

function setLegend(colorsById, labels) {
  legend.innerHTML = '';
  if (legendTitle) {
    const rockCount = (parseResult?.rock_bottles || []).length;
    legendTitle.textContent = `Colors: ${labels.orderedIds.length} · Rock Bottles: ${rockCount}`;
  }
  const ids = labels.orderedIds;
  for (const id of ids) {
    const hex = colorsById[String(id)];
    const name = labels.namesById[id];
    const abbr = labels.abbrById[id];

    const item = document.createElement('div');
    item.className = 'item';
    item.title = `${abbr} (${hex})`;

    const sw = document.createElement('span');
    sw.className = 'swatch';
    sw.style.background = hex;

    const txt = document.createElement('span');
    txt.textContent = name;

    item.appendChild(sw);
    item.appendChild(txt);
    legend.appendChild(item);
  }
}

function cellText(abbr) {
  // width 5 cell. abbr is 3 chars.
  if (!abbr) return '     ';
  return ` ${abbr} `;
}

function sepText(isRock) {
  return isRock ? '-ROCK' : '-----';
}

function numCell(n) {
  const s = String(n);
  if (s.length >= 5) return s.slice(0, 5);
  const padLeft = Math.floor((5 - s.length) / 2);
  const padRight = 5 - s.length - padLeft;
  return ' '.repeat(padLeft) + s + ' '.repeat(padRight);
}

function formatRow(stacks, rowBottleIdxs, capacity, rockSet, abbrById, bottleNumberOffset) {
  // stacks are bottom->top
  const lines = [];

  // build top->bottom slots per bottle for rendering
  const slotsByBottle = rowBottleIdxs.map((bi) => {
    const st = stacks[bi];
    const slots = new Array(capacity).fill(null);
    for (let i = 0; i < st.length; i++) {
      const slotIdx = capacity - 1 - i;
      slots[slotIdx] = st[i];
    }
    return slots; // top->bottom
  });

  for (let r = 0; r < capacity; r++) {
    let line = '|';
    for (let c = 0; c < rowBottleIdxs.length; c++) {
      const cid = slotsByBottle[c][r];
      const abbr = (cid === null || cid === undefined) ? null : abbrById[cid];
      line += cellText(abbr) + '|';
    }
    lines.push(line);
  }

  // sep line (rock)
  let sep = '|';
  for (let c = 0; c < rowBottleIdxs.length; c++) {
    const bi = rowBottleIdxs[c];
    sep += sepText(rockSet.has(bi)) + '|';
  }
  lines.push(sep);

  // numbering line
  let num = '|';
  for (let c = 0; c < rowBottleIdxs.length; c++) {
    const bi = rowBottleIdxs[c];
    num += numCell(bi + 1 + bottleNumberOffset) + '|';
  }
  lines.push(num);

  // closing sep line
  let end = '|';
  for (let c = 0; c < rowBottleIdxs.length; c++) end += '-----|';
  lines.push(end);

  return lines.join('\n');
}

function formatBoard(stacks, layoutRows, capacity, rockSet, abbrById, bottleNumberOffset = 0) {
  // layoutRows: Array<Array<bottleIndex>>
  return layoutRows.map((row) => formatRow(stacks, row, capacity, rockSet, abbrById, bottleNumberOffset)).join('\n');
}

function buildLayoutForFinal(layoutRows, origBottleCount, extraBottles) {
  // Extra bottles will be appended to the last row, to the right.
  if (extraBottles <= 0) return layoutRows;

  const out = layoutRows.map((r) => r.slice());
  const lastRow = out[out.length - 1];

  for (let k = 0; k < extraBottles; k++) {
    lastRow.push(origBottleCount + k);
  }
  return out;
}

function clearEl(el) {
  if (!el) return;
  el.innerHTML = '';
}

function renderBoardEl(container, contentsTopBottomByBottle, layoutRows, cap, rockSet, colorsById, bottleNumberOffset = 0) {
  if (!container) return;
  clearEl(container);

  if (!contentsTopBottomByBottle || !layoutRows || !layoutRows.length) return;

  for (const row of layoutRows) {
    const rowEl = document.createElement('div');
    rowEl.className = 'boardRow';

    for (const bi of row) {
      const wrap = document.createElement('div');
      wrap.className = 'bottleWrap';

      const num = document.createElement('div');
      num.className = 'bottleNum';
      num.textContent = String(bi + 1 + bottleNumberOffset);
      wrap.appendChild(num);

      const bottle = document.createElement('div');
      bottle.className = 'bottle';
      if (rockSet.has(bi)) {
        bottle.classList.add('rock');
        wrap.title = 'Rock bottle (cannot pour out)';
      }

      const slots = document.createElement('div');
      slots.className = 'bottleSlots';

      const contents = contentsTopBottomByBottle[bi] || new Array(cap).fill(null);

      for (let si = 0; si < cap; si++) {
        const cid = contents[si];
        const slot = document.createElement('div');
        slot.className = 'slot';
        if (cid === null || cid === undefined) {
          slot.classList.add('empty');
        } else {
          const hex = colorsById?.[String(cid)] || '#666';
          slot.style.background = hex;
        }
        slots.appendChild(slot);
      }

      bottle.appendChild(slots);
      wrap.appendChild(bottle);
      rowEl.appendChild(wrap);
    }

    container.appendChild(rowEl);
  }
}

function renderSolverPanel(solveInfo, { addBottlesUsed }) {
  if (!solverTitleEl || !solverMetaEl || !movesEl) return;

  clearEl(movesEl);

  if (!solveInfo) {
    solverTitleEl.textContent = 'Solver';
    solverMetaEl.textContent = '(click Solve)';
    return;
  }

  // Header
  solverTitleEl.textContent = solveInfo.solved ? 'Solved!' : 'No solution';
  solverMetaEl.textContent = `(explored ${solveInfo.explored} states${solveInfo.reason ? `, reason=${solveInfo.reason}` : ''})`;

  const mkMoveItem = ({ step, text, sub, colorHex, amount }) => {
    const item = document.createElement('div');
    item.className = 'moveItem';

    const stepEl = document.createElement('div');
    stepEl.className = 'moveStep';
    stepEl.textContent = `${step}.`;

    const textWrap = document.createElement('div');
    const main = document.createElement('div');
    main.className = 'moveText';
    main.textContent = text;
    textWrap.appendChild(main);
    if (sub) {
      const subEl = document.createElement('div');
      subEl.className = 'moveSub';
      subEl.textContent = sub;
      textWrap.appendChild(subEl);
    }

    const squares = document.createElement('div');
    squares.className = 'moveSquares';
    if (amount && colorHex) {
      for (let k = 0; k < amount; k++) {
        const sq = document.createElement('div');
        sq.className = 'moveSquare';
        sq.style.background = colorHex;
        squares.appendChild(sq);
      }
    }

    item.appendChild(stepEl);
    item.appendChild(textWrap);
    item.appendChild(squares);
    return item;
  };

  let stepNo = 1;

  // Extra bottle steps first.
  if (addBottlesUsed > 0) {
    for (let k = 0; k < addBottlesUsed; k++) {
      const bottleNum = parseResult.num_bottles + k + 1;
      movesEl.appendChild(mkMoveItem({
        step: stepNo,
        text: 'Add Empty Bottle',
        sub: `create bottle ${bottleNum}`,
      }));
      stepNo++;
    }
  }

  if (solveInfo.moves && solveInfo.moves.length) {
    for (const m of solveInfo.moves) {
      const hex = parseResult.colors_by_id?.[String(m.color)] || null;
      const cname = colorLabels?.namesById?.[m.color] || `Color ${m.color}`;
      movesEl.appendChild(mkMoveItem({
        step: stepNo,
        text: `Pour ${m.src + 1} → ${m.dst + 1}`,
        sub: `${cname} × ${m.amt}`,
        colorHex: hex,
        amount: m.amt,
      }));
      stepNo++;
    }
  }
}

function renderOutput(initialStacks, finalStacks, solveInfo) {
  const cap = parseResult.capacity;
  const rockSet = new Set(parseResult.rock_bottles || []);

  const layout = parseResult.layout?.rows || [Array.from({ length: parseResult.num_bottles }, (_, i) => i)];

  // Only show/allocate extra bottles if the solver actually uses them.
  const addBottles = (solveInfo && typeof solveInfo.extraUsed === 'number') ? solveInfo.extraUsed : 0;
  const finalLayout = buildLayoutForFinal(layout, parseResult.num_bottles, addBottles);

  // --- Visual UI ---
  renderBoardEl(boardDetectedEl, parseResult.bottle_contents_ids, layout, cap, rockSet, parseResult.colors_by_id, 0);

  if (finalStacks && solveInfo && solveInfo.solved) {
    const finalContents = stacksToContentsTopBottom(finalStacks, cap);
    renderBoardEl(boardFinalEl, finalContents, finalLayout, cap, rockSet, parseResult.colors_by_id, 0);
  } else {
    clearEl(boardFinalEl);
  }

  renderSolverPanel(solveInfo, { addBottlesUsed: addBottles });
}

async function onVisionResult(result, debug) {
  // Keep the UI responsive: draw results immediately, then (optionally)
  // do a silent OCR pass for the add-bottle badge.
  isAnalyzing = false;

  parseResult = result;

  colorLabels = buildColorLabels(result.colors_by_id || {});
  setLegend(result.colors_by_id || {}, colorLabels);

  drawOverlay(debug);

  // Convert parsed contents -> stacks bottom->top
  const cap = parseResult.capacity;
  const stacks = parseResult.bottle_contents_ids.map((slots) => contentsToStacksTopBottom(slots, cap));

  renderOutput(stacks, null, null);

  // Allow re-running analysis.
  btnAnalyze.disabled = false;

  // If we can, silently OCR the add-bottles badge before enabling Solve.
  // (No UI mention; if it fails, we just continue with 0 extra bottles allowed.)
  const badgeBoxes = result.powerups?.badge_boxes || [];
  const shouldTryOcr = DEFAULT_USE_TESSERACT && Array.isArray(badgeBoxes) && badgeBoxes.length >= 3;

  if (shouldTryOcr) {
    btnSolve.disabled = true;
    setStatus('Finalizing…');
    try {
      const addBottles = await ocrAddBottlesFromCurrentImage(badgeBoxes);
      if (typeof addBottles === 'number' && Number.isFinite(addBottles) && addBottles >= 0) {
        parseResult.powerups = { ...(parseResult.powerups || {}), add_bottles: addBottles };
      }
    } catch (err) {
      // Silent failure (log to console only)
      console.warn('[badge OCR failed]', err);
    } finally {
      btnSolve.disabled = false;
      setStatus('Analysis complete.');
    }
  } else {
    btnSolve.disabled = false;
    setStatus('Analysis complete.');
  }
}

function getTessConfig() {
  // Official docs recommend jsDelivr by default.
  // (UI intentionally hides OCR tuning; this should "just work" when available.)
  return {
    workerPath: 'https://cdn.jsdelivr.net/npm/tesseract.js@v5.0.0/dist/worker.min.js',
    langPath: 'https://tessdata.projectnaptha.com/4.0.0',
    corePath: 'https://cdn.jsdelivr.net/npm/tesseract.js-core@v5.0.0',
  };
}

let tessWorkerPromise = null;

async function getTessWorker() {
  if (tessWorkerPromise) return tessWorkerPromise;

  tessWorkerPromise = (async () => {
    if (!window.Tesseract || !window.Tesseract.createWorker) {
      throw new Error('Tesseract.js not loaded (check CDN)');
    }

    const cfg = getTessConfig();

    const worker = await window.Tesseract.createWorker('eng', 1, {
      ...cfg,
      logger: (m) => {
        // only log useful status occasionally
        if (m?.status === 'recognizing text') return;
        // console.log('[tess]', m);
      },
    });

    // Restrict to digits
    await worker.setParameters({
      tessedit_char_whitelist: '0123456789',
    });

    return worker;
  })();

  return tessWorkerPromise;
}

function cropImageData(srcImageData, x, y, w, h) {
  // Clamp
  x = Math.max(0, Math.min(x, srcImageData.width - 1));
  y = Math.max(0, Math.min(y, srcImageData.height - 1));
  w = Math.max(1, Math.min(w, srcImageData.width - x));
  h = Math.max(1, Math.min(h, srcImageData.height - y));

  const tmp = document.createElement('canvas');
  tmp.width = w;
  tmp.height = h;
  const tctx = tmp.getContext('2d', { willReadFrequently: true });
  tctx.putImageData(srcImageData, -x, -y);
  return tctx.getImageData(0, 0, w, h);
}

function rgbToHsv01(r, g, b) {
  const rf = r / 255, gf = g / 255, bf = b / 255;
  const max = Math.max(rf, gf, bf);
  const min = Math.min(rf, gf, bf);
  const d = max - min;
  let h = 0;
  if (d !== 0) {
    if (max === rf) h = ((gf - bf) / d) % 6;
    else if (max === gf) h = (bf - rf) / d + 2;
    else h = (rf - gf) / d + 4;
    h *= 60;
    if (h < 0) h += 360;
  }
  const s = max === 0 ? 0 : d / max;
  const v = max;
  return { h, s, v };
}

function preprocessBadgeToCanvas(badgeImageData, scale = 6) {
  // Similar to Python approach:
  //   digits are white-ish (low saturation, high value)
  //   produce black digits on white background

  const w = badgeImageData.width;
  const h = badgeImageData.height;
  const src = badgeImageData.data;

  const outCanvas = document.createElement('canvas');
  outCanvas.width = w * scale;
  outCanvas.height = h * scale;

  // build a 1x badge binary image first
  const small = new ImageData(w, h);
  const dst = small.data;

  for (let i = 0; i < src.length; i += 4) {
    const r = src[i + 0];
    const g = src[i + 1];
    const b = src[i + 2];
    const { s, v } = rgbToHsv01(r, g, b);

    const isDigit = (s < 0.35) && (v > 0.67);
    const val = isDigit ? 0 : 255; // black digits, white bg
    dst[i + 0] = val;
    dst[i + 1] = val;
    dst[i + 2] = val;
    dst[i + 3] = 255;
  }

  const c1 = document.createElement('canvas');
  c1.width = w;
  c1.height = h;
  const c1ctx = c1.getContext('2d');
  c1ctx.putImageData(small, 0, 0);

  const ctx2 = outCanvas.getContext('2d');
  ctx2.imageSmoothingEnabled = false;
  ctx2.drawImage(c1, 0, 0, outCanvas.width, outCanvas.height);

  return outCanvas;
}

async function ocrOneBadge(worker, badgeBox) {
  const [x, y, w, h] = badgeBox;
  const crop = cropImageData(lastImageData, x, y, w, h);

  // preprocess
  const prep = preprocessBadgeToCanvas(crop, 6);

  const { data } = await worker.recognize(prep);
  const raw = (data?.text || '').trim();
  const digits = raw.replace(/\D/g, '');
  if (!digits) return null;
  return parseInt(digits, 10);
}

async function ocrAddBottlesFromCurrentImage(badgeBoxes) {
  // Badge order from the vision worker: [retries, shuffles, add-bottles]
  if (!badgeBoxes || !Array.isArray(badgeBoxes) || badgeBoxes.length < 3) return null;
  const box = badgeBoxes[2];
  if (!box) return null;

  const worker = await getTessWorker();
  return await ocrOneBadge(worker, box);
}

function getExtraBottlesFromPowerup() {
  const n = parseResult?.powerups?.add_bottles;
  if (typeof n === 'number' && Number.isFinite(n) && n >= 0) return n;
  return 0;
}

function addExtraEmptyBottles(stacks, extra) {
  const out = stacks.map((s) => s.slice());
  for (let k = 0; k < extra; k++) out.push([]);
  return out;
}

function solveAndRender() {
  if (!parseResult) return;
  const cap = parseResult.capacity;
  const rockSet = new Set(parseResult.rock_bottles || []);
  const maxStates = Math.max(1000, DEFAULT_MAX_STATES);

  const baseStacks = parseResult.bottle_contents_ids.map((slots) => contentsToStacksTopBottom(slots, cap));

  // Try to solve with the minimum number of extra empty bottles.
  // Don’t automatically allocate all allowed extras if they aren’t needed.
  const allowedExtra = getExtraBottlesFromPowerup();

  let chosenInfo = null;
  let chosenInitial = null;

  let lastInfo = null;
  let lastInitial = null;

  for (let e = 0; e <= allowedExtra; e++) {
    const init = addExtraEmptyBottles(baseStacks, e);

    // Note: rockSet indices refer to original bottles; extra bottles are not rock.
    const info = solveBfs(init, cap, rockSet, { maxStates, goalMode: 'mono' });
    info.extraUsed = e;
    info.extraAllowed = allowedExtra;

    lastInfo = info;
    lastInitial = init;

    if (info.solved) {
      chosenInfo = info;
      chosenInitial = init;
      break;
    }
  }

  if (!chosenInfo) {
    chosenInfo = lastInfo;
    chosenInitial = lastInitial;
  }

  let finalStacks = null;
  if (chosenInfo && chosenInfo.solved) {
    finalStacks = chosenInfo.moves.reduce((st, mv) => applyMove(st, mv), chosenInitial);
  }

  renderOutput(chosenInitial, finalStacks, chosenInfo);
}

function resetState() {
  imageBitmap = null;
  lastImageData = null;
  parseResult = null;
  colorLabels = null;
  btnAnalyze.disabled = true;
  btnSolve.disabled = true;
  clearOverlay();
  legend.innerHTML = '';
  if (legendTitle) legendTitle.textContent = 'Colors: — · Rock Bottles: —';
  if (boardDetectedEl) boardDetectedEl.innerHTML = '';
  if (boardFinalEl) boardFinalEl.innerHTML = '';
  if (solverTitleEl) solverTitleEl.textContent = 'Solver';
  if (solverMetaEl) solverMetaEl.textContent = '(click Solve)';
  if (movesEl) movesEl.innerHTML = '';
  setStatus('(load an image)');
}

async function loadImageFromFile(file) {
  resetState();
  const bmp = await createImageBitmap(file);
  imageBitmap = bmp;

  canvas.width = bmp.width;
  canvas.height = bmp.height;
  overlay.width = bmp.width;
  overlay.height = bmp.height;

  ctx.drawImage(bmp, 0, 0);
  lastImageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

  btnAnalyze.disabled = false;
  setStatus(`Loaded ${file.name} (${bmp.width}×${bmp.height}). Click Analyze.`);
}

async function analyze() {
  if (!imageBitmap || !lastImageData) return;

  btnAnalyze.disabled = true;
  btnSolve.disabled = true;
  isAnalyzing = true;
  resetAnalysisLogs();
  pushAnalysisLog('Analyzing… (first run may take a few seconds while OpenCV downloads/initializes)');

  const cvVer = DEFAULT_CV_VER;
  const eps = DEFAULT_DBSCAN_EPS;

  const worker = ensureVisionWorker(cvVer);

  // Copy buffer so we keep lastImageData available for OCR cropping
  const rgbaCopy = new Uint8ClampedArray(lastImageData.data);
  worker.postMessage({
    type: 'analyze',
    width: lastImageData.width,
    height: lastImageData.height,
    rgba: rgbaCopy.buffer,
    capacity: 4,
    dbscanEps: eps,
    cvVer,
    debugSamples: false,
  }, [rgbaCopy.buffer]);
}

elFile.addEventListener('change', async () => {
  const file = elFile.files?.[0];
  if (!file) return;
  await loadImageFromFile(file);
});

btnAnalyze.addEventListener('click', analyze);
btnSolve.addEventListener('click', solveAndRender);

