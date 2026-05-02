/* ═══════════════════════════════════════════════════════════════════
   InkSolver — Client Logic
   Handles upload, API calls, pipeline animation, result rendering
   ═══════════════════════════════════════════════════════════════════ */

(function () {
  'use strict';

  // ── DOM refs ──────────────────────────────────────────────────────
  const uploadZone       = document.getElementById('uploadZone');
  const fileInput        = document.getElementById('fileInput');
  const previewSection   = document.getElementById('previewSection');
  const previewImage     = document.getElementById('previewImage');
  const removeBtn        = document.getElementById('removeBtn');
  const solveBtn         = document.getElementById('solveBtn');
  const processingSection = document.getElementById('processingSection');
  const resultsSection   = document.getElementById('resultsSection');
  const solutionCard     = document.getElementById('solutionCard');
  const solutionBadge    = document.getElementById('solutionBadge');
  const solutionEquation = document.getElementById('solutionEquation');
  const solutionResult   = document.getElementById('solutionResult');
  const solutionError    = document.getElementById('solutionError');
  const symbolList       = document.getElementById('symbolList');
  const samplesGrid      = document.getElementById('samplesGrid');
  const errorToast       = document.getElementById('errorToast');
  const errorToastMsg    = document.getElementById('errorToastMessage');

  const imgOriginal     = document.getElementById('imgOriginal');
  const imgPreprocessed = document.getElementById('imgPreprocessed');
  const imgSegmented    = document.getElementById('imgSegmented');

  let selectedFile = null;

  // ── Upload Zone Events ────────────────────────────────────────────
  uploadZone.addEventListener('click', () => fileInput.click());

  fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
      handleFile(e.target.files[0]);
    }
  });

  // drag and drop
  uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadZone.classList.add('drag-over');
  });

  uploadZone.addEventListener('dragleave', () => {
    uploadZone.classList.remove('drag-over');
  });

  uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadZone.classList.remove('drag-over');
    if (e.dataTransfer.files.length > 0) {
      handleFile(e.dataTransfer.files[0]);
    }
  });

  // clipboard paste
  document.addEventListener('paste', (e) => {
    const items = e.clipboardData?.items;
    if (!items) return;
    for (const item of items) {
      if (item.type.startsWith('image/')) {
        const file = item.getAsFile();
        if (file) handleFile(file);
        break;
      }
    }
  });

  // remove
  removeBtn.addEventListener('click', () => {
    resetState();
  });

  // solve
  solveBtn.addEventListener('click', () => {
    if (selectedFile) solveImage(selectedFile);
  });

  // ── File Handling ─────────────────────────────────────────────────
  function handleFile(file) {
    if (!file.type.startsWith('image/')) {
      showError('Please upload an image file (PNG, JPG, BMP)');
      return;
    }

    selectedFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
      previewImage.src = e.target.result;
      previewSection.classList.add('visible');
      resultsSection.classList.remove('visible');
      processingSection.classList.remove('visible');
    };
    reader.readAsDataURL(file);
  }

  function resetState() {
    selectedFile = null;
    fileInput.value = '';
    previewSection.classList.remove('visible');
    processingSection.classList.remove('visible');
    resultsSection.classList.remove('visible');
  }

  // ── Pipeline Animation ────────────────────────────────────────────
  const PIPELINE_STEPS = ['preprocess', 'segment', 'classify', 'parse', 'solve'];

  function resetPipeline() {
    document.querySelectorAll('.pipeline__step').forEach((el) => {
      el.classList.remove('active', 'done');
    });
  }

  function animatePipeline() {
    resetPipeline();
    processingSection.classList.add('visible');
    resultsSection.classList.remove('visible');

    let i = 0;
    const interval = setInterval(() => {
      if (i > 0) {
        const prev = document.querySelector(`.pipeline__step[data-step="${PIPELINE_STEPS[i - 1]}"]`);
        if (prev) { prev.classList.remove('active'); prev.classList.add('done'); }
      }
      if (i < PIPELINE_STEPS.length) {
        const curr = document.querySelector(`.pipeline__step[data-step="${PIPELINE_STEPS[i]}"]`);
        if (curr) curr.classList.add('active');
        i++;
      } else {
        clearInterval(interval);
      }
    }, 600);

    return interval;
  }

  function finishPipeline(intervalId) {
    clearInterval(intervalId);
    document.querySelectorAll('.pipeline__step').forEach((el) => {
      el.classList.remove('active');
      el.classList.add('done');
    });
  }

  // ── API Call ──────────────────────────────────────────────────────
  async function solveImage(file) {
    solveBtn.disabled = true;
    solveBtn.textContent = '⏳ Processing…';

    const pipelineInterval = animatePipeline();

    try {
      const formData = new FormData();
      formData.append('image', file);

      const res = await fetch('/api/solve', {
        method: 'POST',
        body: formData,
      });

      const data = await res.json();
      finishPipeline(pipelineInterval);

      if (data.success) {
        renderResults(data);
      } else {
        showError(data.error || 'Something went wrong');
      }
    } catch (err) {
      finishPipeline(pipelineInterval);
      showError('Network error — is the server running?');
      console.error(err);
    } finally {
      solveBtn.disabled = false;
      solveBtn.textContent = '⚡ Solve Equation';
    }
  }

  async function solveSample(filename) {
    // show pipeline animation
    previewSection.classList.remove('visible');
    solveBtn.disabled = true;
    const pipelineInterval = animatePipeline();

    try {
      const res = await fetch('/api/solve-sample', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filename }),
      });

      const data = await res.json();
      finishPipeline(pipelineInterval);

      if (data.success) {
        renderResults(data);
      } else {
        showError(data.error || 'Something went wrong');
      }
    } catch (err) {
      finishPipeline(pipelineInterval);
      showError('Network error — is the server running?');
      console.error(err);
    } finally {
      solveBtn.disabled = false;
    }
  }

  // ── Render Results ────────────────────────────────────────────────
  function renderResults(data) {
    resultsSection.classList.add('visible');

    // solution card
    if (data.type === 'error' || data.result?.error) {
      solutionBadge.className = 'solution-card__badge solution-card__badge--error';
      solutionBadge.textContent = '✕ Recognition Failed';
      solutionEquation.textContent = data.equation || data.equations?.join(' , ') || '';
      solutionResult.style.display = 'none';
      solutionError.style.display = 'block';

      // build a friendlier error message
      const rawError = data.result?.error || 'Could not solve';
      const avgConf = data.steps?.avg_confidence;
      let errorHtml = rawError;
      if (avgConf !== undefined) {
        errorHtml += `<br><small style="color: var(--text-muted); margin-top: 8px; display: inline-block;">Average confidence: ${avgConf}%</small>`;
      }
      solutionError.innerHTML = errorHtml;
    } else {
      solutionBadge.className = 'solution-card__badge solution-card__badge--success';
      solutionBadge.textContent = '✓ Solved';
      solutionError.style.display = 'none';
      solutionResult.style.display = 'block';

      if (data.mode === 'system') {
        solutionEquation.textContent = data.equations.join('  ,  ');
        const solParts = Object.entries(data.result.solutions)
          .map(([v, val]) => `${v} = ${val}`)
          .join('  ,  ');
        solutionResult.textContent = solParts;
      } else {
        solutionEquation.textContent = data.equation;

        if (data.type === 'arithmetic') {
          solutionResult.textContent = `= ${data.result.value}`;
        } else if (data.type === 'equation') {
          const sols = data.result.solutions.map(s => `${data.result.variable} = ${s}`).join(', ');
          solutionResult.textContent = sols;
        } else if (data.type === 'verification') {
          solutionResult.textContent = data.result.is_true ? '✓ TRUE' : '✕ FALSE';
        } else if (data.type === 'multi_variable') {
          solutionResult.textContent = data.result.simplified;
        }
      }
    }

    // images
    if (data.images) {
      imgOriginal.src     = 'data:image/png;base64,' + data.images.original;
      imgPreprocessed.src = 'data:image/png;base64,' + data.images.preprocessed;
      imgSegmented.src    = 'data:image/png;base64,' + data.images.segmented;

      // reset to show original
      document.querySelectorAll('.image-tab').forEach(t => t.classList.remove('active'));
      document.querySelector('.image-tab[data-target="imgOriginal"]').classList.add('active');
      imgOriginal.classList.remove('hidden');
      imgPreprocessed.classList.add('hidden');
      imgSegmented.classList.add('hidden');
    }

    // symbols
    symbolList.innerHTML = '';
    const recognition = data.mode === 'system'
      ? data.steps?.recognition?.flatMap(r => r.symbols) || []
      : data.steps?.recognition || [];

    recognition.forEach((sym, idx) => {
      const chip = document.createElement('div');
      chip.className = 'symbol-chip';

      const conf = sym.confidence;
      const barClass = conf >= 90 ? 'high' : conf >= 70 ? 'medium' : 'low';

      chip.innerHTML = `
        <span class="symbol-chip__char">${escapeHtml(sym.symbol)}</span>
        <div class="symbol-chip__bar-wrapper">
          <div class="symbol-chip__bar ${barClass}" style="width: 0%"></div>
        </div>
        <span class="symbol-chip__conf">${conf}%</span>
      `;
      symbolList.appendChild(chip);

      // animate bar after a short delay
      requestAnimationFrame(() => {
        setTimeout(() => {
          chip.querySelector('.symbol-chip__bar').style.width = conf + '%';
        }, 100 + idx * 80);
      });
    });

    // scroll to results
    setTimeout(() => {
      solutionCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 200);
  }

  // ── Image Tabs ────────────────────────────────────────────────────
  document.querySelectorAll('.image-tab').forEach((tab) => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.image-tab').forEach(t => t.classList.remove('active'));
      tab.classList.add('active');

      const targetId = tab.dataset.target;
      [imgOriginal, imgPreprocessed, imgSegmented].forEach((img) => {
        img.classList.toggle('hidden', img.id !== targetId);
      });
    });
  });

  // ── Error Toast ───────────────────────────────────────────────────
  let toastTimer = null;

  function showError(message) {
    errorToastMsg.textContent = message;
    errorToast.classList.add('visible');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => {
      errorToast.classList.remove('visible');
    }, 5000);
  }

  // ── Utility ───────────────────────────────────────────────────────
  function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
  }

  // ── Load Samples ──────────────────────────────────────────────────
  async function loadSamples() {
    try {
      const res = await fetch('/api/samples');
      const data = await res.json();

      if (!data.samples || data.samples.length === 0) {
        samplesGrid.innerHTML = '<p style="color: var(--text-muted); font-size: 0.85rem; grid-column: 1/-1; text-align: center;">No sample images found</p>';
        return;
      }

      samplesGrid.innerHTML = '';
      data.samples.forEach((sample) => {
        const card = document.createElement('div');
        card.className = 'sample-card';
        card.innerHTML = `
          <img class="sample-card__thumb" src="data:image/png;base64,${sample.thumbnail}" alt="${escapeHtml(sample.name)}">
          <div class="sample-card__name">${escapeHtml(sample.name)}</div>
        `;
        card.addEventListener('click', () => {
          solveSample(sample.filename);
          // scroll to pipeline
          setTimeout(() => {
            processingSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
          }, 100);
        });
        samplesGrid.appendChild(card);
      });
    } catch (err) {
      console.warn('Could not load samples:', err);
    }
  }

  // ── Init ──────────────────────────────────────────────────────────
  loadSamples();

})();
