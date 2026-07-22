const MODEL_ID = 'voice_lab/nana';
const SPEAKER = 'nana';

let dramaStyles = [];

const $ = (id) => document.getElementById(id);

async function fetchStatus() {
  const res = await fetch('/api/v1/status');
  if (!res.ok) throw new Error('status ' + res.status);
  return res.json();
}

async function loadModel() {
  const fd = new FormData();
  fd.append('model_id', MODEL_ID);
  const res = await fetch('/api/v1/models/load', { method: 'POST', body: fd });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || 'load failed');
  return data;
}

function fillDrama(styles) {
  dramaStyles = Array.isArray(styles) ? styles : [];
  const sel = $('dramaSel');
  sel.innerHTML = '';
  if (!dramaStyles.length) {
    sel.innerHTML = '<option value="">（无戏路配置）</option>';
    return;
  }
  dramaStyles.forEach((s, i) => {
    const opt = document.createElement('option');
    opt.value = s.id;
    opt.textContent = s.label || s.id;
    if (i === 0) opt.selected = true;
    sel.appendChild(opt);
  });
  updateDramaHint();
}

function updateDramaHint() {
  const id = $('dramaSel').value;
  const s = dramaStyles.find((x) => x.id === id);
  $('dramaHint').textContent = s?.instruct || '';
}

function setStatus(loaded, model, loading) {
  const dot = $('statusDot');
  const text = $('statusText');
  const meta = $('modelMeta');
  const gen = $('genBtn');
  const reload = $('reloadBtn');

  if (loading) {
    dot.className = 'dot off';
    text.textContent = '模型加载中…';
    gen.disabled = true;
    reload.hidden = true;
    return;
  }

  if (loaded) {
    dot.className = 'dot on';
    text.textContent = '模型已就绪';
    meta.textContent = `当前：${model || MODEL_ID} · 说话人 ${SPEAKER}`;
    gen.disabled = false;
    reload.hidden = true;
  } else {
    dot.className = 'dot off';
    text.textContent = '模型未加载';
    meta.textContent = '点击重新加载或等待启动预加载';
    gen.disabled = true;
    reload.hidden = false;
  }
}

async function refresh() {
  try {
    const st = await fetchStatus();
    fillDrama(st.drama_styles);
    setStatus(st.loaded, st.model, st.loading);
  } catch (e) {
    $('statusText').textContent = '无法连接服务';
    $('modelMeta').textContent = String(e);
    $('genBtn').disabled = true;
  }
}

async function generate() {
  const text = ($('text').value || '').trim();
  if (!text) {
    $('err').hidden = false;
    $('err').textContent = '请输入文字';
    return;
  }

  const style = dramaStyles.find((x) => x.id === $('dramaSel').value);
  const instruct = style?.instruct || '';

  $('err').hidden = true;
  $('busy').hidden = false;
  $('genBtn').disabled = true;
  $('player').hidden = true;

  const payload = {
    request_id: `demo-${Date.now()}`,
    trace_id: `demo-trace-${Date.now()}`,
    session_id: 'tts-demo',
    text,
    stream: false,
    voice: {
      mode: 'custom',
      speaker_id: SPEAKER,
      language: 'Chinese',
      instruct,
    },
    audio_config: {
      format: 'wav',
      sample_rate: 24000,
      channels: 1,
    },
    metadata: {},
  };

  try {
    const res = await fetch('/api/v1/audio/speech', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || 'generate failed');
    const b64 = data.audio_base64;
    if (!b64) throw new Error('响应无 audio_base64');
    $('player').src = 'data:audio/wav;base64,' + b64;
    $('player').hidden = false;
    $('player').play().catch(() => {});
  } catch (e) {
    $('err').hidden = false;
    $('err').textContent = String(e.message || e);
  } finally {
    $('busy').hidden = true;
    $('genBtn').disabled = false;
  }
}

$('dramaSel').addEventListener('change', updateDramaHint);
$('genBtn').addEventListener('click', generate);
$('reloadBtn').addEventListener('click', async () => {
  $('reloadBtn').disabled = true;
  setStatus(false, null, true);
  try {
    await loadModel();
    await refresh();
  } catch (e) {
    $('err').hidden = false;
    $('err').textContent = String(e.message || e);
    setStatus(false, null, false);
  } finally {
    $('reloadBtn').disabled = false;
  }
});

refresh();
setInterval(refresh, 8000);
