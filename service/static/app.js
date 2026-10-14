'use strict';

const SEV_COLOR = { 1: '#f6c445', 2: '#ee7b30', 3: '#9d1414' };
// Более светлый тон той же степени — идёт на обводку контура, а не на заливку:
// тонкий 1px контур цветом заливки на тёмной подложке карты почти не читался,
// яркая обводка + более плотная заливка дают контурам заметно больше «веса».
const SEV_BORDER = { 1: '#ffe066', 2: '#ffa236', 3: '#ff3b30' };
const $ = (id) => document.getElementById(id);

const map = L.map('map', { preferCanvas: true, zoomControl: true }).setView([49, 44], 6);

// Подложка из OSM. Если интернета нет, карта останется тёмной, но векторные
// слои и вся аналитика продолжат работать — они приходят с нашего сервера.
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
  maxZoom: 18,
  attribution: '&copy; OpenStreetMap',
}).addTo(map);

const layers = {
  footprints: L.layerGroup().addTo(map),
  contours: L.layerGroup().addTo(map),
  hotspots: L.layerGroup().addTo(map),
  selection: L.layerGroup().addTo(map),
};

let catalogExtent = null;
let footprintsLoaded = false;

// ---------------------------------------------------------------- утилиты
function fmt(n, digits = 1) {
  if (n === null || n === undefined || Number.isNaN(n)) return '—';
  return Number(n).toLocaleString('ru-RU', {
    minimumFractionDigits: digits, maximumFractionDigits: digits,
  });
}

function currentBbox() {
  const parts = $('bbox').value.split(',').map((s) => parseFloat(s.trim()));
  if (parts.length !== 4 || parts.some(Number.isNaN)) return null;
  return parts;
}

function setBbox(bounds) {
  const b = [bounds.getWest(), bounds.getSouth(), bounds.getEast(), bounds.getNorth()];
  $('bbox').value = b.map((v) => v.toFixed(4)).join(', ');
  drawSelection(b);
}

function drawSelection(b) {
  layers.selection.clearLayers();
  L.rectangle([[b[1], b[0]], [b[3], b[2]]], {
    color: '#ff6a1a', weight: 2, dashArray: '6 4', fill: false, interactive: false,
  }).addTo(layers.selection);
}

function queryString() {
  const b = currentBbox();
  const params = new URLSearchParams({ bbox: b.join(','), date_from: $('date-from').value, date_to: $('date-to').value });
  return params.toString();
}

function setStatus(text, isError = false) {
  const el = $('status');
  el.textContent = text;
  el.classList.toggle('status--err', isError);
}

// ---------------------------------------------------------------- старт
async function init() {
  try {
    const res = await fetch('/api/catalog');
    if (!res.ok) throw new Error(await res.text());
    catalogExtent = await res.json();
  } catch (err) {
    setStatus('Не удалось получить каталог: ' + err.message, true);
    return;
  }

  const [w, s, e, n] = catalogExtent.bbox;
  map.fitBounds([[s, w], [n, e]]);
  $('date-from').value = catalogExtent.date_from;
  $('date-to').value = catalogExtent.date_to;
  $('date-from').min = $('date-to').min = catalogExtent.date_from;
  $('date-from').max = $('date-to').max = catalogExtent.date_to;
  $('coverage-hint').textContent =
    `Покрытие данными: ${w.toFixed(2)}…${e.toFixed(2)}° в.д., ${s.toFixed(2)}…${n.toFixed(2)}° с.ш. · `
    + `${catalogExtent.n_af_chips} чипов AF, ${catalogExtent.n_bs_chips} чипов BS`;
  setBbox(map.getBounds());
  buildPresets();

  const badge = $('source-badge');
  if (catalogExtent.source === 'model') {
    badge.textContent = 'источник: предсказания модели';
    badge.className = 'badge badge--model';
  } else {
    badge.textContent = 'источник: эталонные маски';
    badge.className = 'badge badge--truth';
    badge.title = catalogExtent.source_note || '';
  }
  setStatus('Готово к запросу.');
}

function buildPresets() {
  const years = [];
  const from = new Date(catalogExtent.date_from).getFullYear();
  const to = new Date(catalogExtent.date_to).getFullYear();
  for (let y = from; y <= to; y += 1) years.push(y);
  const box = $('presets');
  box.innerHTML = '';
  const all = document.createElement('button');
  all.className = 'btn btn--ghost';
  all.type = 'button';
  all.textContent = 'весь период';
  all.onclick = () => {
    $('date-from').value = catalogExtent.date_from;
    $('date-to').value = catalogExtent.date_to;
  };
  box.appendChild(all);
  years.forEach((y) => {
    const b = document.createElement('button');
    b.className = 'btn btn--ghost';
    b.type = 'button';
    b.textContent = y;
    b.onclick = () => {
      $('date-from').value = `${y}-01-01`;
      $('date-to').value = `${y}-12-31`;
    };
    box.appendChild(b);
  });
}

// ---------------------------------------------------------------- рисование рамки
let drawing = false;
let drawStart = null;
let drawRect = null;

$('btn-draw').onclick = () => {
  drawing = !drawing;
  $('btn-draw').classList.toggle('active', drawing);
  $('draw-hint').hidden = !drawing;
  map.dragging[drawing ? 'disable' : 'enable']();
};

map.on('mousedown', (ev) => {
  if (!drawing) return;
  drawStart = ev.latlng;
  drawRect = L.rectangle([drawStart, drawStart], { color: '#ff6a1a', weight: 2, fill: false }).addTo(map);
});

map.on('mousemove', (ev) => {
  if (!drawing || !drawStart) return;
  drawRect.setBounds(L.latLngBounds(drawStart, ev.latlng));
});

map.on('mouseup', (ev) => {
  if (!drawing || !drawStart) return;
  const bounds = L.latLngBounds(drawStart, ev.latlng);
  drawStart = null;
  map.removeLayer(drawRect);
  drawRect = null;
  drawing = false;
  $('btn-draw').classList.remove('active');
  $('draw-hint').hidden = true;
  map.dragging.enable();
  setBbox(bounds);
});

$('btn-view').onclick = () => setBbox(map.getBounds());

// ---------------------------------------------------------------- слои
$('chk-footprints').onchange = async (ev) => {
  if (!ev.target.checked) { layers.footprints.clearLayers(); return; }
  if (!footprintsLoaded) {
    const fc = await (await fetch('/api/chips?kind=bs')).json();
    L.geoJSON(fc, {
      // диамантово-голубой пунктир: явно видно на тёмной теме, но по цвету и
      // штриху сразу читается как «здесь есть данные», а не как гарь
      // (та закрашена сплошным жёлтым/оранжевым/красным)
      style: { color: '#59e6d9', weight: 1.5, dashArray: '5 4', fillColor: '#59e6d9', fillOpacity: 0.12 },
      onEachFeature: (f, layer) => layer.bindPopup(
        `<b>${f.properties.chip_id}</b><br>${f.properties.date_start} → ${f.properties.date_end}`
        + `<br>событие: ${f.properties.fire_event_id || '—'}`),
    }).addTo(layers.footprints);
    footprintsLoaded = true;
  } else {
    map.addLayer(layers.footprints);
  }
};

// ---------------------------------------------------------------- запрос
$('btn-run').onclick = async () => {
  const bbox = currentBbox();
  if (!bbox) { setStatus('Проверьте bbox: нужно четыре числа через запятую.', true); return; }
  drawSelection(bbox);

  const btn = $('btn-run');
  btn.disabled = true;
  setStatus('Считаем… первый запрос по новой территории занимает время: модель прогоняется по чипам.');
  const started = performance.now();

  try {
    const res = await fetch('/api/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        bbox,
        date_from: $('date-from').value,
        date_to: $('date-to').value,
        include_hotspots: $('chk-hotspots').checked,
        include_contours: $('chk-contours').checked,
      }),
    });
    if (!res.ok) {
      const body = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(body.detail || res.statusText);
    }
    const data = await res.json();
    render(data);
    const secs = ((performance.now() - started) / 1000).toFixed(1);
    let note = `Готово за ${secs} с · ${data.summary.chips.n_af} чипов AF, ${data.summary.chips.n_bs} чипов BS.`;
    if (data.hotspots_truncated) note += ' Термоточек больше, чем показано, — полный набор в выгрузке.';
    if (data.summary.chips_truncated) note += ' Территория очень большая, часть чипов не обработана.';
    setStatus(note);
  } catch (err) {
    setStatus('Ошибка: ' + err.message, true);
  } finally {
    btn.disabled = false;
  }
};

function render(data) {
  layers.contours.clearLayers();
  layers.hotspots.clearLayers();

  L.geoJSON(data.burn_contours, {
    style: (f) => ({
      color: SEV_BORDER[f.properties.severity_class],
      weight: 2.5,
      opacity: 1,
      fillColor: SEV_COLOR[f.properties.severity_class],
      fillOpacity: 0.7,
    }),
    onEachFeature: (f, layer) => {
      const p = f.properties;
      layer.bindPopup(
        `<b>Контур ${p.contour_id}</b><br>`
        + `степень: ${p.severity_class} — ${p.severity_label}<br>`
        + `площадь: ${fmt(p.area_ha, 2)} га (${p.pixels} пикс. × ${p.gsd_m} м)<br>`
        + `чип: ${p.chip_id}<br>`
        + `снимки: ${p.date_pre} → ${p.date_post}`);
    },
  }).addTo(layers.contours);

  L.geoJSON(data.hotspots, {
    pointToLayer: (f, latlng) => L.circleMarker(latlng, {
      // яркое ядро + золотое кольцо (цвет --ember темы) - на тёмной карте
      // читается как раскалённая точка, а не тусклый кружок
      radius: 5, color: '#ffd54a', weight: 2, fillColor: '#ff3d00', fillOpacity: 0.95,
    }),
    onEachFeature: (f, layer) => {
      const p = f.properties;
      layer.bindPopup(
        `<b>Термоточка</b><br>дата: ${p.acq_datetime}<br>`
        + `спутник: ${p.satellite || '—'}<br>чип: ${p.chip_id}<br>`
        + `пиксель ${p.gsd_m} м = ${fmt(p.pixel_area_ha, 2)} га`);
    },
  }).addTo(layers.hotspots);

  renderReport(data.summary);
}

function renderReport(s) {
  $('report-block').hidden = false;
  $('export-block').hidden = false;
  $('total-ha').textContent = fmt(s.total_burn_area_ha, 1);

  $('sev-table').innerHTML = s.by_severity.map((item) => `
    <tr>
      <td><span class="swatch" style="background:${SEV_COLOR[item.severity_class]}"></span>${item.severity_class} — ${item.severity_label}</td>
      <td class="num">${fmt(item.area_ha, 1)} га</td>
      <td class="pct">${fmt(item.share_pct, 1)}%</td>
    </tr>`).join('');

  const facts = [
    ['Контуров гари', s.total_contours.toLocaleString('ru-RU')],
    ['Термоточек', s.hotspots_count.toLocaleString('ru-RU')],
    ['Площадь термоточек', fmt(s.hotspot_area_ha, 1) + ' га'],
    ['Чипов AF / BS', `${s.chips.n_af} / ${s.chips.n_bs}`],
    ['Период', `${s.query.date_from} … ${s.query.date_to}`],
    ['Источник', s.source === 'model' ? 'предсказания модели' : 'эталонные маски'],
  ];
  $('facts').innerHTML = facts.map(([k, v]) => `<div><dt>${k}</dt><dd>${v}</dd></div>`).join('');

  $('method-body').innerHTML =
    `<code>${s.method.area_formula}</code><br>`
    + `Разрешение: ${s.method.gsd_m.join(', ') || '—'} м.<br>`
    + `Измеряется в: ${s.method.measured_in}.<br>`
    + `Выгрузка в: ${s.method.output_crs}.<br>`
    + `Проверка: ${s.method.validation}.`;

  const qs = queryString();
  $('dl-contours').href = '/api/export/contours.geojson?' + qs;
  $('dl-hotspots').href = '/api/export/hotspots.geojson?' + qs;
  $('dl-json').href = '/api/export/report.json?' + qs;
  $('dl-csv').href = '/api/export/report.csv?' + qs;
}

init();

// ---------------------------------------------------------------- угольки (декор)
// Чисто декоративный слой в шапке — тлеющие угольки, поднимающиеся вверх.
// Контейнер .embers имеет overflow:hidden и pointer-events:none, так что
// это никак не мешает управлению интерфейсом.
(function spawnEmbers() {
  const box = document.querySelector('.embers');
  if (!box) return;
  const N = 14;
  for (let i = 0; i < N; i += 1) {
    const s = document.createElement('span');
    s.className = 'ember';
    s.style.left = `${(Math.random() * 100).toFixed(1)}%`;
    s.style.setProperty('--drift', `${(Math.random() * 30 - 15).toFixed(0)}px`);
    s.style.animationDuration = `${(3 + Math.random() * 3).toFixed(2)}s`;
    s.style.animationDelay = `${(Math.random() * 4).toFixed(2)}s`;
    box.appendChild(s);
  }
})();
