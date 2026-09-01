/* Interactive map with H3 hex overlay, metro stations, and cell popups */

const map = L.map('map').setView([40.41, 49.87], 11);

L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '&copy; <a href="https://openstreetmap.org">OSM</a>',
    maxZoom: 18,
}).addTo(map);

let hexLayer = null;
let metroLayer = null;
let currentColors = [];
let currentVmin = 0;
let currentVmax = 1;

function interpolateColor(colors, t) {
    t = Math.max(0, Math.min(1, t));
    const idx = t * (colors.length - 1);
    const lo = Math.floor(idx);
    const hi = Math.min(lo + 1, colors.length - 1);
    const frac = idx - lo;

    const cLo = colors[lo], cHi = colors[hi];
    // Parse hex colors
    function hexToRgb(hex) {
        hex = hex.replace('#', '');
        return [parseInt(hex.slice(0,2), 16), parseInt(hex.slice(2,4), 16), parseInt(hex.slice(4,6), 16)];
    }
    const [r1,g1,b1] = hexToRgb(cLo);
    const [r2,g2,b2] = hexToRgb(cHi);
    const r = Math.round(r1 + (r2-r1)*frac);
    const g = Math.round(g1 + (g2-g1)*frac);
    const b = Math.round(b1 + (b2-b1)*frac);
    return `rgb(${r},${g},${b})`;
}

function getColor(value) {
    if (currentVmax === currentVmin) return currentColors[0] || '#ccc';
    const t = (value - currentVmin) / (currentVmax - currentVmin);
    return interpolateColor(currentColors, t);
}

async function loadHexLayer() {
    const scenario = document.getElementById('map-scenario').value;
    const horizon = document.getElementById('map-horizon').value;
    const metric = document.getElementById('map-metric').value;

    try {
        const resp = await fetch(`/api/hexlayer?scenario=${scenario}&horizon=${horizon}&metric=${metric}`);
        const data = await resp.json();

        if (data.error) { console.error(data.error); return; }

        currentColors = data.colors || ['#ffffb2','#fed976','#feb24c','#fd8d3c','#f03b20','#bd0026'];
        currentVmin = data.vmin;
        currentVmax = data.vmax;

        if (hexLayer) map.removeLayer(hexLayer);

        hexLayer = L.geoJSON(data.geojson, {
            style: feature => ({
                fillColor: getColor(feature.properties.value),
                weight: 1,
                opacity: 0.6,
                color: '#333',
                fillOpacity: 0.55,
            }),
            onEachFeature: (feature, layer) => {
                layer.on('mouseover', e => {
                    e.target.setStyle({ weight: 2.5, fillOpacity: 0.8 });
                });
                layer.on('mouseout', e => {
                    hexLayer.resetStyle(e.target);
                });
                layer.on('click', () => {
                    const h3 = feature.properties.h3;
                    if (h3) loadCellDetails(h3);
                });

                const label = feature.properties.label || '';
                const h3 = feature.properties.h3 || '';
                layer.bindTooltip(`<b>${h3.slice(-6)}</b><br>${label}`, { sticky: true });
            }
        }).addTo(map);

        // Update legend
        updateLegend(data.caption);

    } catch (err) {
        console.error('Failed to load hex layer:', err);
    }
}

function updateLegend(caption) {
    const legend = document.getElementById('map-legend');
    const gradient = document.getElementById('legend-gradient');
    const stops = currentColors.map((c, i) => `${c} ${(i / (currentColors.length-1) * 100).toFixed(0)}%`).join(', ');
    gradient.style.background = `linear-gradient(90deg, ${stops})`;
    document.getElementById('legend-min').textContent = currentVmin.toFixed(1);
    document.getElementById('legend-max').textContent = currentVmax.toFixed(1);
    document.getElementById('legend-caption').textContent = caption || '';
    legend.style.display = 'block';
}

async function loadCellDetails(h3Id) {
    const scenario = document.getElementById('map-scenario').value;
    const popup = document.getElementById('cell-popup');
    const content = document.getElementById('cell-popup-content');

    try {
        const resp = await fetch(`/api/cell/${h3Id}?scenario=${scenario}`);
        const data = await resp.json();

        let html = `<div style="font-size:13px;">`;
        html += `<div style="font-weight:600; margin-bottom:8px;">${h3Id.slice(-8)}</div>`;

        if (data.current_price) {
            html += `<div>Current: <b>${data.current_price.toLocaleString()} AZN/m\u00B2</b></div>`;
        }
        html += `<div>Momentum: <b>${data.momentum_3m_pct > 0 ? '+' : ''}${data.momentum_3m_pct.toFixed(1)}%</b></div>`;

        if (data.forecast && data.forecast.length > 0) {
            html += `<div style="margin-top:8px; font-weight:600;">Forecast:</div>`;
            data.forecast.forEach(f => {
                html += `<div>${f.horizon_months}m: ${f.q10.toFixed(0)} — <b>${f.q50.toFixed(0)}</b> — ${f.q90.toFixed(0)}</div>`;
            });
        }

        const feats = data.features || {};
        if (feats.dist_metro_km != null) {
            html += `<div style="margin-top:8px;">Metro: ${feats.dist_metro_km.toFixed(1)} km</div>`;
        }
        if (feats.dist_centre_km != null) {
            html += `<div>Centre: ${feats.dist_centre_km.toFixed(1)} km</div>`;
        }
        if (feats.node_gravity != null) {
            html += `<div>Gravity: ${feats.node_gravity.toFixed(3)}</div>`;
        }

        html += `</div>`;
        content.innerHTML = html;
        popup.style.display = 'block';
    } catch (err) {
        console.error('Failed to load cell:', err);
    }
}

async function loadMetroStations() {
    try {
        const resp = await fetch('/api/metro_stations');
        const stations = await resp.json();

        if (metroLayer) map.removeLayer(metroLayer);
        metroLayer = L.layerGroup();

        const lineColors = { red: '#e74c3c', green: '#27ae60', purple: '#8e44ad' };

        stations.forEach(s => {
            const color = lineColors[s.line] || '#666';
            const isOpen = !!s.opened;
            const marker = L.circleMarker([s.lat, s.lon], {
                radius: 6,
                fillColor: isOpen ? color : 'white',
                color: color,
                weight: 2,
                fillOpacity: isOpen ? 0.8 : 0.3,
            });
            marker.bindTooltip(`<b>${s.name}</b><br>${s.line} line<br>${isOpen ? 'Opened: ' + s.opened : 'Planned'}`, { direction: 'top' });
            marker.addTo(metroLayer);
        });

        metroLayer.addTo(map);
    } catch (err) {
        console.error('Failed to load metro stations:', err);
    }
}

// Event listeners
document.getElementById('map-update').addEventListener('click', loadHexLayer);

// Initial load
loadHexLayer();
loadMetroStations();
