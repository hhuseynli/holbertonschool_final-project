/* Listing grid: AJAX filtering, sorting, pagination */

let currentPage = 1;

function getFilters() {
    const params = new URLSearchParams();
    const district = document.getElementById('filter-district').value;
    const rooms = document.getElementById('filter-rooms').value;
    const priceMin = document.getElementById('filter-price-min').value;
    const priceMax = document.getElementById('filter-price-max').value;
    const areaMin = document.getElementById('filter-area-min').value;
    const areaMax = document.getElementById('filter-area-max').value;
    const building = document.getElementById('filter-building').value;
    const sort = document.getElementById('sort-select').value;

    if (district) params.set('district', district);
    if (rooms) params.set('rooms', rooms);
    if (priceMin) params.set('price_min', priceMin);
    if (priceMax) params.set('price_max', priceMax);
    if (areaMin) params.set('area_min', areaMin);
    if (areaMax) params.set('area_max', areaMax);
    if (building) params.set('building_type', building);
    params.set('sort', sort);
    params.set('page', currentPage);

    return params;
}

function formatPrice(price) {
    if (!price) return '—';
    return price.toLocaleString('az-AZ') + ' AZN';
}

function scoreClass(score) {
    if (score >= 65) return 'score-high';
    if (score >= 40) return 'score-mid';
    return 'score-low';
}

function placeholderSvg(listing) {
    // Deterministic colour from listing id hash
    let h = 0;
    const id = listing.listing_id || '';
    for (let i = 0; i < id.length; i++) h = ((h << 5) - h + id.charCodeAt(i)) | 0;
    const hue = ((h % 360) + 360) % 360;
    const isNew = listing.building_type === 'new';
    const sat = isNew ? '45%' : '30%';
    const c1 = `hsl(${hue}, ${sat}, 35%)`;
    const c2 = `hsl(${(hue + 40) % 360}, ${sat}, 50%)`;
    const rooms = listing.rooms || '?';
    const area = listing.area_m2 ? listing.area_m2 + ' m\u00B2' : '';
    const type = isNew ? 'New build' : 'Old build';
    const district = (listing.district || '').substring(0, 18);
    return `data:image/svg+xml,${encodeURIComponent(`<svg xmlns="http://www.w3.org/2000/svg" width="400" height="260"><defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1"><stop offset="0%" stop-color="${c1}"/><stop offset="100%" stop-color="${c2}"/></linearGradient></defs><rect width="400" height="260" fill="url(#g)"/><text x="200" y="90" text-anchor="middle" fill="rgba(255,255,255,0.9)" font-family="system-ui,sans-serif" font-size="48" font-weight="700">${rooms} rm</text><text x="200" y="140" text-anchor="middle" fill="rgba(255,255,255,0.7)" font-family="system-ui,sans-serif" font-size="22">${area}</text><text x="200" y="175" text-anchor="middle" fill="rgba(255,255,255,0.6)" font-family="system-ui,sans-serif" font-size="18">${type}</text><text x="200" y="210" text-anchor="middle" fill="rgba(255,255,255,0.5)" font-family="system-ui,sans-serif" font-size="16">${district}</text></svg>`)}`;
}

function renderCard(listing) {
    const a = listing.analytics || {};
    const score = a.investment_score || 0;
    const photo = (listing.photos && listing.photos.length > 0) ? listing.photos[0] : null;
    const imgSrc = photo || placeholderSvg(listing);
    const appreciation = a.appreciation_12m_pct;

    let tags = '';
    if (appreciation != null) {
        const cls = appreciation >= 0 ? 'tag-positive' : 'tag-negative';
        tags += `<span class="card-tag ${cls}">${appreciation >= 0 ? '+' : ''}${appreciation.toFixed(1)}% 12m</span>`;
    }
    if (a.price_position && a.price_position !== 'unknown') {
        const cls = a.price_position === 'underpriced' ? 'tag-positive' :
                    a.price_position === 'overpriced' ? 'tag-negative' : 'tag-neutral';
        tags += `<span class="card-tag ${cls}">${a.price_position}</span>`;
    }
    if (a.dist_metro_km != null) {
        tags += `<span class="card-tag tag-neutral">Metro ${a.dist_metro_km} km</span>`;
    }

    return `
        <a href="/listing/${listing.listing_id}" class="listing-card">
            <div class="card-image">
                <img src="${imgSrc}" alt="" onerror="this.parentElement.innerHTML='No photo'">
                <span class="card-score ${scoreClass(score)}">${score}</span>
            </div>
            <div class="card-body">
                <div class="card-price">${formatPrice(listing.price_azn)}</div>
                <div class="card-price-m2">${listing.price_azn_m2 ? listing.price_azn_m2.toLocaleString('az-AZ') + ' AZN/m\u00B2' : ''}</div>
                <div class="card-specs">
                    <span>${listing.rooms || '?'} rooms</span>
                    <span>${listing.area_m2 ? listing.area_m2 + ' m\u00B2' : ''}</span>
                    <span>${listing.floor ? 'Floor ' + listing.floor + (listing.building_floors ? '/' + listing.building_floors : '') : ''}</span>
                </div>
                <div class="card-district">${listing.district || ''}</div>
                ${tags ? `<div class="card-analytics">${tags}</div>` : ''}
            </div>
        </a>
    `;
}

function renderPagination(page, totalPages) {
    const container = document.getElementById('pagination');
    if (totalPages <= 1) { container.innerHTML = ''; return; }

    let html = '';
    html += `<button class="page-btn" onclick="goToPage(${page - 1})" ${page <= 1 ? 'disabled' : ''}>Prev</button>`;

    const start = Math.max(1, page - 2);
    const end = Math.min(totalPages, page + 2);
    if (start > 1) html += `<button class="page-btn" onclick="goToPage(1)">1</button>`;
    if (start > 2) html += `<span class="page-btn" style="border:none;cursor:default">...</span>`;

    for (let i = start; i <= end; i++) {
        html += `<button class="page-btn ${i === page ? 'active' : ''}" onclick="goToPage(${i})">${i}</button>`;
    }

    if (end < totalPages - 1) html += `<span class="page-btn" style="border:none;cursor:default">...</span>`;
    if (end < totalPages) html += `<button class="page-btn" onclick="goToPage(${totalPages})">${totalPages}</button>`;
    html += `<button class="page-btn" onclick="goToPage(${page + 1})" ${page >= totalPages ? 'disabled' : ''}>Next</button>`;

    container.innerHTML = html;
}

async function fetchListings() {
    const grid = document.getElementById('card-grid');
    grid.innerHTML = '<div class="loading"><div class="spinner"></div>Loading...</div>';

    const params = getFilters();
    try {
        const resp = await fetch('/api/listings?' + params.toString());
        const data = await resp.json();

        document.getElementById('results-count').textContent = data.total + ' listings';

        if (data.listings.length === 0) {
            grid.innerHTML = '<div class="empty-state"><p>No listings match your filters</p></div>';
        } else {
            grid.innerHTML = data.listings.map(renderCard).join('');
        }
        renderPagination(data.page, data.total_pages);
    } catch (err) {
        grid.innerHTML = '<div class="empty-state"><p>Error loading listings</p></div>';
    }
}

function goToPage(page) {
    currentPage = page;
    fetchListings();
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// Event listeners
document.getElementById('apply-filters').addEventListener('click', () => {
    currentPage = 1;
    fetchListings();
});

document.getElementById('sort-select').addEventListener('change', () => {
    currentPage = 1;
    fetchListings();
});

// Enter key on filter inputs
document.querySelectorAll('.filter-input').forEach(input => {
    input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') { currentPage = 1; fetchListings(); }
    });
});

// Initial load
fetchListings();
