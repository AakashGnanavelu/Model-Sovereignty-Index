async function loadModels() {
  if (window.location.protocol === "file:") {
    throw new Error(
      "Cannot load data/models.json from a file:// URL. " +
        "Browsers block fetch() on local files. " +
        "Serve the project from its root directory instead, e.g. " +
        "python3 -m http.server 8000 then open http://localhost:8000"
    );
  }

  const response = await fetch("data/models.json");
  if (!response.ok) {
    throw new Error(
      `Failed to load data/models.json (HTTP ${response.status}). ` +
        "Ensure you are serving the project root, not a subfolder."
    );
  }
  const raw = await response.json();
  return normalizeModels(raw);
}

function normalizeModelRecord(m, org = {}) {
  const modelId = m.model_id || m.id || "";
  const nameVal = m.name || (modelId ? modelId.split("/").pop() : "Unknown");
  const organisation = m.author || org.name || "Unknown";

  return {
    id: modelId,
    name: nameVal,
    organisation,
    organisation_type: org.organisation_type || m.organisation_type || "–",
    country: org.country || m.country || "–",
    overall_score: m.value ?? m.overall_score,
    categories: m.categories || {},
    sources: m.sources || [],
    evidence: m.evidence || {},
    explain: m.explanation || m.explain || "",
    used_ground_truth: m.used_ground_truth || false,
    org_aggregate: org.aggregate || null,
  };
}

function normalizeModels(rawModels) {
  if (!Array.isArray(rawModels)) return [];

  // Organisation-grouped structure from generate_data.py:
  // [{ organisation: {...}, models: [...] }, ...]
  if (
    rawModels.length > 0 &&
    rawModels[0].organisation &&
    Array.isArray(rawModels[0].models)
  ) {
    return rawModels.flatMap((entry) => {
      const org = entry.organisation || {};
      return (entry.models || [])
        .map((m) => normalizeModelRecord(m, org))
        .filter(Boolean);
    });
  }

  return rawModels
    .map((item) => {
      if (!item || typeof item !== "object") return null;

      // Wrapped structure: { model, score: { model_id, author, value, ... }, explain }
      if (item.score && typeof item.score === "object") {
        const score = item.score;
        const modelId = score.model_id || item.model || "";
        const organisation = score.author || (modelId.includes("/") ? modelId.split("/")[0] : "Unknown");
        const nameVal = item.model || (modelId ? modelId.split("/").pop() : "Unknown");

        return normalizeModelRecord(
          {
            model_id: modelId,
            author: organisation,
            value: score.value,
            categories: score.categories,
            sources: score.sources,
            evidence: score.evidence || item.evidence,
            explanation: item.explain || score.explanation,
            country: score.country || item.country,
            organisation_type: score.organisation_type || item.organisation_type,
          },
          {
            name: organisation,
            country: score.country || item.country,
            organisation_type: score.organisation_type || item.organisation_type,
          }
        );
      }

      // Flat/legacy structure: { model_id, author, value, categories, ... }
      if ("model_id" in item || "value" in item) {
        return normalizeModelRecord(item, {
          name: item.author || item.organisation,
          country: item.country,
          organisation_type: item.organisation_type,
        });
      }

      return null;
    })
    .filter(Boolean);
}

function formatScore(score) {
  if (score == null || Number.isNaN(score)) return "–";
  return score.toFixed(1);
}

function scoreClass(score) {
  if (score == null) return "score-chip--medium";
  if (score >= 70) return "score-chip--high";
  if (score >= 55) return "score-chip--medium";
  return "score-chip--low";
}

function computeSummaryStats(models) {
  const count = models.length;
  const scores = models.map((m) => m.overall_score).filter((x) => typeof x === "number");
  const avgScore =
    scores.length > 0
      ? scores.reduce((sum, x) => sum + x, 0) / scores.length
      : null;
  const orgs = new Set(models.map((m) => m.organisation));
  const countries = new Set(models.map((m) => m.country));

  return {
    count,
    avgScore,
    orgCount: orgs.size,
    countryCount: countries.size,
  };
}

function renderSummary(models) {
  const stats = computeSummaryStats(models);
  document.getElementById("summary-model-count").textContent = String(stats.count);
  document.getElementById("summary-org-count").textContent = String(stats.orgCount);
  document.getElementById("summary-country-count").textContent = String(
    stats.countryCount
  );
  document.getElementById("summary-avg-score").textContent =
    stats.avgScore != null ? stats.avgScore.toFixed(1) : "–";
}

function applyFiltersAndSort(models, { search, sortBy, orgType }) {
  let filtered = models;

  if (search) {
    const q = search.toLowerCase();
    filtered = filtered.filter((m) => {
      return (
        m.name.toLowerCase().includes(q) ||
        (m.organisation && m.organisation.toLowerCase().includes(q)) ||
        (m.country && m.country.toLowerCase().includes(q)) ||
        (m.organisation_type && m.organisation_type.toLowerCase().includes(q))
      );
    });
  }

  if (orgType) {
    filtered = filtered.filter((m) => m.organisation_type === orgType);
  }

  const sorted = [...filtered];
  switch (sortBy) {
    case "overall_score_asc":
      sorted.sort((a, b) => (a.overall_score || 0) - (b.overall_score || 0));
      break;
    case "name_asc":
      sorted.sort((a, b) => a.name.localeCompare(b.name));
      break;
    case "organisation_asc":
      sorted.sort((a, b) => a.organisation.localeCompare(b.organisation));
      break;
    case "overall_score_desc":
    default:
      sorted.sort((a, b) => (b.overall_score || 0) - (a.overall_score || 0));
      break;
  }

  return sorted;
}

function computeGroups(models, sortBy = "overall_score_desc", groupBy = "organisation") {
  const map = new Map();
  models.forEach((m) => {
    const key =
      groupBy === "country"
        ? m.country || "–"
        : m.organisation || "Unknown";
    if (!map.has(key)) map.set(key, []);
    map.get(key).push(m);
  });

  const groups = Array.from(map.entries()).map(([key, list]) => {
    const scores = list.map((x) => x.overall_score).filter((x) => typeof x === "number");
    const avg = scores.length ? scores.reduce((a, b) => a + b, 0) / scores.length : null;
    const sample = list[0] || {};
    return {
      key,
      groupBy,
      organisation: groupBy === "organisation" ? key : sample.organisation || "–",
      country: groupBy === "country" ? key : sample.country || "–",
      models: sortModels(list, sortBy),
      count: list.length,
      avgScore: avg,
    };
  });

  return sortGroups(groups, sortBy, groupBy);
}

function sortModels(models, sortBy) {
  const sorted = [...models];
  switch (sortBy) {
    case "overall_score_asc":
      sorted.sort((a, b) => (a.overall_score || 0) - (b.overall_score || 0));
      break;
    case "name_asc":
      sorted.sort((a, b) => a.name.localeCompare(b.name));
      break;
    case "organisation_asc":
      sorted.sort((a, b) => a.organisation.localeCompare(b.organisation));
      break;
    case "overall_score_desc":
    default:
      sorted.sort((a, b) => (b.overall_score || 0) - (a.overall_score || 0));
      break;
  }
  return sorted;
}

function sortGroups(groups, sortBy, groupBy = "organisation") {
  const sorted = [...groups];
  switch (sortBy) {
    case "overall_score_asc":
      sorted.sort((a, b) => (a.avgScore || 0) - (b.avgScore || 0));
      break;
    case "name_asc":
      sorted.sort((a, b) =>
        (a.models[0]?.name || "").localeCompare(b.models[0]?.name || "")
      );
      break;
    case "organisation_asc":
      sorted.sort((a, b) => a.key.localeCompare(b.key));
      break;
    case "overall_score_desc":
    default:
      sorted.sort((a, b) => (b.avgScore || 0) - (a.avgScore || 0));
      break;
  }
  return sorted;
}

function renderGroupedTable(groups, groupBy = "organisation") {
  const tbody = document.getElementById("models-table-body");
  tbody.innerHTML = "";

  groups.forEach((g) => {
    const header = document.createElement("tr");
    header.className = "group-row";
    header.dataset.groupKey = g.key;
    header.dataset.groupBy = groupBy;

    const tdName = document.createElement("td");
    tdName.textContent = `${g.count} model${g.count === 1 ? "" : "s"}`;
    tdName.className = "group-row__count";

    const tdOrg = document.createElement("td");
    const tdCountry = document.createElement("td");

    if (groupBy === "country") {
      tdOrg.textContent = "–";
      tdCountry.textContent = g.key;
      tdCountry.className = "group-row__highlight";
    } else {
      tdOrg.textContent = g.key;
      tdOrg.className = "group-row__highlight";
      tdCountry.textContent = g.country || "–";
    }

    const tdScore = document.createElement("td");
    tdScore.className = "table__th--numeric";
    const scoreChip = document.createElement("span");
    scoreChip.className = `score-chip ${scoreClass(g.avgScore)}`;
    scoreChip.textContent = g.avgScore != null ? g.avgScore.toFixed(1) : "–";
    scoreChip.title =
      groupBy === "country"
        ? "Average score across models from this country"
        : "Average score across models from this organisation";
    tdScore.appendChild(scoreChip);

    header.appendChild(tdName);
    header.appendChild(tdOrg);
    header.appendChild(tdCountry);
    header.appendChild(tdScore);

    const memberRows = g.models.map((m) => {
      const tr = document.createElement("tr");
      tr.className = "group-member";
      tr.dataset.groupKey = g.key;

      const tdN = document.createElement("td");
      tdN.textContent = `↳ ${m.name}`;
      tdN.style.paddingLeft = "16px";

      const tdO = document.createElement("td");
      tdO.textContent = m.organisation;

      const tdC = document.createElement("td");
      tdC.textContent = m.country || "–";

      const tdS = document.createElement("td");
      tdS.className = "table__th--numeric";
      const chip = document.createElement("span");
      chip.className = `score-chip ${scoreClass(m.overall_score)}`;
      chip.textContent = formatScore(m.overall_score);
      tdS.appendChild(chip);

      tr.appendChild(tdN);
      tr.appendChild(tdO);
      tr.appendChild(tdC);
      tr.appendChild(tdS);

      tr.style.display = "none";
      tr.addEventListener("click", () => renderDetails(m));
      return tr;
    });

    header.addEventListener("click", () => {
      const expanded = header.dataset.expanded === "true";
      header.dataset.expanded = expanded ? "false" : "true";
      memberRows.forEach((r) => {
        r.style.display = expanded ? "none" : "table-row";
      });
    });

    tbody.appendChild(header);
    memberRows.forEach((r) => tbody.appendChild(r));
  });
}

function renderTable(models, onSelect) {
  const tbody = document.getElementById("models-table-body");
  tbody.innerHTML = "";

  models.forEach((model) => {
    const tr = document.createElement("tr");
    tr.dataset.modelId = model.id;

    const tdName = document.createElement("td");
    tdName.textContent = model.name;

    const tdOrg = document.createElement("td");
    tdOrg.textContent = model.organisation;

    const tdCountry = document.createElement("td");
    tdCountry.textContent = model.country || "–";

    const tdScore = document.createElement("td");
    tdScore.className = "table__th--numeric";
    const scoreChip = document.createElement("span");
    scoreChip.className = `score-chip ${scoreClass(model.overall_score)}`;
    scoreChip.textContent = formatScore(model.overall_score);
    tdScore.appendChild(scoreChip);

    tr.appendChild(tdName);
    tr.appendChild(tdOrg);
    tr.appendChild(tdCountry);
    tr.appendChild(tdScore);

    tr.addEventListener("click", () => {
      onSelect(model);
    });

    tbody.appendChild(tr);
  });
}

function renderPagination(currentPage, pageSize, totalItems, onPageChange) {
  const container = document.getElementById("pagination");
  if (!container) return;

  const totalPages = Math.max(1, Math.ceil(totalItems / pageSize));
  // Clamp current page in case filters reduced the number of pages
  const safeCurrent = Math.min(Math.max(1, currentPage), totalPages);

  container.innerHTML = "";

  const info = document.createElement("div");
  info.className = "pagination__info";
  info.textContent = `Page ${safeCurrent} of ${totalPages}`;

  const prevBtn = document.createElement("button");
  prevBtn.textContent = "Previous";
  prevBtn.disabled = safeCurrent === 1;
  prevBtn.addEventListener("click", () => {
    if (safeCurrent > 1) onPageChange(safeCurrent - 1);
  });

  const nextBtn = document.createElement("button");
  nextBtn.textContent = "Next";
  nextBtn.disabled = safeCurrent === totalPages;
  nextBtn.addEventListener("click", () => {
    if (safeCurrent < totalPages) onPageChange(safeCurrent + 1);
  });

  container.appendChild(info);
  container.appendChild(prevBtn);
  container.appendChild(nextBtn);
}

function renderDetails(model) {
  const placeholder = document.getElementById("details-placeholder");
  const card = document.getElementById("details-card");

  placeholder.style.display = "none";
  card.classList.remove("hidden");

  const nameEl = document.getElementById("details-name");
  const metaEl = document.getElementById("details-meta");
  const scoreEl = document.getElementById("details-score");
  const dimsEl = document.getElementById("details-dimensions");
  const explanationEl = document.getElementById("details-explanation");
  const sourcesEl = document.getElementById("details-sources");

  nameEl.textContent = model.name;
  const orgTypeLabel = model.organisation_type && model.organisation_type !== "–"
    ? ` · ${model.organisation_type}`
    : "";
  metaEl.textContent = `${model.organisation}${orgTypeLabel} · ${model.country || "Unknown country"}`;
  scoreEl.textContent = formatScore(model.overall_score);

  dimsEl.innerHTML = "";

  const dims = model.categories || {};
  const evidenceMap = model.evidence || {};

  Object.entries(dims).forEach(([label, value]) => {
    const row = document.createElement("div");
    row.className = "dimension-row dimension-row--clickable";

    const labelEl = document.createElement("div");
    labelEl.className = "dimension-row__label";
    labelEl.textContent = label;

    const bar = document.createElement("div");
    bar.className = "dimension-row__bar";
    const fill = document.createElement("div");
    fill.className = "dimension-row__bar-fill";
    const pct = Math.max(0, Math.min(1, value || 0)) * 100;
    fill.style.width = `${pct}%`;
    bar.appendChild(fill);

    const valueEl = document.createElement("div");
    valueEl.className = "dimension-row__value";
    valueEl.textContent = value != null ? value.toFixed(2) : "–";

    row.appendChild(labelEl);
    row.appendChild(bar);
    row.appendChild(valueEl);

    row.addEventListener("click", () => {
      showEvidence(label, evidenceMap[label] || []);
    });

    dimsEl.appendChild(row);
  });

  if (explanationEl) {
    explanationEl.textContent = model.explain || "No explanation available for this entry yet.";
  }

  if (sourcesEl) {
    sourcesEl.innerHTML = "";

    const evidence = model.evidence || {};
    const allEvidence = Object.values(evidence).flat();
    const scrapedSources = Array.isArray(model.sources) ? model.sources : [];

    if (!allEvidence.length && !scrapedSources.length) {
      sourcesEl.innerHTML =
        '<div class="details-card__text">No sources available.</div>';
    } else {
      const uniqueEvidence = [...new Map(allEvidence.map((e) => [e.url, e])).values()];
      const evidenceUrls = new Set(uniqueEvidence.map((e) => e.url));
      const remainingSources = scrapedSources.filter((url) => !evidenceUrls.has(url));

      uniqueEvidence.forEach((e) => {
        const item = document.createElement("div");
        item.className = "sources-list__item";

        const a = document.createElement("a");
        a.href = e.url;
        a.target = "_blank";
        a.rel = "noopener noreferrer";
        a.textContent = e.url;

        item.appendChild(a);
        sourcesEl.appendChild(item);
      });

      remainingSources.forEach((url) => {
        const item = document.createElement("div");
        item.className = "sources-list__item";

        const a = document.createElement("a");
        a.href = url;
        a.target = "_blank";
        a.rel = "noopener noreferrer";
        a.textContent = url;

        item.appendChild(a);
        sourcesEl.appendChild(item);
      });
    }
  }
}

function showEvidence(category, evidenceList) {
  const panel = document.getElementById("details-dimension-evidence");
  const title = document.getElementById("evidence-title");
  const content = document.getElementById("evidence-content");

  if (!panel || !title || !content) return;

  title.textContent = category;

  if (!evidenceList || evidenceList.length === 0) {
    content.innerHTML = `<p>No supporting evidence available.</p>`;
    panel.classList.remove("hidden");
    return;
  }

  content.innerHTML = evidenceList.map((e) => {
    const verifiedBadge =
      e.verified === true
        ? '<span class="evidence-verified">Verified quote</span>'
        : e.verified === false
          ? '<span class="evidence-unverified">Unverified quote</span>'
          : "";
    const rationale = e.rationale
      ? `<p class="evidence-rationale">${escapeHtml(e.rationale)}</p>`
      : "";

    return `
    <div class="evidence-item">
      ${verifiedBadge}
      <blockquote class="evidence-quote">
        "${escapeHtml(e.quote || "No quote available")}"
      </blockquote>
      ${rationale}
      <a href="${escapeHtml(e.url)}" target="_blank" rel="noopener noreferrer" class="evidence-link">
        ${escapeHtml(e.url)}
      </a>
    </div>
  `;
  }).join("");

  panel.classList.remove("hidden");
}

function escapeHtml(text) {
  return String(text)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function initDashboard(models) {
  const searchInput = document.getElementById("search-input");
  const sortBySelect = document.getElementById("sort-by");
  const orgTypeSelect = document.getElementById("filter-org-type");
  const groupOrgToggle = document.getElementById("group-by-org-toggle");
  const groupCountryToggle = document.getElementById("group-by-country-toggle");

  const pageSize = 10;
  let currentPage = 1;

  let currentFilters = {
    search: "",
    sortBy: "overall_score_desc",
    orgType: "",
  };

  function getGroupMode() {
    if (groupCountryToggle && groupCountryToggle.checked) return "country";
    if (groupOrgToggle && groupOrgToggle.checked) return "organisation";
    return null;
  }

  function updateView() {
    const filtered = applyFiltersAndSort(models, currentFilters);
    const groupBy = getGroupMode();

    if (groupBy) {
      const groups = computeGroups(filtered, currentFilters.sortBy, groupBy);
      const total = groups.length;
      const totalPages = Math.max(1, Math.ceil(total / pageSize));
      if (currentPage > totalPages) currentPage = totalPages;
      const start = (currentPage - 1) * pageSize;
      const pageItems = groups.slice(start, start + pageSize);
      renderGroupedTable(pageItems, groupBy);
      renderPagination(currentPage, pageSize, total, (nextPage) => {
        currentPage = nextPage;
        updateView();
      });
      return;
    }

    // Normal flat view
    const total = filtered.length;
    const totalPages = Math.max(1, Math.ceil(total / pageSize));
    if (currentPage > totalPages) currentPage = totalPages;
    const start = (currentPage - 1) * pageSize;
    const pageItems = filtered.slice(start, start + pageSize);

    renderTable(pageItems, renderDetails);
    renderPagination(currentPage, pageSize, total, (nextPage) => {
      currentPage = nextPage;
      updateView();
    });
  }

  searchInput.addEventListener("input", (e) => {
    currentFilters.search = e.target.value;
    currentPage = 1;
    updateView();
  });

  sortBySelect.addEventListener("change", (e) => {
    currentFilters.sortBy = e.target.value;
    currentPage = 1;
    updateView();
  });

  if (orgTypeSelect) {
    orgTypeSelect.addEventListener("change", (e) => {
      currentFilters.orgType = e.target.value;
      currentPage = 1;
      updateView();
    });
  }

  function wireGroupToggle(toggle, otherToggle) {
    if (!toggle) return;
    toggle.addEventListener("change", () => {
      if (toggle.checked && otherToggle) {
        otherToggle.checked = false;
      }
      currentPage = 1;
      updateView();
    });
  }

  wireGroupToggle(groupOrgToggle, groupCountryToggle);
  wireGroupToggle(groupCountryToggle, groupOrgToggle);

  renderSummary(models);
  updateView();
}

document.addEventListener("DOMContentLoaded", () => {
  loadModels()
    .then((models) => {
      initDashboard(models);
    })
    .catch((err) => {
      console.error(err);
      const tbody = document.getElementById("models-table-body");
      tbody.innerHTML =
        `<tr><td colspan="4">Failed to load dataset: ${escapeHtml(err.message)}</td></tr>`;
    });
});

