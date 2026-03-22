async function loadModels() {
  const response = await fetch("data/models.json");
  if (!response.ok) {
    throw new Error("Failed to load models.json");
  }
  const raw = await response.json();
  return normalizeModels(raw);
}

function normalizeModels(rawModels) {
  if (!Array.isArray(rawModels)) return [];

  return rawModels
    .map((item) => {
      if (!item || typeof item !== "object") return null;

      // New structure from `generate_data.py`:
      // { model: <display>, score: { model_id, author, value, categories, sources, country, organisation_type }, explain }
      if (item.score && typeof item.score === "object") {
        const score = item.score;
        const modelId = score.model_id || item.model || "";
        const organisation = score.author || (modelId.includes("/") ? modelId.split("/")[0] : "Unknown");
        return {
          id: modelId,
          name: item.model || (modelId ? modelId.split("/").pop() : "Unknown"),
          organisation,
          country: score.country || item.country || "–",
          overall_score: score.value,
          categories: score.categories || {},
          sources: score.sources || [],
          evidence: score.evidence || item.evidence || {},
          explain: item.explain || "",
        };
      }

      // Flat/legacy structure:
      // { model_id, author, value, categories, sources, country, ... }
      if ("model_id" in item || "value" in item) {
        const modelId = item.model_id || item.id || "";
        return {
          id: modelId,
          name: item.name || (modelId ? String(modelId).split("/").pop() : "Unknown"),
          organisation: item.author || item.organisation || "Unknown",
          country: item.country || "–",
          overall_score: item.value ?? item.overall_score,
          categories: item.categories || {},
          sources: item.sources || [],
          evidence: item.evidence || {},
          explain: item.explain || "",
        };
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

function applyFiltersAndSort(models, { search, sortBy }) {
  let filtered = models;

  if (search) {
    const q = search.toLowerCase();
    filtered = filtered.filter((m) => {
      return (
        m.name.toLowerCase().includes(q) ||
        (m.organisation && m.organisation.toLowerCase().includes(q)) ||
        (m.country && m.country.toLowerCase().includes(q))
      );
    });
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
  metaEl.textContent = `${model.organisation} · ${model.country || "Unknown country"}`;
  scoreEl.textContent = formatScore(model.overall_score);

  // dimsEl.innerHTML = "";
  // const dims = model.categories || {};
  // Object.entries(dims).forEach(([label, value]) => {
  //   const row = document.createElement("div");
  //   row.className = "dimension-row";

  //   const labelEl = document.createElement("div");
  //   labelEl.className = "dimension-row__label";
  //   labelEl.textContent = label;

  //   const bar = document.createElement("div");
  //   bar.className = "dimension-row__bar";
  //   const fill = document.createElement("div");
  //   fill.className = "dimension-row__bar-fill";
  //   const pct = Math.max(0, Math.min(1, value || 0)) * 100;
  //   fill.style.width = `${pct}%`;
  //   bar.appendChild(fill);

  //   const valueEl = document.createElement("div");
  //   valueEl.className = "dimension-row__value";
  //   valueEl.textContent = value != null ? value.toFixed(2) : "–";

  //   row.appendChild(labelEl);
  //   row.appendChild(bar);
  //   row.appendChild(valueEl);
  //   dimsEl.appendChild(row);
  // });

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

    // 🔥 CLICK → show evidence
    row.addEventListener("click", () => {
      showEvidence(label, evidenceMap[label] || []);
    });

    dimsEl.appendChild(row);
  });

  if (explanationEl) {
    explanationEl.textContent = model.explain || "No explanation available for this entry yet.";
  }

  // if (sourcesEl) {
  //   sourcesEl.innerHTML = "";
  //   const sources = Array.isArray(model.sources) ? model.sources : [];
  //   if (!sources.length) {
  //     sourcesEl.innerHTML =
  //       '<div class="details-card__text">No scraped sources available for this entry.</div>';
  //   } else {
  //     const maxLinks = 10;
  //     sources.slice(0, maxLinks).forEach((url) => {
  //       const item = document.createElement("div");
  //       item.className = "sources-list__item";
  //       const a = document.createElement("a");
  //       a.href = url;
  //       a.target = "_blank";
  //       a.rel = "noopener noreferrer";
  //       a.textContent = url;
  //       item.appendChild(a);
  //       sourcesEl.appendChild(item);
  //     });
  //     if (sources.length > maxLinks) {
  //       const more = document.createElement("div");
  //       more.className = "details-card__text";
  //       more.textContent = `…and ${sources.length - maxLinks} more.`;
  //       sourcesEl.appendChild(more);
  //     }
  //   }
  // }

  if (sourcesEl) {
    sourcesEl.innerHTML = "";
  
    const evidence = model.evidence || {};
    const allEvidence = Object.values(evidence).flat();
  
    if (!allEvidence.length) {
      sourcesEl.innerHTML =
        '<div class="details-card__text">No sources available.</div>';
    } else {
      const unique = [...new Map(allEvidence.map(e => [e.url, e])).values()];
  
      unique.slice(0, 10).forEach((e) => {
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
  
      if (unique.length > 10) {
        const more = document.createElement("div");
        more.className = "details-card__text";
        more.textContent = `…and ${unique.length - 10} more.`;
        sourcesEl.appendChild(more);
      }
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

  content.innerHTML = evidenceList.map(e => `
    <div class="evidence-item">
      <blockquote class="evidence-quote">
        "${e.quote || "No quote available"}"
      </blockquote>
      <a href="${e.url}" target="_blank" rel="noopener noreferrer" class="evidence-link">
        ${e.url}
      </a>
    </div>
  `).join("");

  panel.classList.remove("hidden");
}

function initDashboard(models) {
  const searchInput = document.getElementById("search-input");
  const sortBySelect = document.getElementById("sort-by");

  const pageSize = 10;
  let currentPage = 1;

  let currentFilters = {
    search: "",
    sortBy: "overall_score_desc",
  };

  function updateView() {
    const filtered = applyFiltersAndSort(models, currentFilters);
    const total = filtered.length;
    const totalPages = Math.max(1, Math.ceil(total / pageSize));
    if (currentPage > totalPages) {
      currentPage = totalPages;
    }
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
        '<tr><td colspan="4">Failed to load dataset. Please check that <code>data/models.json</code> is available.</td></tr>';
    });
});

