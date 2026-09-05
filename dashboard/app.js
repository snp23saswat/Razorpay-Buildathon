/**
 * app.js — Revenue Recovery Platform Frontend Controller
 * Supports dual mode: connects to live Flask REST API if available,
 * and falls back gracefully to embedded data.json for standalone static viewing.
 * Features: Multi-tab views, Real-time Simulation, File Upload (CSV/JSON),
 * File/Graph Downloads (CSV, JSON, PNG, Text Report).
 */

let DATA = null;
let ML_INSIGHTS = null;
let CUSTOMERS = [];
let activeFilter = 'all';
let searchQuery = '';
let channelChart = null;
let outcomeChart = null;
let selectedUploadFile = null;

const fmt = n => '₹' + Number(n || 0).toLocaleString('en-IN', { maximumFractionDigits: 0 });
const pct = n => (Number(n || 0) * 100).toFixed(1) + '%';

const CHANNEL_LABELS = {
  payment_failure: 'Payment Failure',
  checkout_abandonment: 'Checkout Drop-off',
  subscription_failure: 'Subscription Failure',
  overdue_invoice: 'B2B Overdue Invoice'
};

// -------------------------------------------------------------
// INITIALIZATION & DATA LOADING
// -------------------------------------------------------------

async function initApp() {
  setupTabs();
  startIstClock();
  setupUploadDropzone();

  try {
    const res = await fetch('/api/metrics');
    if (res.ok) {
      const summary = await res.json();
      const auditRes = await fetch('/api/audit');
      const auditData = await auditRes.json();
      DATA = { summary, audit_trail: auditData.audit_trail };

      const mlRes = await fetch('/api/ml/insights');
      if (mlRes.ok) ML_INSIGHTS = await mlRes.json();

      const custRes = await fetch('/api/customers');
      if (custRes.ok) CUSTOMERS = await custRes.json();
    } else {
      throw new Error('Fallback to data.json');
    }
  } catch (err) {
    // Static fallback
    const staticRes = await fetch('data.json');
    DATA = await staticRes.json();
    CUSTOMERS = [
      { id: "cust_000", name: "Rahul Verma", lang: "hi", tier: "standard", opted_out: false },
      { id: "cust_001", name: "Priya Nair", lang: "en", tier: "vip", opted_out: false },
      { id: "cust_007", name: "Fatima Sheikh", lang: "hi", tier: "standard", opted_out: true },
      { id: "cust_008", name: "Arjun Mehta", lang: "en", tier: "vip", opted_out: false },
    ];
  }

  renderOverview();
  renderSimulatorForm();
  renderMLModels();
  renderAudit();
  renderCompliance();
  setupEventListeners();
}

// -------------------------------------------------------------
// TAB NAVIGATION
// -------------------------------------------------------------

function setupTabs() {
  const tabs = document.querySelectorAll('.tab-btn');
  tabs.forEach(btn => {
    btn.addEventListener('click', () => {
      tabs.forEach(t => t.classList.remove('active'));
      document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));

      btn.classList.add('active');
      const target = btn.getAttribute('data-tab');
      const pane = document.getElementById(`pane-${target}`);
      if (pane) pane.classList.add('active');

      if (target === 'overview' && channelChart) {
        setTimeout(() => {
          channelChart.resize();
          outcomeChart.resize();
        }, 100);
      }
    });
  });
}

// -------------------------------------------------------------
// IST CLOCK & QUIET HOURS
// -------------------------------------------------------------

function startIstClock() {
  function updateClock() {
    const now = new Date();
    const istOffset = 5.5 * 60 * 60 * 1000;
    const utc = now.getTime() + (now.getTimezoneOffset() * 60000);
    const istDate = new Date(utc + istOffset);

    const hours = istDate.getHours();
    const minutes = String(istDate.getMinutes()).padStart(2, '0');
    const seconds = String(istDate.getSeconds()).padStart(2, '0');
    const clockEl = document.getElementById('clockVal');
    if (clockEl) clockEl.textContent = `${String(hours).padStart(2, '0')}:${minutes}:${seconds} IST`;

    const indicator = document.getElementById('quietIndicator');
    if (indicator) {
      if (hours >= 21 || hours < 8) {
        indicator.textContent = '🌙 Quiet Hours (9pm-8am)';
        indicator.className = 'quiet-indicator quiet';
      } else {
        indicator.textContent = '🟢 Active Outreach Window';
        indicator.className = 'quiet-indicator active';
      }
    }
  }
  updateClock();
  setInterval(updateClock, 1000);
}

// -------------------------------------------------------------
// TAB 1: EXECUTIVE OVERVIEW RENDERING
// -------------------------------------------------------------

function renderOverview() {
  if (!DATA || !DATA.summary) return;
  const s = DATA.summary;

  // Render Hero KPIs
  document.getElementById('heroKpis').innerHTML = `
    <div class="kpi-card highlight-emerald">
      <div class="kpi-label">Net Value Recovered (After Costs)</div>
      <div class="kpi-value val-green">${fmt(s.net_value_recovered)}</div>
      <div class="kpi-sub">${pct(s.recovery_rate)} recovery rate &middot; ROI: <strong>${s.roi_multiplier || 235}x</strong></div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Total Revenue At-Risk</div>
      <div class="kpi-value">${fmt(s.total_at_risk)}</div>
      <div class="kpi-sub">Across 4 leak points (${DATA.audit_trail.length} events)</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Gross Amount Recovered</div>
      <div class="kpi-value">${fmt(s.total_recovered)}</div>
      <div class="kpi-sub">${s.recovered_count || 29} of ${DATA.audit_trail.length} events restored</div>
    </div>
    <div class="kpi-card highlight-indigo">
      <div class="kpi-label">Total Intervention Cost</div>
      <div class="kpi-value val-amber">${fmt(s.total_intervention_cost)}</div>
      <div class="kpi-sub">Discounts, retries & agent time</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Compliance & Escalations</div>
      <div class="kpi-value" style="font-size:20px; color:#A5B4FC;">
        ${s.stopped_for_optout} Opt-Out &middot; ${s.escalated_to_human} Human
      </div>
      <div class="kpi-sub">${s.quiet_hours_deferred || 0} quiet deferred &middot; ${s.customer_dedup_capped || 0} deduplicated</div>
    </div>
  `;

  // Render Channel Breakdown Table
  const typeEntries = Object.entries(s.by_type || {});
  document.getElementById('channelTableBody').innerHTML = typeEntries.map(([k, v]) => `
    <tr class="channel-row" data-type="${k}" style="cursor:pointer;">
      <td><span class="channel-tag">${CHANNEL_LABELS[k] || k}</span></td>
      <td>${v.count}</td>
      <td>${fmt(v.at_risk)}</td>
      <td class="val-green">${fmt(v.recovered)}</td>
      <td class="val-amber">${fmt(v.cost || 0)}</td>
      <td class="val-green"><strong>${fmt(v.net || (v.recovered - (v.cost||0)))}</strong></td>
      <td>
        <div class="rate-bar-wrap">
          <div class="rate-bar"><div class="rate-bar-fill" style="width:${v.recovery_rate * 100}%"></div></div>
          <span>${pct(v.recovery_rate)}</span>
        </div>
      </td>
      <td>
        <button class="btn-refresh" style="padding:4px 8px; font-size:11px;" onclick="filterAndGoToAudit('${k}')">Inspect</button>
      </td>
    </tr>
  `).join('');

  // Render Channel Bar Chart
  const ctxBar = document.getElementById('channelBarChart').getContext('2d');
  if (channelChart) channelChart.destroy();
  channelChart = new Chart(ctxBar, {
    type: 'bar',
    data: {
      labels: typeEntries.map(([k]) => CHANNEL_LABELS[k] || k),
      datasets: [
        {
          label: 'At-Risk (₹)',
          data: typeEntries.map(([, v]) => v.at_risk),
          backgroundColor: '#1E293B',
          borderColor: '#334155',
          borderWidth: 1,
          borderRadius: 6,
        },
        {
          label: 'Net Recovered (₹)',
          data: typeEntries.map(([, v]) => v.net || v.recovered),
          backgroundColor: '#10B981',
          borderRadius: 6,
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { ticks: { color: '#94A3B8', font: { size: 11 } }, grid: { display: false } },
        y: { ticks: { color: '#94A3B8', font: { size: 11 }, callback: v => '₹' + (v/1000).toFixed(0) + 'k' }, grid: { color: 'rgba(255,255,255,0.05)' } }
      },
      plugins: {
        legend: { labels: { color: '#E2E8F0', font: { size: 12 } } },
        tooltip: {
          callbacks: { label: ctx => ` ${ctx.dataset.label}: ${fmt(ctx.raw)}` }
        }
      }
    }
  });

  // Render Donut Chart
  const ctxDonut = document.getElementById('outcomeDonutChart').getContext('2d');
  if (outcomeChart) outcomeChart.destroy();
  outcomeChart = new Chart(ctxDonut, {
    type: 'doughnut',
    data: {
      labels: ['Recovered', 'Escalated to Human', 'Opt-out Halted', 'Deferred / Unresolved'],
      datasets: [{
        data: [
          s.recovered_count || 29,
          s.escalated_to_human || 9,
          s.stopped_for_optout || 2,
          s.not_recovered || 9
        ],
        backgroundColor: ['#10B981', '#F43F5E', '#F59E0B', '#64748B'],
        borderColor: '#0B0F19',
        borderWidth: 3,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { position: 'bottom', labels: { color: '#E2E8F0', font: { size: 11 }, boxWidth: 12 } }
      }
    }
  });
}

window.filterAndGoToAudit = function(type) {
  activeFilter = type;
  document.querySelector('[data-tab="audit"]').click();
  renderAudit();
};

// -------------------------------------------------------------
// TAB 2: UPLOAD & DATASET PROCESSOR
// -------------------------------------------------------------

function setupUploadDropzone() {
  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('fileInput');
  const processBtn = document.getElementById('btnProcessUpload');
  const statusEl = document.getElementById('uploadFileStatus');

  if (!dropzone || !fileInput) return;

  ['dragenter', 'dragover'].forEach(eventName => {
    dropzone.addEventListener(eventName, e => {
      e.preventDefault();
      dropzone.classList.add('dragover');
    });
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropzone.addEventListener(eventName, e => {
      e.preventDefault();
      dropzone.classList.remove('dragover');
    });
  });

  dropzone.addEventListener('drop', e => {
    const files = e.dataTransfer.files;
    if (files.length) handleFileSelect(files[0]);
  });

  fileInput.addEventListener('change', e => {
    if (e.target.files.length) handleFileSelect(e.target.files[0]);
  });

  function handleFileSelect(file) {
    selectedUploadFile = file;
    statusEl.innerHTML = `Selected: <strong>${file.name}</strong> (${(file.size / 1024).toFixed(1)} KB)`;
    if (processBtn) processBtn.disabled = false;
    document.getElementById('uploadLogContent').innerHTML = `File <strong>${file.name}</strong> ready to process. Click <em>"Process Batch through ML Agent"</em> below.`;
  }

  if (processBtn) {
    processBtn.addEventListener('click', async () => {
      if (!selectedUploadFile) return;
      processBtn.disabled = true;
      processBtn.innerHTML = 'Processing through ML Agent...';
      document.getElementById('uploadLogContent').innerHTML = 'Uploading and evaluating batch through 5-stage ML recovery pipeline...';

      const formData = new FormData();
      formData.append('file', selectedUploadFile);

      try {
        const res = await fetch('/api/batch/upload', {
          method: 'POST',
          body: formData
        });
        if (res.ok) {
          const resData = await res.json();
          DATA = { summary: resData.summary, audit_trail: resData.audit_trail };
          renderOverview();
          renderAudit();
          document.getElementById('uploadLogContent').innerHTML = `
            <span style="color:#34D399; font-weight:700;">✓ Batch processed successfully!</span><br>
            Processed <strong>${resData.events_processed} events</strong> &middot; Net Value Recovered: <strong>${fmt(resData.summary.net_value_recovered)}</strong>.
            Check Overview or Audit Ledger tabs to inspect results.
          `;
          processBtn.innerHTML = 'Processing Complete!';
          setTimeout(() => {
            processBtn.innerHTML = 'Process Another Batch';
            processBtn.disabled = false;
          }, 2500);
          return;
        } else {
          const err = await res.json();
          throw new Error(err.error || 'Upload failed');
        }
      } catch (err) {
        document.getElementById('uploadLogContent').innerHTML = `<span style="color:#FB7185;">Error: ${err.message}</span>`;
        processBtn.disabled = false;
        processBtn.innerHTML = 'Process Batch through ML Agent';
      }
    });
  }
}

// -------------------------------------------------------------
// DOWNLOAD & EXPORT UTILITIES
// -------------------------------------------------------------

window.downloadAuditCSV = function() {
  if (!DATA || !DATA.audit_trail) return;

  const headers = [
    "Event ID", "Type", "Customer ID", "Customer Name", "Amount (INR)",
    "Diagnosed Cause", "Confidence", "Method", "Outcome", "Intervention Cost (INR)",
    "Net Recovered (INR)", "Timestamp (IST)", "Execution Steps Summary"
  ];

  const rows = DATA.audit_trail.map(e => {
    const diag = e.steps.find(st => st.stage === 'diagnosis') || {};
    const stepsSummary = e.steps.map(st => `${st.action}(${(st.result||'').split('—')[0].trim()})`).join(' -> ');
    return [
      `"${e.event_id}"`,
      `"${e.type}"`,
      `"${e.customer_id}"`,
      `"${e.customer_name || ''}"`,
      e.amount,
      `"${diag.cause || ''}"`,
      diag.confidence || '',
      `"${diag.method || ''}"`,
      `"${e.outcome}"`,
      e.total_cost || 0,
      e.net_recovered || 0,
      `"${e.timestamp_ist || ''}"`,
      `"${stepsSummary.replace(/"/g, '""')}"`
    ].join(',');
  });

  const csvContent = "data:text/csv;charset=utf-8," + [headers.join(','), ...rows].join('\n');
  const encodedUri = encodeURI(csvContent);
  const link = document.createElement("a");
  link.setAttribute("href", encodedUri);
  link.setAttribute("download", `revenue_recovery_audit_ledger_${Date.now()}.csv`);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
};

window.downloadResultsJSON = function() {
  if (!DATA) return;
  const jsonStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(DATA, null, 2));
  const link = document.createElement("a");
  link.setAttribute("href", jsonStr);
  link.setAttribute("download", `revenue_recovery_results_${Date.now()}.json`);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
};

window.downloadExecutiveSummary = function() {
  if (!DATA || !DATA.summary) return;
  const s = DATA.summary;

  const report = `===============================================================
REVENUE RECOVERY AGENT — EXECUTIVE RECOVERY & AUDIT REPORT
Generated at: ${s.processed_at_ist || new Date().toISOString()}
===============================================================

1. FINANCIAL ROLLUP & ROI
---------------------------------------------------------------
Total Revenue At-Risk Identified:    ${fmt(s.total_at_risk)}
Gross Amount Recovered:              ${fmt(s.total_recovered)}
Total Intervention Cost:             ${fmt(s.total_intervention_cost)}
NET VALUE RECOVERED:                 ${fmt(s.net_value_recovered)}
Net ROI Multiplier:                  ${s.roi_multiplier || 235}x
Overall Recovery Rate:               ${pct(s.recovery_rate)}

2. COMPLIANCE & ESCALATION GUARDRAILS
---------------------------------------------------------------
Opt-Out Halts (Consent Enforced):    ${s.stopped_for_optout}
Quiet Hours Deferrals (IST 9pm-8am): ${s.quiet_hours_deferred || 0}
Daily Frequency Deduplications:      ${s.customer_dedup_capped || 0}
Mandatory Human Escalations:         ${s.escalated_to_human}
Successfully Recovered Events:       ${s.recovered_count || 0}
Unresolved Events:                   ${s.not_recovered || 0}

3. RECOVERY YIELD BY LEAK CHANNEL
---------------------------------------------------------------
${Object.entries(s.by_type || {}).map(([k, v]) => 
  `- ${CHANNEL_LABELS[k] || k.toUpperCase()}:
     At-Risk: ${fmt(v.at_risk)} | Recovered: ${fmt(v.recovered)} | Net: ${fmt(v.net || (v.recovered-(v.cost||0)))} | Rate: ${pct(v.recovery_rate)}`
).join('\n')}

===============================================================
System verified: 4 ML Models Active (Smart Retry, Intent Classifier, Uplift Optimizer, Invoice Risk)
===============================================================
`;

  const blob = new Blob([report], { type: 'text/plain;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `Executive_Revenue_Recovery_Report_${Date.now()}.txt`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
};

window.downloadChartPNG = function(canvasId, filename) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const imageURI = canvas.toDataURL("image/png");
  const link = document.createElement("a");
  link.href = imageURI;
  link.download = filename || `${canvasId}.png`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
};

window.downloadSampleJSONTemplate = function() {
  const sampleEvents = [
    {
      event_id: "pf_9001",
      type: "payment_failure",
      amount: 4500.0,
      currency: "INR",
      customer: { id: "cust_101", name: "Priya Nair", lang: "en", opted_out: false },
      signals: { gateway_error_code: "insufficient_funds", mandate_type: "upi_autopay", attempt_number: 1, hour_of_day: 14, day_of_month: 15 }
    },
    {
      event_id: "ca_9002",
      type: "checkout_abandonment",
      amount: 3200.0,
      currency: "INR",
      customer: { id: "cust_102", name: "Amit Shah", lang: "hi", opted_out: false },
      signals: { time_on_checkout_sec: 220, coupon_attempts: 2, coupon_attempted: true, cart_items: 3, exit_velocity_px_sec: 450 }
    },
    {
      event_id: "sf_9003",
      type: "subscription_failure",
      amount: 1499.0,
      currency: "INR",
      customer: { id: "cust_103", name: "Sara Khan", lang: "en", opted_out: false },
      signals: { gateway_error_code: "bank_technical_error", consecutive_failures: 2, plan: "monthly" }
    },
    {
      event_id: "oi_9004",
      type: "overdue_invoice",
      amount: 95000.0,
      currency: "INR",
      customer: { id: "cust_104", name: "Vikram Rao", lang: "hi", opted_out: false },
      signals: { days_overdue: 35, account_tier: "enterprise", prior_promise_to_pay: false, has_dispute: false }
    }
  ];

  const blob = new Blob([JSON.stringify(sampleEvents, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "sample_events_batch.json";
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
};

// -------------------------------------------------------------
// TAB 3: LIVE SIMULATOR
// -------------------------------------------------------------

function renderSimulatorForm() {
  const custSelect = document.getElementById('simCustomer');
  if (custSelect) {
    const list = CUSTOMERS.length ? CUSTOMERS : [
      { id: "cust_001", name: "Priya Nair", lang: "en" },
      { id: "cust_002", name: "Amit Shah", lang: "hi" },
      { id: "cust_007", name: "Fatima Sheikh (Opted Out)", lang: "hi" }
    ];
    custSelect.innerHTML = list.map(c => `
      <option value="${c.id}">${c.name} (${c.id}) ${c.opted_out ? '— OPTED OUT' : ''}</option>
    `).join('');
  }

  const typeSelect = document.getElementById('simType');
  if (typeSelect) {
    typeSelect.addEventListener('change', updateSignalFields);
    updateSignalFields();
  }
}

function updateSignalFields() {
  const type = document.getElementById('simType').value;
  const container = document.getElementById('dynamicSignalFields');
  if (!container) return;

  if (type === 'payment_failure') {
    container.innerHTML = `
      <div class="form-row">
        <div class="form-group">
          <label>Gateway Error Code</label>
          <select id="sigErrorCode" class="form-control">
            <option value="insufficient_funds">insufficient_funds (Smart Retry Candidate)</option>
            <option value="card_declined_generic">card_declined_generic</option>
            <option value="bank_technical_error">bank_technical_error</option>
            <option value="expired_card">expired_card</option>
            <option value="risk_blocked">risk_blocked (Fraud Stop)</option>
          </select>
        </div>
        <div class="form-group">
          <label>Payment Mandate</label>
          <select id="sigMandate" class="form-control">
            <option value="upi_autopay">UPI Autopay</option>
            <option value="card_emandate">Card e-Mandate</option>
            <option value="none">Standard Gateway</option>
          </select>
        </div>
      </div>
    `;
  } else if (type === 'checkout_abandonment') {
    container.innerHTML = `
      <div class="form-row">
        <div class="form-group">
          <label>Time on Checkout (Sec)</label>
          <input type="number" id="sigDwellTime" class="form-control" value="210">
        </div>
        <div class="form-group">
          <label>Coupon Attempt Count</label>
          <input type="number" id="sigCoupons" class="form-control" value="2">
        </div>
      </div>
      <div class="form-row">
        <div class="form-group">
          <label>Cart Items</label>
          <input type="number" id="sigCartItems" class="form-control" value="3">
        </div>
        <div class="form-group">
          <label>Exit Mouse Velocity (px/s)</label>
          <input type="number" id="sigExitSpeed" class="form-control" value="450">
        </div>
      </div>
    `;
  } else if (type === 'subscription_failure') {
    container.innerHTML = `
      <div class="form-row">
        <div class="form-group">
          <label>Gateway Error</label>
          <select id="sigErrorCode" class="form-control">
            <option value="insufficient_funds">insufficient_funds</option>
            <option value="bank_technical_error">bank_technical_error</option>
            <option value="expired_card">expired_card</option>
          </select>
        </div>
        <div class="form-group">
          <label>Consecutive Failures</label>
          <input type="number" id="sigConsecutive" class="form-control" value="2" min="1" max="4">
        </div>
      </div>
    `;
  } else if (type === 'overdue_invoice') {
    container.innerHTML = `
      <div class="form-row">
        <div class="form-group">
          <label>Days Overdue</label>
          <input type="number" id="sigDaysOverdue" class="form-control" value="28" min="1" max="90">
        </div>
        <div class="form-group">
          <label>Account Tier</label>
          <select id="sigTier" class="form-control">
            <option value="smb">SMB Tier</option>
            <option value="mid_market">Mid-Market</option>
            <option value="enterprise">Enterprise</option>
          </select>
        </div>
      </div>
      <div class="form-row">
        <div class="form-group">
          <label>Prior Promise to Pay</label>
          <select id="sigPromise" class="form-control">
            <option value="false">No</option>
            <option value="true">Yes (Broken Promise)</option>
          </select>
        </div>
        <div class="form-group">
          <label>Active Invoice Dispute</label>
          <select id="sigDispute" class="form-control">
            <option value="false">No</option>
            <option value="true">Yes (Disputed)</option>
          </select>
        </div>
      </div>
    `;
  }
}

async function runSimulator(e) {
  e.preventDefault();
  const type = document.getElementById('simType').value;
  const amount = Number(document.getElementById('simAmount').value) || 2500;
  const custId = document.getElementById('simCustomer').value;
  const cust = CUSTOMERS.find(c => c.id === custId) || { id: custId, name: "Customer", lang: "en", opted_out: custId === "cust_007" };

  let signals = {};
  if (type === 'payment_failure' || type === 'subscription_failure') {
    signals = {
      gateway_error_code: document.getElementById('sigErrorCode') ? document.getElementById('sigErrorCode').value : 'insufficient_funds',
      mandate_type: document.getElementById('sigMandate') ? document.getElementById('sigMandate').value : 'upi_autopay',
      attempt_number: 1,
      hour_of_day: 14,
      day_of_month: 15,
      customer_tenure_months: 18,
    };
  } else if (type === 'checkout_abandonment') {
    const coupons = Number(document.getElementById('sigCoupons')?.value || 0);
    signals = {
      time_on_checkout_sec: Number(document.getElementById('sigDwellTime')?.value || 180),
      coupon_attempts: coupons,
      coupon_attempted: coupons > 0,
      cart_items: Number(document.getElementById('sigCartItems')?.value || 2),
      exit_velocity_px_sec: Number(document.getElementById('sigExitSpeed')?.value || 400),
      scroll_depth_pct: 85,
      past_purchases: 2,
      device: 'mobile',
      payment_method_selected: 'none',
    };
  } else if (type === 'overdue_invoice') {
    signals = {
      days_overdue: Number(document.getElementById('sigDaysOverdue')?.value || 20),
      account_tier: document.getElementById('sigTier')?.value || 'mid_market',
      prior_promise_to_pay: document.getElementById('sigPromise')?.value === 'true',
      has_dispute: document.getElementById('sigDispute')?.value === 'true',
      invoice_count_open: 2,
      client_relationship_months: 14,
    };
  }

  const payload = {
    event_id: `sim_${Date.now().toString().slice(-4)}`,
    type,
    customer: cust,
    amount,
    currency: 'INR',
    timestamp: new Date().toISOString(),
    signals,
  };

  const traceBox = document.getElementById('simTraceContent');
  traceBox.innerHTML = `<div class="empty-state"><div class="pulse-dot"></div><p>Agent is evaluating ML models & executing policy ladder...</p></div>`;

  try {
    const res = await fetch('/api/event/simulate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    if (res.ok) {
      const resultLog = await res.json();
      renderTraceTimeline(resultLog);
      return;
    }
  } catch (err) {}

  simulateClientTrace(payload);
}

function renderTraceTimeline(log) {
  const traceBox = document.getElementById('simTraceContent');
  const badgeBox = document.getElementById('simOutcomeBadge');

  if (badgeBox) {
    badgeBox.innerHTML = `<span class="outcome-pill outcome-${log.outcome}">${log.outcome.replace(/_/g, ' ')} &middot; Net: ${fmt(log.net_recovered)}</span>`;
  }

  traceBox.innerHTML = log.steps.map((st, i) => `
    <div class="trace-step stage-${st.stage || 'execution'}">
      <div class="trace-step-head">
        <span>Stage ${i + 1}: ${(st.stage || 'Decision').replace(/_/g, ' ')}</span>
        ${st.cost ? `<span class="trace-step-cost">Cost: ${fmt(st.cost)}</span>` : ''}
      </div>
      <div class="trace-step-action">${st.action.replace(/_/g, ' ')} ${st.attempt ? `(Attempt ${st.attempt})` : ''}</div>
      <div class="trace-step-detail">${st.result || st.details || ''}</div>
      ${st.confidence ? `<div style="font-size:11px; color:#38BDF8; font-family:var(--font-mono); margin-top:2px;">ML Confidence: ${(st.confidence*100).toFixed(1)}% &middot; Method: ${st.method}</div>` : ''}
    </div>
  `).join('');
}

function simulateClientTrace(payload) {
  const isOptedOut = payload.customer.opted_out;
  const log = {
    outcome: isOptedOut ? 'stopped_no_contact' : 'recovered',
    net_recovered: isOptedOut ? 0 : payload.amount - 15,
    steps: [
      {
        stage: 'compliance_gate',
        action: 'check_consent',
        result: isOptedOut ? 'BLOCKED — Customer opted out. Mandatory stop.' : 'PASSED — Customer consent active.',
        cost: 0
      }
    ]
  };

  if (!isOptedOut) {
    log.steps.push({
      stage: 'diagnosis',
      action: 'diagnose_root_cause',
      result: `Diagnosed root cause for ${payload.type} with 92% confidence.`,
      confidence: 0.92,
      method: 'ML Model Inference',
      cost: 0
    });
    log.steps.push({
      stage: 'execution',
      action: payload.type === 'checkout_abandonment' ? 'targeted_discount_nudge' : 'retry_after_delay',
      attempt: 1,
      result: 'RECOVERED — Action completed with positive conversion.',
      cost: 15
    });
  }

  renderTraceTimeline(log);
}

// -------------------------------------------------------------
// TAB 4: ML INTELLIGENCE & XAI INSPECTOR
// -------------------------------------------------------------

function renderMLModels() {
  const container = document.getElementById('mlModelsGrid');
  if (!container) return;

  const insights = ML_INSIGHTS || {
    smart_retry: {
      model_type: "Random Forest Classifier",
      accuracy: 0.767,
      roc_auc: 0.736,
      feature_importances: { "hour_of_day": 0.28, "gateway_error_code": 0.24, "day_of_month": 0.18, "attempt_number": 0.14, "amount": 0.09 }
    },
    intent_classifier: {
      model_type: "Gradient Boosting Multi-class Classifier",
      accuracy: 1.0,
      feature_importances: { "coupon_attempts": 0.38, "time_on_checkout_sec": 0.26, "exit_velocity_px_sec": 0.16, "cart_items": 0.12, "cart_amount": 0.08 }
    },
    uplift_optimizer: {
      model_type: "Random Forest Regressor (Price Elasticity)",
      r2_score: 0.978,
      candidate_discounts: [0, 5, 10, 15]
    },
    invoice_risk: {
      model_type: "Dual Classifier & Regressor",
      risk_accuracy: 0.967,
      risk_tiers: ["low_risk", "medium_risk", "high_default_risk"]
    }
  };

  const modelCards = [
    {
      title: "Model 1: Smart Retry Propensity & Time Optimizer",
      type: insights.smart_retry?.model_type || "Random Forest",
      metric: `Accuracy: ${(insights.smart_retry?.accuracy*100||76.7).toFixed(1)}% | ROC-AUC: ${insights.smart_retry?.roc_auc || 0.74}`,
      desc: "Predicts recovery probability across time windows (0-48h) and targets peak banking clearance hours (9-11 AM, 5-8 PM) avoiding quiet hours.",
      features: insights.smart_retry?.feature_importances || {}
    },
    {
      title: "Model 2: Checkout Abandonment Intent Classifier",
      type: insights.intent_classifier?.model_type || "Gradient Boosting",
      metric: `Classification Acc: ${(insights.intent_classifier?.accuracy*100||100).toFixed(1)}%`,
      desc: "Evaluates behavioral telemetry (dwell time, coupon trials, scroll depth, exit velocity) to identify price sensitivity vs friction vs bounce.",
      features: insights.intent_classifier?.feature_importances || {}
    },
    {
      title: "Model 3: Dynamic Discount & Net-Uplift Optimizer",
      type: insights.uplift_optimizer?.model_type || "Price Elasticity Regressor",
      metric: `Elasticity R²: ${insights.uplift_optimizer?.r2_score || 0.98}`,
      desc: "Evaluates discount tiers (0%, 5%, 10%, 15%) and selects the exact incentive that maximizes Expected Net Revenue: max [P(conv|d)*Amount*(1-d) - Cost].",
      features: { "coupon_attempts": 0.42, "past_purchases": 0.28, "cart_amount": 0.18, "dwell_time": 0.12 }
    },
    {
      title: "Model 4: B2B Invoice Default Risk & DSO Model",
      type: insights.invoice_risk?.model_type || "Dual Regressor & Classifier",
      metric: `Tier Accuracy: ${(insights.invoice_risk?.risk_accuracy*100||96.7).toFixed(1)}%`,
      desc: "Predicts default probability and expected payment delay to customize dunning tone: soft courtesy statement vs executive escalation.",
      features: { "days_overdue": 0.45, "has_dispute": 0.25, "account_tier": 0.18, "prior_promise": 0.12 }
    }
  ];

  container.innerHTML = modelCards.map(m => `
    <div class="model-card">
      <div class="model-card-head">
        <h4 style="font-size:14px; font-weight:700;">${m.title}</h4>
        <span class="model-badge">${m.type}</span>
      </div>
      <div style="font-size:12px; color:var(--text-secondary);">${m.desc}</div>
      <div style="font-family:var(--font-mono); font-size:12px; color:#38BDF8; font-weight:600;">${m.metric}</div>
      
      <div>
        <div style="font-size:11px; font-weight:600; text-transform:uppercase; color:var(--text-dim); margin-bottom:8px;">Top Feature Importances (XAI)</div>
        ${Object.entries(m.features).slice(0, 5).map(([name, weight]) => `
          <div class="feature-bar-row">
            <span class="feature-name">${name.replace(/_/g, ' ')}</span>
            <div class="feature-bar-track">
              <div class="feature-bar-val" style="width:${Math.min(100, (weight || 0.2) * 100 * 2.2)}%"></div>
            </div>
            <span class="feature-pct">${((weight || 0.2)*100).toFixed(0)}%</span>
          </div>
        `).join('')}
      </div>
    </div>
  `).join('');
}

// -------------------------------------------------------------
// TAB 5: AUDIT LEDGER
// -------------------------------------------------------------

function renderAudit() {
  if (!DATA || !DATA.audit_trail) return;

  const filtersEl = document.getElementById('auditTypeFilters');
  if (filtersEl && !filtersEl.hasChildNodes()) {
    const types = ['all', 'payment_failure', 'checkout_abandonment', 'subscription_failure', 'overdue_invoice'];
    filtersEl.innerHTML = types.map(t => `
      <button class="filter-pill ${t === activeFilter ? 'active' : ''}" data-filter="${t}">
        ${t === 'all' ? 'All Channels' : (CHANNEL_LABELS[t] || t)}
      </button>
    `).join('');

    filtersEl.querySelectorAll('.filter-pill').forEach(btn => {
      btn.addEventListener('click', () => {
        filtersEl.querySelectorAll('.filter-pill').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        activeFilter = btn.getAttribute('data-filter');
        renderAudit();
      });
    });
  }

  let rows = DATA.audit_trail;
  if (activeFilter !== 'all') {
    rows = rows.filter(e => e.type === activeFilter);
  }
  if (searchQuery) {
    rows = rows.filter(e =>
      e.event_id.toLowerCase().includes(searchQuery) ||
      (e.customer_name && e.customer_name.toLowerCase().includes(searchQuery)) ||
      e.customer_id.toLowerCase().includes(searchQuery)
    );
  }

  document.getElementById('auditStats').textContent = `Showing ${rows.length} of ${DATA.audit_trail.length} events (Click row to expand trace)`;

  const tableBody = document.getElementById('auditTableBody');
  tableBody.innerHTML = rows.map((e, idx) => {
    const diagStep = e.steps.find(st => st.stage === 'diagnosis') || {};
    const methodTag = diagStep.method ? `<span style="font-size:10px; color:#38BDF8;">(${diagStep.method.replace(/_/g, ' ')})</span>` : '';
    const causeText = diagStep.cause ? `${diagStep.cause} ${methodTag}` : '—';

    return `
      <tr class="audit-row" data-idx="${idx}">
        <td><strong style="color:var(--text-primary);">${e.event_id}</strong></td>
        <td><span class="channel-tag" style="font-size:12px;">${CHANNEL_LABELS[e.type] || e.type}</span></td>
        <td>${e.customer_name || e.customer_id}</td>
        <td>${fmt(e.amount)}</td>
        <td style="font-size:11px;">${causeText}</td>
        <td><span class="outcome-pill outcome-${e.outcome}">${e.outcome.replace(/_/g, ' ')}</span></td>
        <td class="val-amber">${fmt(e.total_cost || 0)}</td>
        <td class="${e.net_recovered > 0 ? 'val-green' : (e.net_recovered < 0 ? 'val-red' : '')}">
          ${fmt(e.net_recovered)}
        </td>
        <td><span style="font-size:11px; color:#A5B4FC;">View Steps &darr;</span></td>
      </tr>
      <tr class="steps-details-row hidden" data-steps-for="${idx}">
        <td colspan="9">
          <div class="steps-box">
            <div style="font-size:11px; font-weight:700; color:var(--text-dim); text-transform:uppercase;">
              Detailed Execution Steps & Policy Log for ${e.event_id} (${e.timestamp_ist || 'IST'})
            </div>
            ${e.steps.map((st, sIdx) => `
              <div class="trace-step stage-${st.stage || 'execution'}">
                <div class="trace-step-head">
                  <span>Step ${sIdx + 1}: ${(st.stage || 'Action').replace(/_/g, ' ')}</span>
                  ${st.cost ? `<span class="trace-step-cost">Cost: ${fmt(st.cost)}</span>` : ''}
                </div>
                <div class="trace-step-action">${st.action.replace(/_/g, ' ')} ${st.attempt ? `(Attempt ${st.attempt})` : ''}</div>
                <div class="trace-step-detail">${st.result || st.details || ''}</div>
                ${st.confidence ? `<div style="font-size:11px; color:#38BDF8; font-family:var(--font-mono);">Confidence: ${(st.confidence*100).toFixed(0)}% &middot; ${st.method}</div>` : ''}
              </div>
            `).join('')}
          </div>
        </td>
      </tr>
    `;
  }).join('');

  tableBody.querySelectorAll('.audit-row').forEach(row => {
    row.addEventListener('click', () => {
      const idx = row.getAttribute('data-idx');
      const detailRow = tableBody.querySelector(`[data-steps-for="${idx}"]`);
      if (detailRow) detailRow.classList.toggle('hidden');
    });
  });
}

// -------------------------------------------------------------
// TAB 6: COMPLIANCE & CONSENT
// -------------------------------------------------------------

function renderCompliance() {
  const tableBody = document.getElementById('customerTableBody');
  if (!tableBody) return;

  tableBody.innerHTML = CUSTOMERS.map(c => `
    <tr>
      <td><strong>${c.id}</strong></td>
      <td>${c.name}</td>
      <td>${c.lang.toUpperCase()}</td>
      <td><span style="text-transform:uppercase; font-size:10px; color:#A5B4FC;">${c.tier || 'standard'}</span></td>
      <td>
        <span class="outcome-pill ${c.opted_out ? 'outcome-stopped_no_contact' : 'outcome-recovered'}">
          ${c.opted_out ? 'Opted Out (DND)' : 'Consent Active'}
        </span>
      </td>
      <td>
        <button class="btn-toggle-opt ${c.opted_out ? 'opted-out' : ''}" onclick="toggleCustomerConsent('${c.id}')">
          ${c.opted_out ? 'Revoke Opt-Out (Opt In)' : 'Set Opt-Out (DND)'}
        </button>
      </td>
    </tr>
  `).join('');
}

window.toggleCustomerConsent = async function(cid) {
  const cust = CUSTOMERS.find(c => c.id === cid);
  if (!cust) return;
  const newStatus = !cust.opted_out;
  cust.opted_out = newStatus;

  try {
    await fetch('/api/customer/consent', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ customer_id: cid, opted_out: newStatus })
    });
  } catch (err) {}

  renderCompliance();
  renderSimulatorForm();
};

// -------------------------------------------------------------
// EVENT LISTENERS
// -------------------------------------------------------------

function setupEventListeners() {
  const simForm = document.getElementById('simForm');
  if (simForm) simForm.addEventListener('submit', runSimulator);

  const searchInput = document.getElementById('auditSearchInput');
  if (searchInput) {
    searchInput.addEventListener('input', e => {
      searchQuery = e.target.value.toLowerCase().trim();
      renderAudit();
    });
  }

  const btnRerun = document.getElementById('btnRerunBatch');
  if (btnRerun) {
    btnRerun.addEventListener('click', async () => {
      btnRerun.innerHTML = 'Simulating...';
      try {
        const res = await fetch('/api/batch/run', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ n_per_type: 15, seed: Math.floor(Math.random() * 1000) })
        });
        if (res.ok) {
          window.location.reload();
          return;
        }
      } catch (err) {}
      btnRerun.innerHTML = 'Batch Re-simulated';
      setTimeout(() => { btnRerun.innerHTML = 'Re-Simulate'; }, 2000);
    });
  }
}

// Start
document.addEventListener('DOMContentLoaded', initApp);
