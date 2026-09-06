// VITE_API_URL="" (set at docker build) means same-origin: the browser calls
// /api/... on the frontend host and nginx proxies to the backend service.
const RAW = import.meta.env.VITE_API_URL;
const BASE = (RAW === undefined ? "http://localhost:8000" : RAW).replace(/\/$/, "");

async function req(path, opts = {}) {
  const res = await fetch(`${BASE}${path}`, opts);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed (${res.status})`);
  }
  return res.json();
}

export const recommend = (payload) =>
  req("/api/recommend", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });

export const recommendCsv = (file, datasetName, targetColumn) => {
  const form = new FormData();
  form.append("file", file);
  const q = new URLSearchParams();
  if (datasetName) q.set("dataset_name", datasetName);
  if (targetColumn) q.set("target_column", targetColumn);
  const qs = q.toString() ? `?${q}` : "";
  return req(`/api/recommend-csv${qs}`, { method: "POST", body: form });
};

export const fetchHistory = () => req("/api/history");
export const fetchDetail = (id) => req(`/api/history/${id}`);
