import { useCallback, useEffect, useState } from "react";
import { fetchDetail, fetchHistory, recommend, recommendCsv } from "./api.js";

const SAMPLE_CSV = `size,rooms,age,style
1500,3,12,apt
,4,8,detached
1800,,15,apt
1600,3,,semi
2100,5,2,detached
1400,2,30,apt`;

const LABELS = { simple: "Simple (median)", knn: "KNN (k=5)", mice: "MICE" };

function parseCsv(text) {
  const lines = text.split(/\r?\n/).map((l) => l.trim()).filter(Boolean);
  if (lines.length < 2) throw new Error("Need a header row plus at least one data row.");
  const columns = lines[0].split(",").map((s) => s.trim());
  const rows = lines.slice(1).map((line) =>
    line.split(",").map((cell) => {
      const v = cell.trim();
      if (v === "" || v.toLowerCase() === "null" || v.toLowerCase() === "nan") return null;
      const n = Number(v);
      return v !== "" && !Number.isNaN(n) ? n : v;
    })
  );
  return { columns, rows };
}

function ScoreBar({ name, value, highlight }) {
  return (
    <div className={`score ${highlight ? "winner" : ""}`}>
      <div className="score-head">
        <span>{LABELS[name] || name}</span>
        <span className="score-val">{value}</span>
      </div>
      <div className="bar"><div className="fill" style={{ width: `${value}%` }} /></div>
    </div>
  );
}

export default function App() {
  const [datasetName, setDatasetName] = useState("housing-sample");
  const [targetColumn, setTargetColumn] = useState("");
  const [csvText, setCsvText] = useState(SAMPLE_CSV);
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const loadHistory = useCallback(async () => {
    try { setHistory(await fetchHistory()); } catch { /* api may be down */ }
  }, []);
  useEffect(() => { loadHistory(); }, [loadHistory]);

  const runRecommend = async () => {
    setLoading(true); setError("");
    try {
      const { columns, rows } = parseCsv(csvText);
      const rec = await recommend({
        dataset_name: datasetName || null,
        columns, rows,
        target_column: targetColumn.trim() || null
      });
      setResult(rec);
      loadHistory();
    } catch (e) { setError(e.message); }
    finally { setLoading(false); }
  };

  const runFile = async (file) => {
    if (!file) return;
    setLoading(true); setError("");
    try {
      const rec = await recommendCsv(file, datasetName || null, targetColumn.trim() || null);
      setResult(rec);
      loadHistory();
    } catch (e) { setError(e.message); }
    finally { setLoading(false); }
  };

  const openHistory = async (id) => {
    setError("");
    try {
      const d = await fetchDetail(id);
      setResult({ record_id: d.id, ...d });
      window.scrollTo({ top: 0, behavior: "smooth" });
    } catch (e) { setError(e.message); }
  };

  const ch = result?.characteristics;
  return (
    <div className="page">
      <header className="hero">
        <div>
          <p className="eyebrow">scikit-learn · missing-data guide</p>
          <h1>Imputer Advisor</h1>
          <p className="sub">Paste data or upload a CSV — get an explainable <b>Simple / KNN / MICE</b> recommendation.</p>
        </div>
        <div className="hero-badge">KNN vs MICE vs Simple</div>
      </header>

      {error && <div className="alert">⚠ {error}</div>}

      <main className="grid">
        <section className="card">
          <h2>1 · Your data</h2>
          <div className="row">
            <label>Dataset name
              <input value={datasetName} onChange={(e) => setDatasetName(e.target.value)} placeholder="housing-sample" />
            </label>
            <label>Target column (optional, excluded)
              <input value={targetColumn} onChange={(e) => setTargetColumn(e.target.value)} placeholder="price" />
            </label>
          </div>
          <label>CSV (header + rows, empty = missing)
            <textarea rows={9} value={csvText} onChange={(e) => setCsvText(e.target.value)} spellCheck={false} />
          </label>
          <div className="actions">
            <button className="primary" onClick={runRecommend} disabled={loading}>
              {loading ? "Analyzing…" : "Recommend imputer"}
            </button>
            <label className="ghost-btn">Upload CSV
              <input type="file" accept=".csv" hidden onChange={(e) => runFile(e.target.files?.[0])} />
            </label>
            <button className="ghost" onClick={() => setCsvText(SAMPLE_CSV)}>Reset sample</button>
          </div>
        </section>

        <section className="card">
          <h2>2 · Recommendation</h2>
          {!result && <p className="muted">No analysis yet — run the advisor to see scores, characteristics and rationale.</p>}
          {result && (
            <>
              <div className="verdict">
                <span className={`pill ${result.recommended}`}>{LABELS[result.recommended]}</span>
                <span className="conf">confidence {(result.confidence * 100).toFixed(0)}%</span>
              </div>
              {Object.entries(result.scores).map(([k, v]) => (
                <ScoreBar key={k} name={k} value={v} highlight={k === result.recommended} />
              ))}
              {ch && (
                <div className="stats">
                  {[["Rows", ch.n_samples], ["Features", ch.n_features],
                    ["Numeric", ch.n_numeric], ["Categorical", ch.n_categorical],
                    ["Missing", ch.missing_rate_overall], ["Max-col missing", ch.max_col_missing],
                    ["Mean |r|", ch.mean_abs_corr], ["Scale disparity", ch.scale_disparity ?? "—"]
                  ].map(([k, v]) => (
                    <div className="stat" key={k}><span>{k}</span><b>{String(v)}</b></div>
                  ))}
                </div>
              )}
              <p className="rationale">{result.rationale}</p>
            </>
          )}
        </section>
      </main>

      <section className="card history">
        <h2>Recent analyses</h2>
        {history.length === 0 && <p className="muted">Nothing stored yet — recommendations are saved to Postgres.</p>}
        <ul>
          {history.map((h) => (
            <li key={h.id}>
              <button onClick={() => openHistory(h.id)} title="Open detail">
                <b>#{h.id}</b> {h.dataset_name || "unnamed"} · {h.n_samples}×{h.n_features}
                <span className={`mini-pill ${h.recommended}`}>{h.recommended}</span>
              </button>
            </li>
          ))}
        </ul>
      </section>

      <footer>Imputer Advisor · explainable heuristics over scikit-learn imputers · history in PostgreSQL</footer>
    </div>
  );
}
