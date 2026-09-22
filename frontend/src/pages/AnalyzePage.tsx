import { useState } from 'react';
import { UploadCloud, ShieldAlert } from 'lucide-react';

export function AnalyzePage() {
  const [file, setFile] = useState<File | null>(null);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<any>(null);

  async function analyze() {
    if (!file) return;
    setBusy(true);
    const form = new FormData();
    form.append('file', file);
    try {
      const response = await fetch('http://localhost:8000/api/v1/analyze', { method: 'POST', body: form });
      setResult(await response.json());
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="analysis-grid">
      <div className="panel upload-panel">
        <div className="panel-heading"><div><p className="eyebrow">INPUT</p><h2>Analyze image</h2></div></div>
        <label className="dropzone">
          <UploadCloud size={32} />
          <strong>{file ? file.name : 'Drop an image here'}</strong>
          <span>PNG or JPEG · max 10 MB</span>
          <input type="file" accept="image/png,image/jpeg" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
        </label>
        <button className="primary-button" disabled={!file || busy} onClick={analyze}>{busy ? 'Analyzing…' : 'Run steganalysis'}</button>
      </div>

      <div className="panel result-panel">
        <div className="panel-heading"><div><p className="eyebrow">RESULT</p><h2>Detection signal</h2></div><ShieldAlert size={20} /></div>
        {!result ? <div className="empty-result"><span>—</span><p>Upload an image to see the model output.</p></div> : (
          <div>
            <div className="score"><strong>{Number(result.stego_score).toFixed(1)}%</strong><span>{result.verdict?.replaceAll('_', ' ')}</span></div>
            <div className="indicator-list">{(result.indicators ?? []).map((x: any) => <div className="indicator" key={x.name}><span>{x.name}</span><b>{Number(x.value).toFixed(3)}</b></div>)}</div>
            <p className="fine-print">Model: {result.model} · {result.model_version}</p>
          </div>
        )}
      </div>
    </section>
  );
}
