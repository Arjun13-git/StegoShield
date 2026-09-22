import { useState } from 'react';
import { ShieldCheck, ScanSearch, WandSparkles, FlaskConical } from 'lucide-react';
import { AnalyzePage } from './pages/AnalyzePage';

const tabs = [
  { id: 'analyze', label: 'Analyze', icon: ScanSearch },
  { id: 'encode', label: 'Encode', icon: WandSparkles },
  { id: 'research', label: 'Research', icon: FlaskConical },
] as const;

type Tab = typeof tabs[number]['id'];

export default function App() {
  const [tab, setTab] = useState<Tab>('analyze');

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand"><ShieldCheck size={22} /><span>StegoShield</span></div>
        <div className="status-pill"><span className="status-dot" /> Local analysis mode</div>
      </header>

      <main className="workspace">
        <section className="hero">
          <div>
            <p className="eyebrow">IMAGE STEGANALYSIS</p>
            <h1>Detect what an image may be hiding.</h1>
            <p className="hero-copy">Statistical features + machine learning for explainable, LSB-focused steganography analysis.</p>
          </div>
        </section>

        <nav className="tabs" aria-label="StegoShield modules">
          {tabs.map(({ id, label, icon: Icon }) => (
            <button className={tab === id ? 'tab active' : 'tab'} key={id} onClick={() => setTab(id)}>
              <Icon size={17} /> {label}
            </button>
          ))}
        </nav>

        {tab === 'analyze' && <AnalyzePage />}
        {tab === 'encode' && <Placeholder title="Encode" text="LSB encoder UI will be implemented after the detection pipeline is validated." />}
        {tab === 'research' && <Placeholder title="Research" text="Model comparison, payload sensitivity and robustness charts will live here." />}
      </main>
    </div>
  );
}

function Placeholder({ title, text }: { title: string; text: string }) {
  return <section className="panel empty-state"><h2>{title}</h2><p>{text}</p></section>;
}
