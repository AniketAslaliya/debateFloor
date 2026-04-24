import React, { useState, useEffect, useRef } from 'react';
import { Play, CheckCircle, XCircle, AlertCircle, Shield, AlertTriangle, FileText, Gavel, Scale } from 'lucide-react';
import { TASK_STRATEGIES, TASK_DESCRIPTIONS } from './tasks';
import './index.css';

const CALIB_MATRIX = {
  HIGH_correct: { val: 1.0 },
  HIGH_wrong:   { val: -0.8 },
  MED_correct:  { val: 0.6 },
  MED_wrong:    { val: -0.2 },
  LOW_correct:  { val: 0.1 },
  LOW_wrong:    { val: 0.0 }
};

const TASK_STEPS_HINT = {
  clean_claim:             'approve_claim + HIGH confidence',
  contradictory_claim:     'deny_claim + MED confidence + Debate Panel',
  distribution_shift_claim:'escalate_to_human + LOW confidence',
};

function App() {
  const [task, setTask] = useState('contradictory_claim');
  const [isRunning, setIsRunning] = useState(false);
  const [isDone, setIsDone] = useState(false);
  const [claimText, setClaimText] = useState(null);
  const [history, setHistory] = useState([]);
  const [debate, setDebate] = useState(null);

  const [matrixConf, setMatrixConf] = useState(null);
  const [matrixOutcome, setMatrixOutcome] = useState(null);

  const [reward, setReward] = useState('—');
  const [calib, setCalib] = useState('—');
  const [finalOutcome, setFinalOutcome] = useState(null); // 'correct' | 'wrong'

  const [cursorPos, setCursorPos] = useState({ x: -100, y: -100 });
  const [isHovering, setIsHovering] = useState(false);
  const terminalRef = useRef(null);

  useEffect(() => {
    const h = (e) => setCursorPos({ x: e.clientX, y: e.clientY });
    window.addEventListener('mousemove', h);
    return () => window.removeEventListener('mousemove', h);
  }, []);

  // Auto-scroll terminal
  useEffect(() => {
    if (terminalRef.current) {
      terminalRef.current.scrollTop = terminalRef.current.scrollHeight;
    }
  }, [history]);

  const handleRun = async () => {
    setIsRunning(true);
    setIsDone(false);
    setHistory([]);
    setDebate(null);
    setMatrixConf(null);
    setMatrixOutcome(null);
    setReward('—');
    setCalib('—');
    setFinalOutcome(null);
    setClaimText('resetting');

    try {
      const resetRes = await fetch('/reset', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ task_id: task, seed: 42 })
      });
      if (!resetRes.ok) throw new Error('Reset failed');
      const resetData = await resetRes.json();
      const sessionId = resetData.session_id;
      setClaimText(resetData.observation);

      const actions = TASK_STRATEGIES[task];
      let currentHistory = [];

      for (let i = 0; i < actions.length; i++) {
        const action = actions[i];
        const payload = { ...action };
        if (payload.confidence === undefined || payload.confidence === null) delete payload.confidence;

        const stepRes = await fetch('/step', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ session_id: sessionId, action: payload })
        });
        if (!stepRes.ok) throw new Error('Step failed');
        const stepData = await stepRes.json();

        const r = stepData.reward || 0;
        const rb = stepData.observation?.reward_breakdown || {};
        const c = rb.calibration_score;
        const d = stepData.observation?.debate_transcript;

        currentHistory = [...currentHistory, { ...action, reward: r, calibration: c }];
        setHistory([...currentHistory]);
        setReward(r.toFixed(3));

        if (d) setDebate(d);

        if (action.confidence && c !== undefined && c !== null) {
          setCalib(c);
          setMatrixConf(action.confidence);
          const outcome = c >= 0 ? 'correct' : 'wrong';
          setMatrixOutcome(outcome);
          setFinalOutcome(outcome);
        }

        await new Promise(res => setTimeout(res, action.action_type === 'convene_debate_panel' ? 1000 : 550));
      }
      setIsDone(true);
    } catch (err) {
      console.error(err);
      setClaimText('error');
    } finally {
      setIsRunning(false);
    }
  };

  const getMatrixCellClass = (conf, outcome) => {
    const isActive = matrixConf === conf && matrixOutcome === outcome;
    return `matrix-cell cell-${conf.toLowerCase()}-${outcome}${isActive ? ' active' : ''}`;
  };

  const outcomeLabel = finalOutcome === 'correct' ? '✅ CORRECT' : finalOutcome === 'wrong' ? '❌ WRONG' : null;

  return (
    <>
      <div className={`custom-cursor${isHovering ? ' hovering' : ''}`} style={{ left: cursorPos.x, top: cursorPos.y }} />
      <div className="bg-glow" />
      <div className="bg-glow-2" />

      {/* ── TOP NAV BAR ─────────────────────────────── */}
      <nav className="nav-bar">
        <div className="nav-logo">
          <Scale size={22} color="var(--accent-primary)" />
          <span>DebateFloor</span>
        </div>
        <div className="nav-links">
          <a href="https://github.com/AniketAslaliya/debateFloor" target="_blank" rel="noreferrer">GitHub</a>
          <a href="https://arxiv.org/abs/2603.05881" target="_blank" rel="noreferrer">CoCA Paper</a>
          <span className="nav-badge">Meta PyTorch × Scaler 2026</span>
        </div>
      </nav>

      {/* ── HERO BANNER ─────────────────────────────── */}
      <section className="hero-section">
        <div className="hero-content">
          <h1 className="hero-title title-gradient">The AI That Knows When It Doesn't Know</h1>
          <p className="hero-sub">
            DebateFloor trains LLM agents to declare <strong>calibrated confidence</strong> before every insurance decision.
            Overconfident? <span style={{ color: 'var(--error)' }}>Penalised −0.8.</span>&nbsp;
            Wrong but humble? <span style={{ color: 'var(--success)' }}>Rewarded.</span>
          </p>
          <p className="hero-sub" style={{ fontSize: '0.875rem', marginTop: '0.5rem', color: 'var(--text-tertiary)' }}>
            The <strong>Debate Panel</strong> below is unique — no other OpenEnv environment has it. Watch it unfold.
          </p>
        </div>
      </section>

      {/* ── MAIN APP GRID ───────────────────────────── */}
      <div className="app-container">

        {/* SIDEBAR */}
        <div className="flex flex-col gap-4">

          {/* Control Panel */}
          <div className="glass-panel p-6">
            <h2 className="mb-1" style={{ fontSize: '1.1rem' }}>Run an Episode</h2>
            <p className="text-xs text-secondary mb-4">Pick a task, click Run, watch the agent investigate.</p>

            <div className="select-wrapper mb-3"
              onMouseEnter={() => setIsHovering(true)}
              onMouseLeave={() => setIsHovering(false)}>
              <select
                className="custom-select"
                value={task}
                onChange={e => setTask(e.target.value)}
                disabled={isRunning}
              >
                {Object.keys(TASK_STRATEGIES).map(t => (
                  <option key={t} value={t}>{t.replace(/_/g, ' ')}</option>
                ))}
              </select>
            </div>

            <div className="task-hint mb-4">
              <span className="text-xs text-secondary">{TASK_DESCRIPTIONS[task]}</span>
              <br/>
              <span className="text-xs" style={{ color: 'var(--accent-primary)', marginTop: '0.25rem', display: 'inline-block' }}>
                Expected: {TASK_STEPS_HINT[task]}
              </span>
            </div>

            <button
              className="btn-primary"
              onClick={handleRun}
              disabled={isRunning}
              onMouseEnter={() => setIsHovering(true)}
              onMouseLeave={() => setIsHovering(false)}
            >
              <Play size={18} fill="currentColor" />
              {isRunning ? 'Investigating...' : 'Run Episode'}
            </button>

            {isDone && outcomeLabel && (
              <div className={`outcome-badge mt-4 ${finalOutcome}`}>{outcomeLabel}</div>
            )}
          </div>

          {/* Live Metrics */}
          <div className="glass-panel p-6">
            <h3 className="mb-4 text-secondary font-medium" style={{ fontSize: '0.9rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Live Metrics</h3>
            <div className="metric-row">
              <span className="text-secondary text-sm">Reward</span>
              <span className="metric-val" style={{ color: reward !== '—' && parseFloat(reward) >= 0 ? 'var(--success)' : 'var(--error)' }}>{reward}</span>
            </div>
            <div className="metric-row">
              <span className="text-secondary text-sm">Calibration Score</span>
              <span className="metric-val">{calib}</span>
            </div>
            <div className="metric-row">
              <span className="text-secondary text-sm">Declared Confidence</span>
              <span className={`confidence-badge conf-${(matrixConf || '').toLowerCase()}`}>{matrixConf || '—'}</span>
            </div>
            <div className="metric-row">
              <span className="text-secondary text-sm">Steps taken</span>
              <span className="metric-val">{history.length}</span>
            </div>
          </div>

          {/* Calibration Matrix */}
          <div className="glass-panel p-6">
            <h3 className="mb-1" style={{ fontSize: '0.95rem' }}>3×2 Calibration Matrix</h3>
            <p className="text-xs text-secondary mb-4">The highlighted cell = agent's confidence × outcome.<br/><strong style={{ color: 'var(--error)' }}>HIGH + wrong = −0.8</strong> is the worst possible outcome.</p>

            <div className="matrix-container">
              <div className="matrix-header" style={{ borderRight: '1px solid var(--glass-border)' }}>Confidence</div>
              <div className="matrix-header"><CheckCircle size={13} className="inline mr-1" color="var(--success)"/>Correct</div>
              <div className="matrix-header"><XCircle size={13} className="inline mr-1" color="var(--error)"/>Wrong</div>

              {['HIGH', 'MED', 'LOW'].map(conf => (
                <React.Fragment key={conf}>
                  <div className="matrix-label">{conf}</div>
                  <div className={getMatrixCellClass(conf, 'correct')}>
                    <span className="matrix-value">+{CALIB_MATRIX[`${conf}_correct`].val}</span>
                  </div>
                  <div className={getMatrixCellClass(conf, 'wrong')}>
                    <span className="matrix-value">{CALIB_MATRIX[`${conf}_wrong`].val}</span>
                  </div>
                </React.Fragment>
              ))}
            </div>
          </div>

        </div>

        {/* MAIN CONTENT */}
        <div className="flex flex-col gap-4">

          {/* Claim + Terminal side by side */}
          <div className="grid" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1rem' }}>

            {/* Claim Details */}
            <div className="glass-panel p-6" style={{ minHeight: '220px' }}>
              <h3 className="mb-3 flex items-center gap-2" style={{ fontSize: '0.95rem' }}>
                <FileText size={16} color="var(--accent-primary)" /> Claim Under Investigation
              </h3>
              {!claimText && (
                <p className="text-secondary text-sm">Select a task and click Run Episode.</p>
              )}
              {claimText === 'resetting' && (
                <p className="text-secondary text-sm pulse-animation">Contacting environment server...</p>
              )}
              {claimText === 'error' && (
                <p style={{ color: 'var(--error)' }} className="text-sm">⚠ Could not reach environment server.</p>
              )}
              {claimText && typeof claimText === 'object' && (
                <div className="text-sm">
                  <div className="claim-id-tag">#{claimText.claim_id} · {claimText.task_id}</div>
                  <p className="mb-1 mt-2"><strong>Claimant:</strong> {claimText.claimant?.name}</p>
                  <p className="mb-1"><strong>Incident:</strong> {claimText.incident?.type} — {claimText.incident?.description?.slice(0, 90)}...</p>
                  <p className="mb-2"><strong>Amount:</strong> ₹{claimText.payout_amount_inr?.toLocaleString('en-IN') || '—'}</p>
                  <p className="font-medium mb-1 text-secondary">Documents ({claimText.documents?.length || 0}):</p>
                  <ul className="claim-docs">
                    {claimText.documents?.slice(0, 3).map(d => (
                      <li key={d.doc_id}><code>{d.doc_id}</code> — {d.content?.slice(0, 60)}...</li>
                    ))}
                  </ul>
                  {claimText.linked_claims?.length > 0 && (
                    <p className="mt-3 flex items-center gap-2" style={{ color: 'var(--error)', fontWeight: 600 }}>
                      <AlertTriangle size={14} /> {claimText.linked_claims.length} linked claims flagged!
                    </p>
                  )}
                </div>
              )}
            </div>

            {/* Terminal */}
            <div className="terminal-window">
              <div className="terminal-header">
                <div className="terminal-dot dot-red" />
                <div className="terminal-dot dot-yellow" />
                <div className="terminal-dot dot-green" />
                <span className="ml-2 text-xs text-secondary" style={{ fontFamily: 'Inter' }}>agent-trace.log</span>
                {isRunning && <span className="ml-2 text-xs pulse-animation" style={{ color: 'var(--accent-primary)' }}>● LIVE</span>}
              </div>
              <div className="terminal-body" ref={terminalRef}>
                {history.length === 0 ? (
                  <div className="text-secondary" style={{ fontStyle: 'italic' }}>Waiting for episode to start...</div>
                ) : (
                  history.map((h, i) => (
                    <div key={i} className="log-entry">
                      <span className="text-secondary">[{String(i + 1).padStart(2, '0')}]</span>{' '}
                      {h.action_type === 'convene_debate_panel'
                        ? <span style={{ color: 'var(--warning)', fontWeight: 700 }}>⚖ {h.action_type}</span>
                        : <span className="log-action">{h.action_type}</span>
                      }
                      {h.confidence && <span style={{ color: '#c4b5fd' }}> [CONF:{h.confidence}]</span>}
                      <br />
                      <span className="text-secondary pl-6" style={{ fontSize: '0.8rem' }}>↳ {h.reasoning}</span>
                      <br />
                      <span className="pl-6" style={{ fontSize: '0.78rem' }}>
                        reward: <span className="log-reward">{h.reward?.toFixed(3)}</span>
                        {h.calibration !== undefined && h.calibration !== null &&
                          <span style={{ color: '#fcd34d' }}> | calib: {h.calibration}</span>
                        }
                      </span>
                    </div>
                  ))
                )}
              </div>
            </div>

          </div>

          {/* ── DEBATE PANEL — hero section ─────────── */}
          <div className={`debate-container glass-panel p-6${debate ? ' debate-active' : ''}`}>
            <div className="debate-header">
              <Gavel size={20} color={debate ? 'var(--warning)' : 'var(--text-tertiary)'} />
              <h2 style={{ fontSize: '1rem' }}>
                {debate
                  ? `⚖ Debate Panel Convened — Step ${debate.step_convened}`
                  : 'Multi-Agent Debate Panel'}
              </h2>
              {!debate && (
                <span className="text-xs text-secondary ml-2">(appears when agent calls <code>convene_debate_panel</code>)</span>
              )}
            </div>

            {!debate ? (
              <div className="debate-placeholder">
                <p className="text-secondary text-sm">Run <strong>contradictory_claim</strong> to see the Prosecutor vs Defender debate unfold live.</p>
                <div className="debate-preview-grid">
                  <div className="preview-card prosecutor-preview">
                    <strong>Prosecutor</strong>
                    <p>Builds case from discovered fraud signals. Argues for denial.</p>
                  </div>
                  <div className="preview-card defender-preview">
                    <strong>Defender</strong>
                    <p>Argues from document consistency. Assumes innocence.</p>
                  </div>
                </div>
              </div>
            ) : (
              <>
                <div className="debate-grid">
                  <div className="argument-card argument-prosecutor">
                    <div className="argument-header">
                      <span style={{ color: 'var(--error)' }}>⚔ Prosecutor</span>
                      <span className={`strength-badge strength-${(debate.prosecutor_strength || '').toLowerCase()}`}>
                        {debate.prosecutor_strength}
                      </span>
                    </div>
                    <p className="text-sm text-secondary" style={{ lineHeight: '1.65', marginTop: '0.5rem' }}>
                      {debate.prosecutor_argument}
                    </p>
                  </div>
                  <div className="argument-card argument-defender">
                    <div className="argument-header">
                      <span style={{ color: 'var(--success)' }}>🛡 Defender</span>
                      <span className={`strength-badge strength-${(debate.defender_strength || '').toLowerCase()}`}>
                        {debate.defender_strength}
                      </span>
                    </div>
                    <p className="text-sm text-secondary" style={{ lineHeight: '1.65', marginTop: '0.5rem' }}>
                      {debate.defender_argument}
                    </p>
                  </div>
                </div>
                <div className="verdict-box" style={{
                  borderColor: debate.panel_lean === 'prosecution' ? 'var(--error)' : 'var(--success)',
                  color: debate.panel_lean === 'prosecution' ? 'var(--error)' : 'var(--success)',
                  background: debate.panel_lean === 'prosecution' ? 'var(--error-bg)' : 'var(--success-bg)',
                }}>
                  <Gavel size={16} style={{ flexShrink: 0 }} />
                  <span>VERDICT: {debate.panel_verdict}</span>
                </div>
              </>
            )}
          </div>

        </div>
      </div>

      {/* ── FOOTER ──────────────────────────────────── */}
      <footer className="site-footer">
        <span>DebateFloor · Meta PyTorch × Scaler Hackathon 2026 · Based on <a href="https://arxiv.org/abs/2603.05881" target="_blank" rel="noreferrer">CoCA arXiv:2603.05881</a></span>
        <span>Aniket Aslaliya · Mitali Mehta · Aditya Sharma</span>
      </footer>
    </>
  );
}

export default App;
