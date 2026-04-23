import React, { useState, useEffect } from 'react';
import { Play, CheckCircle, XCircle, AlertCircle, Shield, AlertTriangle, FileText } from 'lucide-react';
import { TASK_STRATEGIES, TASK_DESCRIPTIONS } from './tasks';
import './index.css';

const CALIB_MATRIX = {
  HIGH_correct: { val: 1.0, color: 'var(--matrix-high-correct)' },
  HIGH_wrong: { val: -0.8, color: 'var(--matrix-high-wrong)' },
  MED_correct: { val: 0.6, color: 'var(--matrix-med-correct)' },
  MED_wrong: { val: -0.2, color: 'var(--matrix-med-wrong)' },
  LOW_correct: { val: 0.1, color: 'var(--matrix-low-correct)' },
  LOW_wrong: { val: 0.0, color: 'var(--text-tertiary)' }
};

function App() {
  const [task, setTask] = useState('contradictory_claim');
  const [isRunning, setIsRunning] = useState(false);
  const [claimText, setClaimText] = useState(null);
  const [history, setHistory] = useState([]);
  const [debate, setDebate] = useState(null);
  
  // Matrix state
  const [matrixConf, setMatrixConf] = useState(null);
  const [matrixOutcome, setMatrixOutcome] = useState(null);

  // Stats
  const [reward, setReward] = useState('—');
  const [calib, setCalib] = useState('—');

  // Custom cursor
  const [cursorPos, setCursorPos] = useState({ x: -100, y: -100 });
  const [isHovering, setIsHovering] = useState(false);

  useEffect(() => {
    const handleMouseMove = (e) => {
      setCursorPos({ x: e.clientX, y: e.clientY });
    };
    window.addEventListener('mousemove', handleMouseMove);
    return () => window.removeEventListener('mousemove', handleMouseMove);
  }, []);

  const handleRun = async () => {
    setIsRunning(true);
    setHistory([]);
    setDebate(null);
    setMatrixConf(null);
    setMatrixOutcome(null);
    setReward('—');
    setCalib('—');
    setClaimText("Resetting environment...");

    try {
      const resetRes = await fetch('/reset', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ task_id: task, seed: 42 })
      });
      if (!resetRes.ok) throw new Error('Reset failed');
      const resetData = await resetRes.json();
      
      const sessionId = resetData.session_id;
      const obs = resetData.observation;
      setClaimText(obs);

      const actions = TASK_STRATEGIES[task];
      let currentHistory = [];

      for (let i = 0; i < actions.length; i++) {
        const action = actions[i];
        const payload = { ...action };
        if (payload.confidence === undefined || payload.confidence === null) {
          delete payload.confidence;
        }

        const stepRes = await fetch('/step', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ session_id: sessionId, action: payload })
        });
        
        if (!stepRes.ok) throw new Error('Step failed');
        const stepData = await stepRes.json();
        
        const r = stepData.reward || 0;
        const c = stepData.observation?.reward_breakdown?.calibration_score;
        const d = stepData.observation?.debate_transcript;
        
        currentHistory = [...currentHistory, { 
          ...action, 
          reward: r, 
          calibration: c 
        }];
        setHistory(currentHistory);
        setReward(r.toFixed(3));
        
        if (d) setDebate(d);
        
        if (action.confidence && c !== undefined && c !== null) {
          setCalib(c);
          setMatrixConf(action.confidence);
          setMatrixOutcome(c >= 0 ? 'correct' : 'wrong');
        }

        await new Promise(resolve => setTimeout(resolve, 600)); // Animation delay
      }
    } catch (err) {
      console.error(err);
      setClaimText("Error: Could not connect to environment.");
    } finally {
      setIsRunning(false);
    }
  };

  const getMatrixCellClass = (conf, outcome) => {
    const isActive = matrixConf === conf && matrixOutcome === outcome;
    return `matrix-cell cell-${conf.toLowerCase()}-${outcome} ${isActive ? 'active' : ''}`;
  };

  return (
    <>
      <div 
        className={`custom-cursor ${isHovering ? 'hovering' : ''}`} 
        style={{ left: cursorPos.x, top: cursorPos.y }}
      />
      <div className="bg-glow"></div>
      <div className="bg-glow-2"></div>
      
      <div className="app-container">
        
        {/* Sidebar */}
        <div className="flex flex-col gap-4">
          <div className="glass-panel p-6">
            <h1 className="title-gradient mb-2" style={{ fontSize: '2rem' }}>DebateFloor <Shield size={28} className="inline ml-2" color="var(--accent-primary)" /></h1>
            <p className="text-sm text-secondary mb-4">
              Insurance claims, calibrated confidence, and multi-agent debate. The worst case is being wrong and overconfident.
            </p>
            
            <div className="select-wrapper mb-4" onMouseEnter={() => setIsHovering(true)} onMouseLeave={() => setIsHovering(false)}>
              <select 
                className="custom-select" 
                value={task} 
                onChange={(e) => setTask(e.target.value)}
                disabled={isRunning}
              >
                {Object.keys(TASK_STRATEGIES).map(t => (
                  <option key={t} value={t}>{t.replace(/_/g, ' ')}</option>
                ))}
              </select>
            </div>
            
            <p className="text-xs text-secondary mb-4 p-4" style={{ background: 'rgba(0,0,0,0.3)', borderRadius: '8px' }}>
              {TASK_DESCRIPTIONS[task]}
            </p>

            <button 
              className="btn-primary" 
              onClick={handleRun} 
              disabled={isRunning}
              onMouseEnter={() => setIsHovering(true)} 
              onMouseLeave={() => setIsHovering(false)}
            >
              <Play size={18} fill="currentColor" />
              {isRunning ? 'Running Episode...' : 'Run Episode'}
            </button>
          </div>

          <div className="glass-panel p-6">
            <h3 className="mb-4 text-secondary font-medium">Live Metrics</h3>
            <div className="flex justify-between items-center mb-2">
              <span className="text-secondary text-sm">Reward</span>
              <span className="font-medium" style={{ color: 'var(--success)' }}>{reward}</span>
            </div>
            <div className="flex justify-between items-center mb-2">
              <span className="text-secondary text-sm">Calibration</span>
              <span className="font-medium">{calib}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-secondary text-sm">Declared Confidence</span>
              <span className="font-medium">{matrixConf || '—'}</span>
            </div>
          </div>
        </div>

        {/* Main Content */}
        <div className="flex flex-col gap-4">
          
          {/* Matrix & Claim Top Row */}
          <div className="grid" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '1rem' }}>
            
            <div className="glass-panel p-6">
              <h3 className="mb-1 text-secondary font-medium">3×2 Calibration Matrix</h3>
              <p className="text-xs text-secondary mb-4">Highlighted cell = confidence × outcome. HIGH + wrong = −0.8 (worst penalty).</p>
              
              <div className="matrix-container">
                <div className="matrix-header" style={{ borderRight: '1px solid var(--glass-border)' }}>Confidence</div>
                <div className="matrix-header"><CheckCircle size={16} className="inline mr-1" color="var(--success)"/> Correct</div>
                <div className="matrix-header"><XCircle size={16} className="inline mr-1" color="var(--error)"/> Wrong</div>
                
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

            <div className="glass-panel p-6">
              <h3 className="mb-4 text-secondary font-medium"><FileText size={18} className="inline mr-2"/>Claim Details</h3>
              {claimText ? (
                typeof claimText === 'string' ? (
                  <div className="p-4" style={{ background: 'rgba(0,0,0,0.3)', borderRadius: '8px' }}>{claimText}</div>
                ) : (
                  <div className="text-sm">
                    <p className="mb-1"><strong>ID:</strong> {claimText.claim_id}</p>
                    <p className="mb-1"><strong>Claimant:</strong> {claimText.claimant?.name}</p>
                    <p className="mb-4"><strong>Incident:</strong> {claimText.incident?.description}</p>
                    <p className="font-medium mb-1 text-secondary">Documents ({claimText.documents?.length || 0}):</p>
                    <ul className="pl-4" style={{ listStyleType: 'circle' }}>
                      {claimText.documents?.slice(0,2).map(d => (
                        <li key={d.doc_id} className="text-secondary">{d.content.slice(0, 50)}...</li>
                      ))}
                    </ul>
                    {claimText.linked_claims?.length > 0 && (
                      <p className="mt-4 text-error font-medium flex items-center gap-2">
                        <AlertTriangle size={16}/> {claimText.linked_claims.length} linked claims flagged!
                      </p>
                    )}
                  </div>
                )
              ) : (
                <div className="h-full flex items-center justify-center text-secondary text-sm">
                  Select a task and run the episode.
                </div>
              )}
            </div>

          </div>

          {/* Action Log Terminal */}
          <div className="terminal-window mt-2">
            <div className="terminal-header">
              <div className="terminal-dot dot-red"></div>
              <div className="terminal-dot dot-yellow"></div>
              <div className="terminal-dot dot-green"></div>
              <span className="ml-2 text-xs text-secondary" style={{ fontFamily: 'Inter' }}>agent-trace.log</span>
            </div>
            <div className="terminal-body" id="terminal-body">
              {history.length === 0 ? (
                <div className="text-secondary italic">Waiting for execution to start...</div>
              ) : (
                history.map((h, i) => (
                  <div key={i} className="log-entry">
                    <span className="text-secondary">[{i + 1}]</span> Executing <span className="log-action">{h.action_type}</span>
                    {h.confidence && <span style={{ color: '#c4b5fd' }}> (Conf: {h.confidence})</span>}
                    <br/>
                    <span className="text-secondary pl-6">↳ {h.reasoning}</span>
                    <br/>
                    <span className="pl-6 text-xs">
                      Reward: <span className="log-reward">+{h.reward?.toFixed(3)}</span> 
                      {h.calibration !== undefined && h.calibration !== null && 
                        <span style={{ color: '#fcd34d' }}> | Calib: {h.calibration}</span>
                      }
                    </span>
                  </div>
                ))
              )}
            </div>
          </div>

          {/* Debate Panel */}
          {debate && (
            <div className="glass-panel p-6 mt-2 debate-panel" style={{ display: 'flex', flexDirection: 'column' }}>
              <div className="flex items-center gap-2 mb-2">
                <AlertCircle size={20} color="var(--warning)" />
                <h3 className="font-medium text-warning" style={{ color: 'var(--warning)' }}>Debate Panel Convened (Step {debate.step_convened})</h3>
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="argument-card argument-prosecutor">
                  <h4 className="font-medium mb-2" style={{ color: 'var(--error)' }}>Prosecutor [{debate.prosecutor_strength}]</h4>
                  <p className="text-sm text-secondary">{debate.prosecutor_argument}</p>
                </div>
                <div className="argument-card argument-defender">
                  <h4 className="font-medium mb-2" style={{ color: 'var(--success)' }}>Defender [{debate.defender_strength}]</h4>
                  <p className="text-sm text-secondary">{debate.defender_argument}</p>
                </div>
              </div>
              
              <div className="mt-4 p-3 rounded-lg text-center font-bold" style={{ 
                background: debate.panel_lean === 'prosecution' ? 'var(--error-bg)' : 'var(--success-bg)',
                color: debate.panel_lean === 'prosecution' ? 'var(--error)' : 'var(--success)',
                border: `1px solid ${debate.panel_lean === 'prosecution' ? 'var(--error)' : 'var(--success)'}`
              }}>
                VERDICT: {debate.panel_verdict}
              </div>
            </div>
          )}

        </div>
      </div>
    </>
  );
}

export default App;
