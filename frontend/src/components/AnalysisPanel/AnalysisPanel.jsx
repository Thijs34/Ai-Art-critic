import { useEffect, useState } from 'react'
import AnalysisCard from '../AnalysisCard/AnalysisCard'
import './AnalysisPanel.css'

/* ── Animated circular score ring ── */
function ScoreRing({ value, max = 10 }) {
  const [displayed, setDisplayed] = useState(0)
  const r = 34
  const circ = 2 * Math.PI * r
  const filled = (displayed / max) * circ

  useEffect(() => {
    let frame
    const start = performance.now()
    const duration = 1300
    const animate = (now) => {
      const t = Math.min((now - start) / duration, 1)
      const ease = 1 - Math.pow(1 - t, 3)
      setDisplayed(parseFloat((value * ease).toFixed(1)))
      if (t < 1) frame = requestAnimationFrame(animate)
    }
    frame = requestAnimationFrame(animate)
    return () => cancelAnimationFrame(frame)
  }, [value])

  return (
    <div className="score-ring">
      <svg width="96" height="96" viewBox="0 0 96 96" fill="none" aria-hidden="true">
        <defs>
          <linearGradient id="score-grad" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="var(--gold)" />
            <stop offset="100%" stopColor="var(--gold-light)" />
          </linearGradient>
        </defs>
        {/* Track */}
        <circle cx="48" cy="48" r={r} stroke="var(--bg-elevated)" strokeWidth="5" fill="none" />
        {/* Fill */}
        <circle
          cx="48" cy="48" r={r}
          stroke="url(#score-grad)"
          strokeWidth="5"
          fill="none"
          strokeLinecap="round"
          strokeDasharray={`${filled} ${circ}`}
          transform="rotate(-90 48 48)"
        />
      </svg>
      <div className="score-ring__text">
        <span className="score-ring__value">{displayed}</span>
        <span className="score-ring__denom">/ {max}</span>
      </div>
    </div>
  )
}

export default function AnalysisPanel({ imageUrl, imageName, analysis, onReset }) {
  const { style, period, emotions, aestheticScore, composition, palette, influences, critique } = analysis
  const paletteGradient = `linear-gradient(90deg, ${palette.join(', ')})`

  return (
    <div className="analysis-panel">
      {/* ── Left: Artwork ── */}
      <aside className="analysis-panel__artwork-col">
        <div className="artwork-frame">
          <div className="artwork-frame__mat">
            <img src={imageUrl} alt="Analysed artwork" className="artwork-frame__img" />
          </div>
        </div>
        <div className="artwork-meta">
          <p className="artwork-meta__name" title={imageName}>{imageName}</p>
          <button className="artwork-meta__reset" onClick={onReset}>
            <svg width="13" height="13" viewBox="0 0 13 13" fill="none" aria-hidden="true">
              <path d="M6.5 1v3L9 1.5M6.5 1A5.5 5.5 0 106.5 12" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
            Analyse another
          </button>
        </div>
      </aside>

      {/* ── Right: Results ── */}
      <section className="analysis-panel__results">
        <div className="results-header">
          <h2 className="results-header__title">Analysis Complete</h2>
          <p className="results-header__sub">AI-generated art critique &amp; classification</p>
        </div>

        <div className="results-grid">

          {/* Art Style */}
          <AnalysisCard label="Art Style" icon="🖼" index={0}>
            <p className="card-value">{style.label}</p>
            <div className="confidence-wrap">
              <div className="confidence-track">
                <div
                  className="confidence-fill"
                  style={{ '--pct': `${style.confidence}%` }}
                  role="progressbar"
                  aria-valuenow={style.confidence}
                  aria-valuemin={0}
                  aria-valuemax={100}
                />
              </div>
              <span className="confidence-label">{style.confidence}%</span>
            </div>
          </AnalysisCard>

          {/* Aesthetic Score */}
          <AnalysisCard label="Aesthetic Score" icon="✦" index={1}>
            <ScoreRing value={aestheticScore} />
          </AnalysisCard>

          {/* Historical Period */}
          <AnalysisCard label="Historical Period" icon="⏳" index={2}>
            <p className="card-value">{period.label}</p>
            <p style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 6 }}>{period.range}</p>
            <div className="confidence-wrap">
              <div className="confidence-track">
                <div
                  className="confidence-fill"
                  style={{ '--pct': `${period.confidence}%` }}
                  role="progressbar"
                  aria-valuenow={period.confidence}
                  aria-valuemin={0}
                  aria-valuemax={100}
                />
              </div>
              <span className="confidence-label">{period.confidence}%</span>
            </div>
          </AnalysisCard>

          {/* Emotional Resonance */}
          <AnalysisCard label="Emotional Resonance" icon="◎" index={3}>
            <div className="emotion-row">
              {emotions.map((e) => (
                <div key={e.label} className="emotion-bar">
                  <span className="emotion-bar__label">{e.label}</span>
                  <div className="emotion-bar__track">
                    <div
                      className="emotion-bar__fill"
                      style={{ '--pct': `${Math.round(e.score * 100)}%` }}
                      role="progressbar"
                      aria-valuenow={Math.round(e.score * 100)}
                      aria-valuemin={0}
                      aria-valuemax={100}
                    />
                  </div>
                  <span className="emotion-bar__pct">{Math.round(e.score * 100)}%</span>
                </div>
              ))}
            </div>
          </AnalysisCard>

          {/* Composition */}
          <AnalysisCard label="Composition" icon="⊞" index={4}>
            <div className="metric-grid">
              <div className="metric-cell">
                <div className="metric-cell__key">Principle</div>
                <div className="metric-cell__val">{composition.rule}</div>
              </div>
              <div className="metric-cell">
                <div className="metric-cell__key">Focal Element</div>
                <div className="metric-cell__val">{composition.element}</div>
              </div>
              <div className="metric-cell">
                <div className="metric-cell__key">Visual Depth</div>
                <div className="metric-cell__val">{composition.depth}</div>
              </div>
            </div>
          </AnalysisCard>

          {/* Colour Palette */}
          <AnalysisCard label="Colour Palette" icon="◐" index={5}>
            <div className="palette-gradient" style={{ background: paletteGradient }} aria-hidden="true" />
            <div className="palette-row">
              {palette.map((hex) => (
                <div key={hex} className="swatch">
                  <div className="swatch__color" style={{ background: hex }} title={hex} />
                  <span className="swatch__hex">{hex}</span>
                </div>
              ))}
            </div>
          </AnalysisCard>

          {/* Artistic Influences */}
          <AnalysisCard label="Artistic Influences" icon="✦" index={6}>
            <div className="influence-list">
              {influences.map((name) => (
                <div key={name} className="influence-item">{name}</div>
              ))}
            </div>
          </AnalysisCard>

          {/* Critical Analysis */}
          <AnalysisCard label="Critical Analysis" icon="✎" index={7}>
            <p className="critique-text">"{critique}"</p>
          </AnalysisCard>

        </div>
      </section>
    </div>
  )
}
