import { useEffect, useState } from 'react'
import AnalysisCard from '../AnalysisCard/AnalysisCard'
import './AnalysisPanel.css'

/* ── Animated circular score ring (used for style confidence) ── */
function ConfidenceRing({ value }) {
  const [displayed, setDisplayed] = useState(0)
  const r = 34
  const circ = 2 * Math.PI * r
  const filled = (displayed / 100) * circ

  useEffect(() => {
    let frame
    const start = performance.now()
    const duration = 1300
    const animate = (now) => {
      const t = Math.min((now - start) / duration, 1)
      const ease = 1 - Math.pow(1 - t, 3)
      setDisplayed(Math.round(value * ease))
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
        <circle cx="48" cy="48" r={r} stroke="var(--bg-elevated)" strokeWidth="5" fill="none" />
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
        <span className="score-ring__denom">%</span>
      </div>
    </div>
  )
}

/* ── Pending card — shown for models not yet trained ── */
function PendingCard({ label, icon, index }) {
  return (
    <AnalysisCard label={label} icon={icon} index={index} pending>
      <div className="card-pending">
        <div className="card-pending__dot" aria-hidden="true" />
        <span>Model in development</span>
      </div>
    </AnalysisCard>
  )
}

export default function AnalysisPanel({ imageUrl, imageName, analysis, onReset }) {
  const style = analysis.style
  const artist = analysis.artist ?? { label: 'Unknown Artist', confidence: 0 }
  const top5 = analysis.top5 ?? []

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
          <p className="results-header__sub">AI-generated art classification</p>
        </div>

        <div className="results-grid">

          {/* Art Style — live */}
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

          {/* Confidence ring — live */}
          <AnalysisCard label="Confidence" icon="✦" index={1}>
            <ConfidenceRing value={style.confidence} />
          </AnalysisCard>

          {/* Historical Period — pending */}
          <PendingCard label="Historical Period" icon="⏳" index={2} />

          {/* Emotional Tone — pending */}
          <PendingCard label="Emotional Tone" icon="◎" index={3} />

          {/* Context — pending */}
          <PendingCard label="Context" icon="⊞" index={4} />

          {/* Artist — live */}
          <AnalysisCard label="Artist / Looks like" icon="✦" index={5}>
            <p className="card-value">{artist.label}</p>
            <div className="confidence-wrap">
              <div className="confidence-track">
                <div
                  className="confidence-fill"
                  style={{ '--pct': `${artist.confidence}%` }}
                  role="progressbar"
                  aria-valuenow={artist.confidence}
                  aria-valuemin={0}
                  aria-valuemax={100}
                />
              </div>
              <span className="confidence-label">{artist.confidence}%</span>
            </div>
          </AnalysisCard>

          {/* Top-5 breakdown — live */}
          <AnalysisCard label="Style Breakdown" icon="◐" index={6}>
            <div className="top5-list">
              {top5.map((item, i) => (
                <div key={item.label} className="top5-row">
                  <span className="top5-row__rank">{i + 1}</span>
                  <span className="top5-row__label">{item.label}</span>
                  <div className="top5-row__track">
                    <div
                      className="top5-row__fill"
                      style={{ '--pct': `${item.confidence}%`, animationDelay: `${0.4 + i * 0.1}s` }}
                    />
                  </div>
                  <span className="top5-row__pct">{item.confidence}%</span>
                </div>
              ))}
            </div>
          </AnalysisCard>

        </div>
      </section>
    </div>
  )
}
