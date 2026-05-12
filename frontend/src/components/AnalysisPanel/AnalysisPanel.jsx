import AnalysisCard from '../AnalysisCard/AnalysisCard'
import ArtworkChat from '../ArtworkChat/ArtworkChat'
import './AnalysisPanel.css'

function PendingCard({ label, icon, index, className = '' }) {
  return (
    <AnalysisCard label={label} icon={icon} index={index} pending className={className}>
      <div className="card-pending">
        <div className="card-pending__dot" aria-hidden="true" />
        <span>Not available from this analysis</span>
      </div>
    </AnalysisCard>
  )
}

export default function AnalysisPanel({ imageUrl, imageName, analysis, onReset }) {
  const style = analysis.style
  const artist = analysis.artist ?? { label: 'Unknown Artist', confidence: 0 }
  const top5 = analysis.top5 ?? []
  const timePeriod = analysis.timePeriod
  const emotionalTone = analysis.emotionalTone
  const context = analysis.context
  const styleUsesLocalConfidence = style?.source !== 'openai' && Number.isFinite(style?.confidence)
  const artistUsesLocalConfidence = analysis.confidence?.high_confidence === true && Number.isFinite(artist?.confidence)
  const showStyleBreakdown = styleUsesLocalConfidence && top5.length > 0

  return (
    <div className="analysis-panel">
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

      <section className="analysis-panel__results">
        <div className="results-header">
          <h2 className="results-header__title">Analysis Complete</h2>
          <p className="results-header__sub">AI-generated art classification</p>
        </div>

        <div className="results-grid">
          <AnalysisCard label="Art Style" icon="style" index={0}>
            <p className="card-value">{style.label}</p>
            {styleUsesLocalConfidence ? (
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
            ) : (
              <p className="card-source">OpenAI visual estimate</p>
            )}
          </AnalysisCard>

          <AnalysisCard label="Artist / Looks Like" icon="artist" index={1}>
            <p className="card-value">{artist.label}</p>
            {artistUsesLocalConfidence ? (
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
            ) : (
              <p className="card-source">
                {analysis.usedOpenWorldLlm ? 'OpenAI visual estimate' : 'Attribution uncertain'}
              </p>
            )}
          </AnalysisCard>

          {timePeriod ? (
            <AnalysisCard label="Historical Period" icon="period" index={2}>
              <p className="card-value">{timePeriod}</p>
            </AnalysisCard>
          ) : (
            <PendingCard label="Historical Period" icon="period" index={2} />
          )}

          {emotionalTone ? (
            <AnalysisCard label="Mood & Tone" icon="mood" index={3}>
              <p className="card-value">{emotionalTone}</p>
            </AnalysisCard>
          ) : (
            <PendingCard label="Mood & Tone" icon="mood" index={3} />
          )}

          {context ? (
            <AnalysisCard label="Visual Context" icon="context" index={4} className="analysis-card--wide">
              <p className="card-note">{context}</p>
            </AnalysisCard>
          ) : (
            <PendingCard label="Visual Context" icon="context" index={4} className="analysis-card--wide" />
          )}

          {showStyleBreakdown && (
            <AnalysisCard label="Style Breakdown" icon="breakdown" index={5} className="analysis-card--wide">
              <div className="top5-list">
                {top5.map((item, i) => (
                  <div key={`${item.label}-${i}`} className="top5-row">
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
          )}
        </div>

        <ArtworkChat analysis={analysis} />
      </section>
    </div>
  )
}
