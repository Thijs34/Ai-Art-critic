import './AnalysisCard.css'

const icons = {
  style: (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M4 6.5h16M4 12h16M4 17.5h9" />
      <path d="M17.5 15.5l2 2 2.5-4" />
    </svg>
  ),
  confidence: (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M12 3v3M12 18v3M3 12h3M18 12h3" />
      <path d="M7.8 7.8l2.4 2.4M16.2 7.8l-2.4 2.4M7.8 16.2l2.4-2.4M16.2 16.2l-2.4-2.4" />
    </svg>
  ),
  period: (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M12 7v5l3 2" />
      <path d="M5 12a7 7 0 1 0 2-4.9" />
      <path d="M5 4.5V7h2.5" />
    </svg>
  ),
  mood: (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M12 21s-7-4.4-7-10a4 4 0 0 1 7-2.6A4 4 0 0 1 19 11c0 5.6-7 10-7 10z" />
    </svg>
  ),
  context: (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M5 5h14v14H5z" />
      <path d="M8 9h8M8 13h8M8 17h4" />
    </svg>
  ),
  artist: (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M12 12a4 4 0 1 0 0-8 4 4 0 0 0 0 8z" />
      <path d="M4.5 20a7.5 7.5 0 0 1 15 0" />
    </svg>
  ),
  breakdown: (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M5 19V9M12 19V5M19 19v-7" />
      <path d="M4 19h16" />
    </svg>
  ),
}

export default function AnalysisCard({ label, icon, children, index = 0, pending = false, className = '' }) {
  const iconNode = icons[icon]
  const classes = [
    'analysis-card',
    pending ? 'analysis-card--pending' : '',
    className,
  ].filter(Boolean).join(' ')

  return (
    <div
      className={classes}
      style={{ animationDelay: `${index * 90}ms` }}
    >
      <div className="analysis-card__header">
        {iconNode && <span className={`analysis-card__icon analysis-card__icon--${icon}`}>{iconNode}</span>}
        <span className="analysis-card__label">{label}</span>
      </div>
      <div className="analysis-card__body">
        {children}
      </div>
    </div>
  )
}
