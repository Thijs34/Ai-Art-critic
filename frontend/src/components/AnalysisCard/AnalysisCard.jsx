import './AnalysisCard.css'

export default function AnalysisCard({ label, icon, children, index = 0, pending = false }) {
  return (
    <div
      className={`analysis-card${pending ? ' analysis-card--pending' : ''}`}
      style={{ animationDelay: `${index * 90}ms` }}
    >
      <div className="analysis-card__header">
        {icon && <span className="analysis-card__icon" aria-hidden="true">{icon}</span>}
        <span className="analysis-card__label">{label}</span>
      </div>
      <div className="analysis-card__body">
        {children}
      </div>
    </div>
  )
}
