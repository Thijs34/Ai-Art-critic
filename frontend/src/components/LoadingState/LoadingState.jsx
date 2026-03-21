import { useEffect, useState } from 'react'
import './LoadingState.css'

const STEPS = [
  'Identifying artistic style…',
  'Reading emotional tone…',
  'Mapping historical context…',
  'Analysing composition…',
  'Extracting colour palette…',
  'Compiling critique…',
]

export default function LoadingState({ imageUrl }) {
  const [stepIndex, setStepIndex] = useState(0)

  useEffect(() => {
    const id = setInterval(() => {
      setStepIndex((prev) => Math.min(prev + 1, STEPS.length - 1))
    }, 420)
    return () => clearInterval(id)
  }, [])

  return (
    <div className="loading">
      <div className="loading__artwork-wrap">
        <img src={imageUrl} alt="Artwork being analysed" className="loading__artwork" />
        <div className="loading__scan-overlay" aria-hidden="true">
          <div className="loading__scan-line" />
        </div>
        <div className="loading__vignette" aria-hidden="true" />
      </div>

      <div className="loading__info">
        <div className="loading__dots" aria-hidden="true">
          <span /><span /><span />
        </div>
        <p className="loading__step" aria-live="polite">
          {STEPS[stepIndex]}
        </p>
        <p className="loading__label">AI analysis in progress</p>
      </div>
    </div>
  )
}
