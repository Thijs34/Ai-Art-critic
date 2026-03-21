import { useRef, useState, useCallback } from 'react'
import './UploadZone.css'

const ACCEPTED_TYPES = ['image/jpeg', 'image/png', 'image/webp', 'image/tiff', 'image/gif']

function CanvasIcon() {
  return (
    <svg className="upload-zone__canvas-icon" width="72" height="72" viewBox="0 0 72 72" fill="none" aria-hidden="true">
      {/* Outer frame */}
      <rect x="2" y="2" width="68" height="68" rx="3" stroke="currentColor" strokeWidth="1.2" />
      {/* Inner mat */}
      <rect x="11" y="11" width="50" height="50" rx="2" stroke="currentColor" strokeWidth="1" strokeDasharray="4 4" opacity="0.3" />
      {/* Upload arrow */}
      <path d="M36 44 L36 30" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
      <path d="M28 38 L36 29 L44 38" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  )
}

function CameraIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path d="M23 19a2 2 0 01-2 2H3a2 2 0 01-2-2V8a2 2 0 012-2h4l2-3h6l2 3h4a2 2 0 012 2z"
        stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round" />
      <circle cx="12" cy="13" r="4" stroke="currentColor" strokeWidth="1.5" />
    </svg>
  )
}

export default function UploadZone({ onImageSelect }) {
  const inputRef                    = useRef(null)
  const [isDragging, setIsDragging] = useState(false)
  const [error, setError]           = useState(null)

  const processFile = useCallback((file) => {
    if (!file) return
    if (!ACCEPTED_TYPES.includes(file.type)) {
      setError('Please upload a JPG, PNG, WebP, TIFF, or GIF.')
      return
    }
    setError(null)
    onImageSelect(file)
  }, [onImageSelect])

  const handleDragOver = useCallback((e) => {
    e.preventDefault()
    setIsDragging(true)
  }, [])

  const handleDragLeave = useCallback((e) => {
    if (!e.currentTarget.contains(e.relatedTarget)) setIsDragging(false)
  }, [])

  const handleDrop = useCallback((e) => {
    e.preventDefault()
    setIsDragging(false)
    processFile(e.dataTransfer.files[0])
  }, [processFile])

  const openPicker = useCallback((e) => {
    e?.stopPropagation()
    inputRef.current?.click()
  }, [])

  return (
    <div
      className={`upload-zone ${isDragging ? 'upload-zone--dragging' : ''}`}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      onClick={openPicker}
      role="button"
      tabIndex={0}
      aria-label="Upload artwork — click or drag a file"
      onKeyDown={(e) => e.key === 'Enter' && openPicker()}
    >
      {/* Corner brackets */}
      <span className="corner corner--tl" aria-hidden="true" />
      <span className="corner corner--tr" aria-hidden="true" />
      <span className="corner corner--bl" aria-hidden="true" />
      <span className="corner corner--br" aria-hidden="true" />

      {/* Inner glow sweep on drag */}
      {isDragging && <div className="upload-zone__drag-glow" aria-hidden="true" />}

      <div className="upload-zone__body">
        <CanvasIcon />

        <p className="upload-zone__instruction">
          {isDragging ? 'Release to analyse' : 'Drop your artwork here'}
        </p>

        {!isDragging && (
          <div className="upload-zone__actions">
            <button
              className="upload-zone__browse-btn"
              type="button"
              onClick={openPicker}
            >
              Browse Files
            </button>

            <button
              className="upload-zone__camera-btn"
              type="button"
              disabled
              title="Camera vision coming soon"
              onClick={(e) => e.stopPropagation()}
            >
              <CameraIcon />
              Camera
              <span className="upload-zone__soon">Soon</span>
            </button>
          </div>
        )}

        {error && <p className="upload-zone__error" role="alert">{error}</p>}
      </div>

      <input
        ref={inputRef}
        type="file"
        accept={ACCEPTED_TYPES.join(',')}
        onChange={(e) => processFile(e.target.files[0])}
        style={{ display: 'none' }}
        aria-hidden="true"
      />
    </div>
  )
}
