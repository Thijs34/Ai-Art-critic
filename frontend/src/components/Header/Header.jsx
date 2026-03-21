import './Header.css'

export default function Header({ hasResult, onReset }) {
  return (
    <header className="header">
      <div className="header__logo">
        <div className="header__logo-name">
          <svg width="7" height="7" viewBox="0 0 7 7" fill="none" aria-hidden="true">
            <rect x="3.5" y="0.2" width="4.67" height="4.67" rx="0.4" transform="rotate(45 3.5 3.5)" fill="var(--gold)" />
          </svg>
          <span>Lumora</span>
        </div>
      </div>

      {!hasResult && (
        <nav className="header__nav" aria-label="Main navigation">
          <a href="#how-it-works" className="header__nav-link">How It Works</a>
          <a href="#about" className="header__nav-link">About</a>
          <a
            href="https://github.com"
            className="header__nav-link"
            target="_blank"
            rel="noopener noreferrer"
          >
            GitHub
          </a>
        </nav>
      )}

      <div className="header__right">
        {hasResult ? (
          <button className="header__reset-btn" onClick={onReset}>
            <svg width="13" height="13" viewBox="0 0 13 13" fill="none" aria-hidden="true">
              <path d="M7.5 2L2.5 6.5L7.5 11" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
            New Artwork
          </button>
        ) : (
          <div className="header__badge">
            <span className="header__badge-dot" aria-hidden="true" />
            Beta
          </div>
        )}
      </div>
    </header>
  )
}
