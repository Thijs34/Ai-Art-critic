import './Footer.css'

export default function Footer() {
  return (
    <footer className="footer">
      <div className="footer__inner">
        {/* Brand */}
        <div className="footer__brand">
          <div className="footer__logo">
            <svg width="16" height="16" viewBox="0 0 20 20" fill="none" aria-hidden="true">
              <rect x="1" y="1" width="18" height="18" rx="2" stroke="var(--gold)" strokeWidth="1.2" />
              <path d="M5 15 L10 5 L15 15" stroke="var(--gold)" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round" />
              <path d="M7 11 L13 11" stroke="var(--gold)" strokeWidth="1.2" strokeLinecap="round" />
            </svg>
            <span>Artiq</span>
          </div>
          <p className="footer__tagline">
            AI-powered art analysis &amp; critique.
            <br />
            Built for art lovers, researchers, and creators.
          </p>
        </div>

        {/* Links */}
        <div className="footer__links-group">
          <span className="footer__links-title">Product</span>
          <a href="#how-it-works" className="footer__link">How It Works</a>
          <a href="#" className="footer__link">Features</a>
          <a href="#" className="footer__link">Roadmap</a>
        </div>

        <div className="footer__links-group">
          <span className="footer__links-title">Explore</span>
          <a href="#" className="footer__link">Art Styles Guide</a>
          <a href="#" className="footer__link">Emotion Taxonomy</a>
          <a href="#" className="footer__link">Model Overview</a>
        </div>

        <div className="footer__links-group">
          <span className="footer__links-title">Connect</span>
          <a href="https://github.com" className="footer__link" target="_blank" rel="noopener noreferrer">GitHub</a>
          <a href="#" className="footer__link">Contact</a>
          <a href="#" className="footer__link">Privacy</a>
        </div>
      </div>

      <div className="footer__bottom">
        <p className="footer__copy">
          © {new Date().getFullYear()} Artiq. Made with care for the art world.
        </p>
        <p className="footer__disclaimer">
          Artiq is an AI tool. Analyses are generative and intended for exploration, not attribution.
        </p>
      </div>
    </footer>
  )
}
