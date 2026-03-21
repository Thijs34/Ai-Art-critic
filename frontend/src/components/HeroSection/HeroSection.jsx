import UploadZone from '../UploadZone/UploadZone'
import './HeroSection.css'

const FEATURE_PILLS = [
  'Art Style', 'Emotional Tone', 'Historical Period',
  'Composition', 'Colour Palette', 'Artist Influences',
]

const STYLE_CHIPS = [
  'Impressionism', 'Baroque', 'Surrealism', 'Romanticism', 'Cubism',
  'Abstract Expressionism', 'Renaissance', 'Realism', 'Post-Impressionism',
  'Symbolism', 'Art Nouveau', 'Modernism', 'Futurism', 'Pop Art',
  'Dadaism', 'Minimalism', 'Expressionism', 'Neoclassicism',
]

const HOW_IT_WORKS = [
  {
    num: '01',
    icon: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
        <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
        <polyline points="17 8 12 3 7 8" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
        <line x1="12" y1="3" x2="12" y2="15" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
      </svg>
    ),
    title: 'Upload Your Artwork',
    desc: 'Drop any painting, photograph, or print. No account or sign-up required.',
  },
  {
    num: '02',
    icon: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
        <path d="M12 2a10 10 0 110 20 10 10 0 010-20z" stroke="currentColor" strokeWidth="1.5" />
        <path d="M12 8v4l3 3" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
      </svg>
    ),
    title: 'AI Analyses',
    desc: 'Our model identifies style, period, composition, emotion, and artistic influences in seconds.',
  },
  {
    num: '03',
    icon: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
        <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z" stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round" />
        <polyline points="14 2 14 8 20 8" stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round" />
        <line x1="8" y1="13" x2="16" y2="13" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
        <line x1="8" y1="17" x2="12" y2="17" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
      </svg>
    ),
    title: 'Read Your Critique',
    desc: 'Receive a full critique: aesthetic score, colour palette, historical context, and expert commentary.',
  },
]

export default function HeroSection({ onImageSelect }) {
  return (
    <div className="hero">
      {/* Background effects */}
      <div className="hero__bg-orb hero__bg-orb--tl" aria-hidden="true" />
      <div className="hero__bg-orb hero__bg-orb--br" aria-hidden="true" />
      <div className="hero__bg-grid"                 aria-hidden="true" />

      {/* ═══════════════════════════════════
          Upload section — above the fold
      ═══════════════════════════════════ */}
      <section className="hero__stage">
        <div className="hero__headline">
          <div className="hero__eyebrow">
            <span className="hero__eyebrow-dot" aria-hidden="true" />
            AI Art Analysis
          </div>

          <h1 className="hero__title">
            Discover the Soul<br />
            <em>of Any Artwork</em>
          </h1>

          <p className="hero__subtitle">
            Upload a painting to get an instant expert critique —
            style, emotion, historical context, and aesthetic score.
          </p>

          <div className="hero__feature-pills" aria-hidden="true">
            {FEATURE_PILLS.map((label, i) => (
              <span
                key={label}
                className="hero__feature-pill"
                style={{ animationDelay: `${0.15 + i * 0.07}s` }}
              >
                {label}
              </span>
            ))}
          </div>
        </div>

        <UploadZone onImageSelect={onImageSelect} />

        <p className="hero__file-hint">
          JPG · PNG · WebP · TIFF · up to 20 MB &nbsp;·&nbsp; Your files are never stored
        </p>
      </section>

      {/* ═══════════════════════════════════
          How It Works
      ═══════════════════════════════════ */}
      <section className="how-it-works" id="how-it-works" aria-labelledby="hiw-heading">
        <header className="how-it-works__header">
          <h2 className="how-it-works__title" id="hiw-heading">How It Works</h2>
          <p className="how-it-works__sub">Three steps. No account. Instant results.</p>
        </header>

        <div className="how-it-works__grid">
          {HOW_IT_WORKS.map(({ num, icon, title, desc }, i) => (
            <article key={i} className="hiw-card">
              <div className="hiw-card__top">
                <div className="hiw-card__icon" aria-hidden="true">{icon}</div>
                <span className="hiw-card__num">{num}</span>
              </div>
              <h3 className="hiw-card__title">{title}</h3>
              <p className="hiw-card__desc">{desc}</p>
            </article>
          ))}
        </div>
      </section>

      {/* ═══════════════════════════════════
          Style ticker — just above footer
      ═══════════════════════════════════ */}
      <div className="style-ticker-section" aria-hidden="true">
        <span className="style-ticker-section__label">27 art styles recognised</span>
        <div className="style-ticker">
          <div className="style-ticker__track">
            {[...STYLE_CHIPS, ...STYLE_CHIPS].map((label, i) => (
              <span key={i} className="style-ticker__chip">{label}</span>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
