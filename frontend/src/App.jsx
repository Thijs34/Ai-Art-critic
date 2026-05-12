import { useState, useCallback } from 'react'
import Header from './components/Header/Header'
import HeroSection from './components/HeroSection/HeroSection'
import LoadingState from './components/LoadingState/LoadingState'
import AnalysisPanel from './components/AnalysisPanel/AnalysisPanel'
import Footer from './components/Footer/Footer'
import './App.css'

const API_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:5000'

async function analyzeImage(file) {
  const formData = new FormData()
  formData.append('image', file)

  const res = await fetch(`${API_URL}/api/predict-artist-open-world`, {
    method: 'POST',
    body: formData,
  })

  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.error || `Server error ${res.status}`)
  }

  const data = await res.json()

  return {
    style: data.style,
    artist: data.artist,
    top5: data.top5,
    styleTopk: data.style_topk ?? [],
    artistTopk: data.artist_topk ?? [],
    finalArtist: data.final_artist ?? data.artist?.label ?? null,
    candidates: data.candidates ?? [],
    retrievalHits: data.retrieval_hits ?? [],
    llm: data.llm ?? null,
    llmError: data.llm_error ?? null,
    confidence: data.confidence ?? null,
    usedOpenWorldLlm: data.used_open_world_llm ?? false,
    usedOpenAiStyle: data.used_openai_style ?? false,
    timePeriod: data.time_period ?? null,
    emotionalTone: data.emotional_tone ?? null,
    artworkTitle: data.title ?? null,
    context: data.context ?? null,
  }
}

function App() {
  const [imageUrl, setImageUrl]       = useState(null)
  const [imageName, setImageName]     = useState('')
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [analysis, setAnalysis]       = useState(null)

  const [analysisError, setAnalysisError] = useState(null)

  const handleImageSelect = useCallback(async (file) => {
    const url = URL.createObjectURL(file)
    setImageUrl(url)
    setImageName(file.name)
    setAnalysis(null)
    setAnalysisError(null)
    setIsAnalyzing(true)

    try {
      const result = await analyzeImage(file)
      setAnalysis(result)
    } catch (err) {
      console.error('Analysis failed:', err)
      setAnalysisError(err.message)
    } finally {
      setIsAnalyzing(false)
    }
  }, [])

  const handleReset = useCallback(() => {
    if (imageUrl) URL.revokeObjectURL(imageUrl)
    setImageUrl(null)
    setImageName('')
    setAnalysis(null)
    setAnalysisError(null)
    setIsAnalyzing(false)
  }, [imageUrl])

  const showHero    = !imageUrl
  const showLoading = imageUrl && isAnalyzing
  const showResults = imageUrl && !isAnalyzing && analysis
  const showError   = imageUrl && !isAnalyzing && analysisError

  return (
    <div className="app">
      <Header hasResult={!!imageUrl} onReset={handleReset} />

      <main className="app__main">
        {showHero    && <HeroSection onImageSelect={handleImageSelect} />}
        {showLoading && <LoadingState imageUrl={imageUrl} />}
        {showResults && (
          <AnalysisPanel
            imageUrl={imageUrl}
            imageName={imageName}
            analysis={analysis}
            onReset={handleReset}
          />
        )}
        {showError && (
          <div className="app__error">
            <p className="app__error-title">Analysis failed</p>
            <p className="app__error-msg">{analysisError}</p>
            <button className="app__error-retry" onClick={handleReset}>Try another image</button>
          </div>
        )}
      </main>

      {showHero && <Footer />}
    </div>
  )
}

export default App
