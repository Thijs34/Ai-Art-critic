import { useState, useCallback } from 'react'
import Header from './components/Header/Header'
import HeroSection from './components/HeroSection/HeroSection'
import LoadingState from './components/LoadingState/LoadingState'
import AnalysisPanel from './components/AnalysisPanel/AnalysisPanel'
import Footer from './components/Footer/Footer'
import './App.css'

// TODO: Replace with real API call — POST /api/analyze with the image file
const mockAnalyze = () =>
  new Promise((resolve) =>
    setTimeout(() => {
      resolve({
        style:     { label: 'Post-Impressionism', confidence: 94 },
        period:    { label: 'Late 19th Century', range: '1886 – 1905', confidence: 91 },
        emotions: [
          { label: 'Awe',         score: 0.42 },
          { label: 'Melancholy',  score: 0.29 },
          { label: 'Serenity',    score: 0.18 },
          { label: 'Excitement',  score: 0.11 },
        ],
        aestheticScore: 7.8,
        composition: {
          rule:    'Rule of Thirds',
          element: 'Atmospheric Light',
          depth:   'High',
        },
        palette:    ['#2b4d6e', '#8b7355', '#c4a882', '#3a3a2e', '#6b8e7f', '#e8d5b0'],
        influences: ['Vincent van Gogh', 'Paul Gauguin', 'Édouard Vuillard'],
        critique:
          'This work demonstrates a sophisticated command of expressive brushwork and chromatic tension. The composition draws the viewer into an interior emotional space, where the interplay of warm and cool tones creates a quiet dialogue between presence and absence. A deeply personal statement rendered in the visual language of its era, balancing spontaneity with deliberate structural intent.',
      })
    }, 2800)
  )

function App() {
  const [imageUrl, setImageUrl]       = useState(null)
  const [imageName, setImageName]     = useState('')
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [analysis, setAnalysis]       = useState(null)

  const handleImageSelect = useCallback(async (file) => {
    const url = URL.createObjectURL(file)
    setImageUrl(url)
    setImageName(file.name)
    setAnalysis(null)
    setIsAnalyzing(true)

    // TODO: swap with: const result = await fetch('/api/analyze', { method: 'POST', body: formData }).then(r => r.json())
    const result = await mockAnalyze(file)
    setAnalysis(result)
    setIsAnalyzing(false)
  }, [])

  const handleReset = useCallback(() => {
    if (imageUrl) URL.revokeObjectURL(imageUrl)
    setImageUrl(null)
    setImageName('')
    setAnalysis(null)
    setIsAnalyzing(false)
  }, [imageUrl])

  const showHero    = !imageUrl
  const showLoading = imageUrl && isAnalyzing
  const showResults = imageUrl && !isAnalyzing && analysis

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
      </main>

      {showHero && <Footer />}
    </div>
  )
}

export default App
