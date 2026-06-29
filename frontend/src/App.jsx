import { useMemo, useState } from 'react'
import axios from 'axios'

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || 'http://localhost:8000',
})

const ACCEPTED_TYPES = '.pdf,.pptx,.docx,.txt,.md'
const MAX_FILE_SIZE_MB = 20

function ConfidenceBadge({ value }) {
  const tone = value >= 80 ? 'high' : value >= 50 ? 'mid' : 'low'
  return <span className={`badge ${tone}`}>{value}% confidence</span>
}

function ErrorBanner({ message, onDismiss }) {
  if (!message) return null
  return (
    <div className="error-banner">
      <span>{message}</span>
      <button onClick={onDismiss} aria-label="Dismiss error">&times;</button>
    </div>
  )
}

function App() {
  const [userId] = useState('demo-user')
  const [question, setQuestion] = useState('')
  const [answer, setAnswer] = useState(null)
  const [sources, setSources] = useState([])
  const [confidence, setConfidence] = useState(0)
  const [busy, setBusy] = useState(false)
  const [file, setFile] = useState(null)
  const [error, setError] = useState('')

  const recognitionSupported = useMemo(
    () => typeof window !== 'undefined' && ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window),
    [],
  )

  const startVoice = () => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition
    if (!SpeechRecognition) return
    const recognition = new SpeechRecognition()
    recognition.lang = 'en-US'
    recognition.onresult = (event) => {
      const transcript = event.results[0][0].transcript
      setQuestion(transcript)
    }
    recognition.onerror = () => setError('Voice recognition failed. Please try again.')
    recognition.start()
  }

  const handleFileChange = (e) => {
    const selected = e.target.files?.[0] || null
    setError('')
    if (!selected) {
      setFile(null)
      return
    }
    if (selected.size > MAX_FILE_SIZE_MB * 1024 * 1024) {
      setError(`File too large. Maximum size is ${MAX_FILE_SIZE_MB}MB.`)
      setFile(null)
      return
    }
    setFile(selected)
  }

  const uploadDocument = async () => {
    if (!file) return
    const formData = new FormData()
    formData.append('file', file)
    formData.append('user_id', userId)
    setBusy(true)
    setError('')
    try {
      await api.post('/upload', formData, { headers: { 'Content-Type': 'multipart/form-data' } })
      setFile(null)
    } catch (err) {
      const detail = err.response?.data?.detail || err.message || 'Upload failed.'
      setError(detail)
    } finally {
      setBusy(false)
    }
  }

  const askQuestion = async () => {
    if (!question.trim()) return
    setBusy(true)
    setError('')
    try {
      const res = await api.post('/ask', { question, user_id: userId })
      setAnswer(res.data.answer)
      setSources(res.data.sources || [])
      setConfidence(res.data.confidence || 0)
    } catch (err) {
      const detail = err.response?.data?.detail || err.message || 'Failed to get answer.'
      setError(detail)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="shell">
      <header className="hero">
        <p className="eyebrow">AI Study Assistant</p>
        <h1>Ask questions grounded in your notes.</h1>
        <p className="lede">Upload PDFs, get cited answers, and use voice input from the browser.</p>
      </header>

      <ErrorBanner message={error} onDismiss={() => setError('')} />

      <section className="card">
        <h2>Upload Document</h2>
        <p className="hint">Supported formats: PDF, PPTX, DOCX, TXT, MD (max {MAX_FILE_SIZE_MB}MB)</p>
        <div className="row">
          <input type="file" accept={ACCEPTED_TYPES} onChange={handleFileChange} />
          <button onClick={uploadDocument} disabled={busy || !file}>Upload</button>
        </div>
      </section>

      <section className="card">
        <h2>Ask a question</h2>
        <div className="row stack-on-mobile">
          <input value={question} onChange={(e) => setQuestion(e.target.value)} placeholder="What is gradient descent?" />
          <button onClick={startVoice} disabled={!recognitionSupported}>🎤</button>
          <button onClick={askQuestion} disabled={busy}>Ask</button>
        </div>
      </section>

      {answer && (
        <section className="card">
          <div className="answer-head">
            <h2>Answer</h2>
            <ConfidenceBadge value={confidence} />
          </div>
          <p>{answer}</p>
          <h3>Sources</h3>
          <ul className="sources">
            {sources.map((source, index) => (
              <li key={`${source.filename}-${index}`}>
                <strong>{source.filename}</strong> · Page {source.page}
                <div>{source.excerpt}</div>
              </li>
            ))}
          </ul>
        </section>
      )}
    </div>
  )
}

export default App
