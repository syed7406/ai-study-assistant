import { useEffect, useMemo, useState } from 'react'

const BASE = import.meta.env.VITE_API_URL || ''

async function api(path, opts = {}) {
  const isForm = opts.body instanceof FormData
  const res = await fetch(`${BASE}${path}`, {
    ...opts,
    headers: isForm ? {} : { 'Content-Type': 'application/json', ...opts.headers },
  })
  const data = await res.json()
  if (!res.ok) throw { response: { data } }
  return data
}

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

function SuccessBanner({ message, onDismiss }) {
  if (!message) return null
  return (
    <div className="success-banner">
      <span>{message}</span>
      <button onClick={onDismiss} aria-label="Dismiss">&times;</button>
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
  const [success, setSuccess] = useState('')
  const [uploadedFiles, setUploadedFiles] = useState([])
  const [openFile, setOpenFile] = useState(null)
  const [fileContent, setFileContent] = useState('')
  const [loadingContent, setLoadingContent] = useState(false)

  const recognitionSupported = useMemo(
    () => typeof window !== 'undefined' && ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window),
    [],
  )

  const fetchDocuments = async () => {
    try {
      const data = await api(`/documents?request_user_id=${userId}`)
      setUploadedFiles(data.documents || [])
    } catch {
      setUploadedFiles([])
    }
  }

  useEffect(() => {
    fetchDocuments()
  }, [])

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
    setSuccess('')
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
    setSuccess('')
    try {
      const data = await api('/upload', { method: 'POST', body: formData })
      setSuccess(`Uploaded "${file.name}" — ${data.chunks_stored} chunks indexed. You can now ask questions about it.`)
      setFile(null)
      fetchDocuments()
    } catch (err) {
      const detail = err.response?.data?.detail || (err && err.message) || 'Upload failed.'
      setError(detail)
    } finally {
      setBusy(false)
    }
  }

  const askQuestion = async () => {
    if (!question.trim()) return
    setBusy(true)
    setError('')
    setSuccess('')
    try {
      const data = await api('/ask', { method: 'POST', body: JSON.stringify({ question, user_id: userId }) })
      setAnswer(data.answer)
      setSources(data.sources || [])
      setConfidence(data.confidence || 0)
    } catch (err) {
      const detail = err.response?.data?.detail || (err && err.message) || 'Failed to get answer.'
      setError(detail)
    } finally {
      setBusy(false)
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !busy) {
      askQuestion()
    }
  }

  const toggleFile = async (filename) => {
    if (openFile === filename) {
      setOpenFile(null)
      setFileContent('')
      return
    }
    setOpenFile(filename)
    setLoadingContent(true)
    try {
      const data = await api(`/documents/content?request_user_id=${userId}&filename=${encodeURIComponent(filename)}`)
      setFileContent(data.content || 'No content available.')
    } catch {
      setFileContent('Failed to load content.')
    } finally {
      setLoadingContent(false)
    }
  }

  return (
    <div className="shell">
      <header className="hero">
        <p className="eyebrow">AI Study Assistant</p>
        <h1>Ask questions grounded in your notes.</h1>
        <p className="lede">Upload documents, get cited answers from your files, and use voice input.</p>
      </header>

      <ErrorBanner message={error} onDismiss={() => setError('')} />
      <SuccessBanner message={success} onDismiss={() => setSuccess('')} />

      <section className="card">
        <h2>Upload Document</h2>
        <p className="hint">Supported: PDF, PPTX, DOCX, TXT, MD (max {MAX_FILE_SIZE_MB}MB)</p>
        <div className="row">
          <input type="file" accept={ACCEPTED_TYPES} onChange={handleFileChange} />
          <button onClick={uploadDocument} disabled={busy || !file}>Upload</button>
        </div>
        {uploadedFiles.length > 0 && (
          <div className="doc-list">
            <p className="hint" style={{ marginTop: '0.75rem', marginBottom: '0.25rem' }}>Loaded documents:</p>
            <ul>
              {uploadedFiles.map((doc) => (
                <li key={doc.filename}>
                  <span className="doc-item" onClick={() => toggleFile(doc.filename)} role="button" tabIndex={0}>
                    <span className="doc-name">{doc.filename}</span>
                    <span className="doc-chunks">({doc.chunks} chunks)</span>
                    <span className="doc-toggle">{openFile === doc.filename ? '▲' : '▼'}</span>
                  </span>
                  {openFile === doc.filename && (
                    <div className="doc-content">
                      {loadingContent ? <em>Loading...</em> : <pre>{fileContent}</pre>}
                    </div>
                  )}
                </li>
              ))}
            </ul>
          </div>
        )}
      </section>

      <section className="card">
        <h2>Ask a question</h2>
        <p className="hint">Ask anything about your uploaded documents</p>
        <div className="row stack-on-mobile">
          <input
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="What is gradient descent?"
          />
          <button onClick={startVoice} disabled={!recognitionSupported}>🎤</button>
          <button onClick={askQuestion} disabled={busy || !question.trim()}>Ask</button>
        </div>
      </section>

      {answer && (
        <section className="card">
          <div className="answer-head">
            <h2>Answer</h2>
            <ConfidenceBadge value={confidence} />
          </div>
          <p>{answer}</p>
          {sources.length > 0 && (
            <>
              <h3>Sources</h3>
              <ul className="sources">
                {sources.map((source, index) => (
                  <li key={`${source.filename}-${index}`}>
                    <strong>{source.filename}</strong> · Page {source.page}
                    <div>{source.excerpt}</div>
                  </li>
                ))}
              </ul>
            </>
          )}
        </section>
      )}
    </div>
  )
}

export default App
