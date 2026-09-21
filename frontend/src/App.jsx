import { useState, useRef, useCallback } from 'react'
import './App.css'

const API_URL = 'http://localhost:8000'

function App() {
  const [selectedFile, setSelectedFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [dragging, setDragging] = useState(false)
  const fileInputRef = useRef(null)

  // Handle file selection
  const handleFileSelect = useCallback((file) => {
    if (!file) return
    const allowed = ['image/jpeg', 'image/png', 'image/bmp', 'image/webp']
    if (!allowed.includes(file.type)) {
      setError('Please upload a valid image file (JPG, PNG, BMP, or WEBP)')
      return
    }
    setSelectedFile(file)
    setPreviewUrl(URL.createObjectURL(file))
    setResult(null)
    setError(null)
  }, [])

  // Drag & Drop handlers
  const handleDragOver = useCallback((e) => {
    e.preventDefault()
    setDragging(true)
  }, [])

  const handleDragLeave = useCallback(() => {
    setDragging(false)
  }, [])

  const handleDrop = useCallback((e) => {
    e.preventDefault()
    setDragging(false)
    const file = e.dataTransfer.files[0]
    handleFileSelect(file)
  }, [handleFileSelect])

  const handleInputChange = useCallback((e) => {
    handleFileSelect(e.target.files[0])
  }, [handleFileSelect])

  // Send image to FastAPI backend for prediction
  const handlePredict = async () => {
    if (!selectedFile) return

    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const formData = new FormData()
      formData.append('file', selectedFile)

      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        const errData = await response.json()
        throw new Error(errData.detail || 'Prediction failed')
      }

      const data = await response.json()
      setResult(data)
    } catch (err) {
      setError(err.message || 'Could not connect to backend server. Make sure it is running on port 8000.')
    } finally {
      setLoading(false)
    }
  }

  // Reset everything
  const handleReset = () => {
    setSelectedFile(null)
    setPreviewUrl(null)
    setResult(null)
    setError(null)
    if (fileInputRef.current) fileInputRef.current.value = ''
  }

  const isTumor = result?.prediction === 'Pituitary Tumor'

  return (
    <div className="app-container">
      {/* ---- Header ---- */}
      <header className="header">
        <div className="header-logo">
          <div className="logo-icon">N</div>
          <h1>NeuroScan AI <span>v1.0</span></h1>
        </div>
        <div className="header-status">
          <div className="status-dot"></div>
          Model Active
        </div>
      </header>

      {/* ---- Main Content ---- */}
      <main className="main-content">
        <div className="section-title">
          <h2>Pituitary Tumor Detection</h2>
          <p>Upload an MRI scan for AI-powered diagnosis with Grad-CAM explainability</p>
        </div>

        {/* ---- Upload Card ---- */}
        <div
          className={`upload-card ${dragging ? 'dragging' : ''}`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
        >
          <input
            ref={fileInputRef}
            type="file"
            className="file-input"
            accept="image/jpeg,image/png,image/bmp,image/webp"
            onChange={handleInputChange}
          />

          {!previewUrl ? (
            <>
              <div className="upload-icon">&#8682;</div>
              <h3>Upload MRI Image</h3>
              <p>Drag & drop your MRI scan here, or click to browse</p>
              <p style={{ marginTop: '8px', fontSize: '12px', color: '#444' }}>
                Supports: JPG, PNG, BMP, WEBP
              </p>
            </>
          ) : (
            <div className="preview-section" onClick={(e) => e.stopPropagation()}>
              <img src={previewUrl} alt="MRI Preview" className="preview-image" />
              <div className="file-info">
                <div className="file-name">{selectedFile?.name}</div>
                <div className="file-size">
                  {(selectedFile?.size / 1024).toFixed(1)} KB
                </div>
                <button
                  className="predict-btn"
                  style={{ marginTop: '16px', padding: '10px 32px', fontSize: '14px' }}
                  onClick={handleReset}
                >
                  Change Image
                </button>
              </div>
            </div>
          )}
        </div>

        {/* ---- Predict Button ---- */}
        {selectedFile && !result && (
          <div style={{ textAlign: 'center', marginTop: '24px' }}>
            <button
              className={`predict-btn ${loading ? 'loading' : ''}`}
              onClick={handlePredict}
              disabled={loading}
            >
              {loading ? '' : 'Analyze MRI'}
              {loading && <div className="spinner"></div>}
            </button>
          </div>
        )}

        {/* ---- Error Message ---- */}
        {error && (
          <div className="error-message">
            {error}
          </div>
        )}

        {/* ---- Results Section ---- */}
        {result && (
          <div className="results-section">
            {/* Prediction Result Card */}
            <div className="result-card">
              <div className="result-header">
                <div className="result-label-group">
                  <div className={`result-badge ${isTumor ? 'tumor' : 'no-tumor'}`}>
                    <span className="badge-icon">{isTumor ? '!' : '✓'}</span>
                    {result.prediction}
                  </div>
                </div>

                <div className="confidence-section">
                  <div>
                    <div className="confidence-label">Confidence</div>
                    <div className="confidence-value">
                      {result.confidence_percent}
                    </div>
                  </div>
                </div>
              </div>

              {/* Confidence Bar */}
              <div className="confidence-bar-container">
                <div className="confidence-bar-bg">
                  <div
                    className={`confidence-bar-fill ${isTumor ? 'tumor' : 'no-tumor'}`}
                    style={{ width: `${result.confidence * 100}%` }}
                  ></div>
                </div>
              </div>

              {/* Metrics */}
              <div className="metrics-grid">
                <div className="metric-item">
                  <div className="metric-value">{result.prediction === 'Pituitary Tumor' ? 'Positive' : 'Negative'}</div>
                  <div className="metric-label">Detection</div>
                </div>
                <div className="metric-item">
                  <div className="metric-value">{result.confidence_percent}</div>
                  <div className="metric-label">Confidence</div>
                </div>
                <div className="metric-item">
                  <div className="metric-value">{result.raw_score}</div>
                  <div className="metric-label">Raw Score</div>
                </div>
                <div className="metric-item">
                  <div className="metric-value">MobileNetV2</div>
                  <div className="metric-label">Model</div>
                </div>
              </div>
            </div>

            {/* Grad-CAM Visualization Card */}
            <div className="gradcam-card">
              <h3>Grad-CAM Explainability</h3>
              <p className="gradcam-subtitle">
                Red regions indicate areas the AI focused on for its diagnosis.
                This does not confirm a medical finding — it shows model attention areas.
              </p>
              <div className="gradcam-image-container">
                <img
                  src={`data:image/png;base64,${result.gradcam_image}`}
                  alt="Grad-CAM Visualization"
                />
              </div>
            </div>

            {/* New Scan Button */}
            <div style={{ textAlign: 'center', marginTop: '32px' }}>
              <button className="predict-btn" onClick={handleReset}>
                New Scan
              </button>
            </div>
          </div>
        )}
      </main>

      {/* ---- Footer ---- */}
      <footer className="footer">
        <span>NeuroScan AI</span> — Pituitary Tumor Detection with Explainable AI | MobileNetV2 + Grad-CAM
      </footer>
    </div>
  )
}

export default App
