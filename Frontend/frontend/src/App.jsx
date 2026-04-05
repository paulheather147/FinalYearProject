import { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  // state variables for file upload, preview, results, loading state, and error handling
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // set file state and create a preview URL when user selects a file
  const handleFileChange = (e) => {
    const f = e.target.files[0];
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setError(null);
  };

  // handle form submission to send the image file to the backend API for analysis
  const handleSubmit = async () => {
    if (!file) return;
    
    setLoading(true);
    setError(null);
    
    // create FormData object to send the file in a multipart/form-data POST request
    const formData = new FormData();
    formData.append("file", file);
    try {
      const res = await axios.post("http://localhost:8000/predict", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      // set the result state with the response data from the backend
      setResult(res.data);
    } catch (err) {
      console.error("API error:", err);
      setError(err.response?.data?.detail || "Error analyzing image");
      setResult(null);
    } finally {
      // reset loading state after API call completes to allow for another file to be uploaded 
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <h1>Alzheimer's MRI Classifier</h1>
      
      <div className="upload-section">
        <input 
          type="file" 
          accept="image/*" 
          onChange={handleFileChange}
          disabled={loading}
        />
        {preview && (
          <img 
            src={preview} 
            alt="preview" 
            style={{ width: 256, maxHeight: 256, objectFit: 'contain' }} 
          />
        )}
        <button onClick={handleSubmit} disabled={!file || loading}>
          {loading ? "Analysing..." : "Analyse"}
        </button>
      </div>

      {error && (
        <div className="error">
          <p><strong>Error:</strong> {error}</p>
        </div>
      )}

      {result && (
        <div className="results">
          <div className="classification">
            <p><strong>Predicted Class:</strong> {result.class}</p>
            <p><strong>Confidence:</strong> {(result.confidence * 100).toFixed(1)}%</p>
          </div>

          {result.images && result.images.length > 0 && (
            <div className="output-images">
              <h3>Analysis Visualizations</h3>
              <div className="images-grid">
                {result.images.map((imageBase64, index) => (
                  <div key={index} className="image-container">
                    <img 
                      src={`data:image/png;base64,${imageBase64}`}
                      alt={`Result ${index + 1}`}
                      style={{ maxWidth: '300px', maxHeight: '300px' }}
                    />
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default App;
