import React, { useState } from 'react';
import './App.css';

function App() {
  const [videoFile, setVideoFile] = useState(null);
  const [analysisResults, setAnalysisResults] = useState('');
  const [command, setCommand] = useState('');

  const handleVideoUpload = (event) => {
    setVideoFile(event.target.files[0]);
  };

  const handleVoiceCommand = () => {
    // Example voice command handling (you'll need to implement actual voice recognition logic)
    const mockCommand = "Analyze the video"; // This would come from voice recognition
    setCommand(mockCommand);
    analyzeVideo(videoFile);
  };

  const analyzeVideo = (file) => {
    // Placeholder for video analysis logic
    if (file) {
      setAnalysisResults(`Analysis results for ${file.name}: ...`);
    } else {
      setAnalysisResults('No video uploaded.');
    }
  };

  return (
    <div className="App">
      <h1>Voice-Controlled CCTV Video Analyzer</h1>
      <input type="file" accept="video/*" onChange={handleVideoUpload} />
      <button onClick={handleVoiceCommand}>Start Voice Command</button>
      <h2>Voice Command: {command}</h2>
      <div className="results">
        <h2>Analysis Results:</h2>
        <p>{analysisResults}</p>
      </div>
    </div>
  );
}

export default App;