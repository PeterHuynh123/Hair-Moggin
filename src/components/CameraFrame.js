import React, { useState, useRef, useEffect } from 'react';
import cameraFrame from '../assets/camera frame.png';
import loadingGif from '../assets/loading.gif';
import './CameraFrame.css';

const CameraFrame = () => {
  const [showCamera, setShowCamera] = useState(false);
  const videoRef = useRef(null);

  // run this when showCamera changes
  useEffect(() => {
    if (showCamera) {
      // Request access to the user's camera
      navigator.mediaDevices.getUserMedia({ video: true })
        .then((stream) => {
          if (videoRef.current) {
            videoRef.current.srcObject = stream;
          }
        })
        .catch((err) => {
          console.error("Error accessing camera:", err);
          setShowCamera(false); // Reset state if camera access fails
        });
    }
  }, [showCamera]);

  const handleLoadingClick = () => {
    setShowCamera(true);
  };

  return (
    <div className="camera-frame-container">
      <img 
        src={cameraFrame} 
        onClick={handleLoadingClick} 
        alt="Camera Frame" 
        className="camera-frame" 
        style={{ pointerEvents: 'auto' }}
      />
      <div className="camera-frame-content"> 
        {!showCamera ? (
          <img 
            src={loadingGif} 
            alt="Loading" 
            className="loading-gif" 
            onClick={handleLoadingClick}
            style={{ cursor: 'pointer' }}
          />
        ) : (
          <video
            ref={videoRef}
            className="camera-feed"
            autoPlay
            playsInline // doesnt go full screen on mobile
            style={{ width: '100%', height: '100%' }}
          />
        )}
      </div>
    </div>
  );
};

export default CameraFrame; 