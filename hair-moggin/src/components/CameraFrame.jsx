import React, { useState, useRef, useEffect } from 'react';
import cameraFrame from '../assets/camera frame.png';
import loadingGif from '../assets/loading.gif';
import './CameraFrame.css';

const CameraFrame = () => {
  const [showCamera, setShowCamera] = useState(false);
  const videoRef = useRef(null);
  const streamRef = useRef(null); // To keep track of the stream for cleanup

  // Handle camera on/off
  useEffect(() => {
    if (showCamera) {
      navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' } })
        .then((stream) => {
          if (videoRef.current) {
            videoRef.current.srcObject = stream;
          }
          streamRef.current = stream;
        })
        .catch((err) => {
          console.error("Error accessing camera:", err);
          setShowCamera(false);
        });
    } else {
      // stop camera stream
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
        streamRef.current = null;
      }
      if (videoRef.current) {
        videoRef.current.srcObject = null;
      }
    }
  }, [showCamera]);

  const handleLoadingClick = () => {
    setShowCamera(true);
  };

  const handleVideoClick = () => {
    setShowCamera(false);
  };

  return (
    <div className="camera-frame-container">
      <img 
        src={cameraFrame} 
        
        onClick= {showCamera ? handleVideoClick : handleLoadingClick} 
        alt="Camera Frame" 
        className="camera-frame" 
        style={{ pointerEvents: 'auto' }}
      />
      <div className="camera-frame-content">
        <img src="http://127.0.0.1:5000/video_feed" className="camera-feed" /> 
        {/* {!showCamera ? (
          <img 
            src={loadingGif} 
            alt="Loading" 
            className="loading-gif" 
            onClick={handleLoadingClick}
            style={{ cursor: 'pointer' }}
          />
        ) : (
          // <video
          //   ref={videoRef}
          //   className="camera-feed"
          //   autoPlay
          //   playsInline
          //   onClick={handleVideoClick}
          // />

          
        )} */}
      </div>
    </div>
  );
};

export default CameraFrame; 