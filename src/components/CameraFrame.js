import React from 'react';
import cameraFrame from '../assets/camera frame.png';
import loadingGif from '../assets/loading.gif';
import './CameraFrame.css';

const CameraFrame = () => {
  return (
    <div className="camera-frame-container">
      <img src={cameraFrame} alt="Camera Frame" className="camera-frame" />
      <img src={loadingGif} alt="Loading" className="loading-gif" />
    </div>
  );
};

export default CameraFrame; 