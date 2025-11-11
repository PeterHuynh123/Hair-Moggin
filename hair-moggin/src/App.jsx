import { useEffect, useState } from 'react';
import './App.css';
import CameraFrame from './components/CameraFrame';
import HairsCarousel from './components/HairsCarousel';
import './components/HairsCarousel.css';
import ThemeToogleButton from './components/ThemeToogleButton';

const OPTIONS = { axis: 'y', dragFree: true }
const SLIDE_COUNT = 21;
const SLIDES = Array.from(Array(SLIDE_COUNT).keys());

function App() {
  const [faceShape, setFaceShape] = useState('Analyzing...');
  const [hairCuts, setHaircuts] = useState([
                "textured_crop.png",
                "quiff.png",
                "pompadour.png",
                "side_part.png",
                "crew_cut.png",
                "slicked_back.png",
                "medium_wavy_top.png",
                "buzz_cut.png",
                "messy_fringe.png",
                "side_swept_fringe.png",
                "angular_fringe.png",
                "medium_wavy_side_sweep.png",
                "faux_hawk.png",
                "high_fade.png"
            ])
  const [isLocked, setIsLocked] = useState(false)

  useEffect(() => {
    const eventSource = new EventSource('http://127.0.0.1:5000/faceshapes');

    // fetch('http://127.0.0.1:5000/shape-images')
    //   .then(res => res.json())
    //   .then(data => {
    //     console.log(data)
    //   }
    //   )

    eventSource.onmessage = (event) => {
      console.log(event.data);
      // Only update if not locked
      if (!isLocked) {
        setFaceShape(event.data);
        if (event.data !== "Please re-adjust your face") {
          fetch(`http://127.0.0.1:5000/haircuts/${event.data}`)
          .then(res => res.json())
          .then(data => {
            setHaircuts(data);
            console.log(faceShape, data)
          }
          )
          .catch(error => {
            console.log(error)
          })
        }
      }
    };

    return () => {
      eventSource.close();
    }
    
  }, [isLocked, faceShape]);

  return (
    <div className='root'>
      <ThemeToogleButton />
      <header className='header'>
        <h1 className='title'>HAIR MAXXING</h1>
      </header>
      <main className='main'>
        <section className='camera-mobile-wrapper'>
          <CameraFrame />
        </section>
        <section className='left'>
          <div className='instruct'>click on the screen to start webcam</div>
          <div className='faceshapes'>
            <h2 className='section-title'>Faceshapes</h2>
            <div className='recommend-item'>Best Match: {faceShape}</div>
            <button 
              className='lock-button'
              onClick={() => setIsLocked(!isLocked)}
            >
              {isLocked ? 'Unlock faceshape stream' : 'Lock faceshape'}
            </button>
          </div>
        </section>
        <section className='right'>
          <div className='recommend haircut'>
            <div className='recommend-label'>Recommended</div>
            <div className='section-title'>Haircuts</div>
            {/* <div className="recommend-item cut1">Whata cut</div>
            <div className="recommend-item cut2">Whata cut</div>
            <div className="recommend-item cut3">Whata cut</div> */}
            <section className='embla'>
              <HairsCarousel hairCuts={hairCuts} options={OPTIONS} />
            </section>
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;