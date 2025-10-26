import { useEffect, useState } from 'react';
import './App.css';
import CameraFrame from './components/CameraFrame';
import HairsCarousel from './components/HairsCarousel';
import './components/HairsCarousel.css';
import ThemeToogleButton from './components/ThemeToogleButton';

const OPTIONS = { loop: true };
const SLIDE_COUNT = 21;
const SLIDES = Array.from(Array(SLIDE_COUNT).keys());

function App() {
  const [faceShape, setFaceShape] = useState('Analyzing...');

  useEffect(() => {
    const eventSource = new EventSource('http://127.0.0.1:5000/faceshapes');

    eventSource.onmessage = (event) => {
      console.log(event.data);
      setFaceShape(event.data);
    };

    return () => {
      eventSource.close();
    }
    
  }, []);

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
              <HairsCarousel slides={SLIDES} options={OPTIONS} />
            </section>
          </div>
          <div className='recommend barbershop'>
            <div className='recommend-label'>Recommended</div>
            <div className='section-title'>Barbershop</div>
            <div className='current-location'>*around current location</div>
            <div className='recommend-item shop1'>Whata babershop</div>
            <div className='recommend-item shop2'>Whata babershop</div>
            <div className='recommend-item shop3'>Whata babershop</div>
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;