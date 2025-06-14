import './App.css';
import CameraFrame from './components/CameraFrame';

function App() {
  return (
    <div className="root">
      <header className="header">
        <h1 className="title">HAIR MOGGIN</h1>
      </header>
      <main className="main">
        <section className="camera-mobile-wrapper">
          <CameraFrame />
        </section>
        <section className="left">
          <div className="instruct">click on the screen to start webcam</div>
          <div className="faceshapes">
            <h2 className="section-title">Faceshapes</h2>
            <div className="recommend-item s1">Best Match: SHAPE 1</div>
            <div className="recommend-item s2">Best Match: SHAPE 1</div>
            <div className="recommend-item s3">Best Match: SHAPE 1</div>
          </div>
          <div className="filler filler-left">
            Lorem Ipsum<br/>
            Hihi haha<br/>
            WHAT the<br/>
            Whasddg EH<br/>
            OH wowww<br/>
            okay thats all<br/>
            But i guess don<br/>
            Tirila tirlai sasd<br/>
            Brr brr tun tung tun <br/>
            skibid ie oNe fi don<br/>
            lorem ipsum<br/>
            lorem ipsum<br/>
            lorem ipsum<br/>  
            outnei
          </div>
        </section>
        <section className="right">
          <div className="recommend haircut">
            <div className="recommend-label">Recommended</div>
            <div className="section-title">Haircut</div>
            <div className="recommend-item cut1">Whata cut</div>
            <div className="recommend-item cut2">Whata cut</div>
            <div className="recommend-item cut3">Whata cut</div>
          </div>
          <div className="recommend barbershop">
            <div className="recommend-label">Recommended</div>
            <div className="section-title">Barbershop</div>
            <div className="current-location">*around current location</div>
            <div className="recommend-item shop1">Whata babershop</div>
            <div className="recommend-item shop2">Whata babershop</div>
            <div className="recommend-item shop3">Whata babershop</div>
          </div>
          
          <div className="filler filler-right">
            Ipsum<br/>
            Hihi haha<br/>
            WHAT the<br/>
            But i guess don<br/>
            Tirila tirlai sasd<br/>
            Brr brr tun tung tun <br/>
            skibid ie oNe fi don<br/>
            lorem ipsum<br/>
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;
