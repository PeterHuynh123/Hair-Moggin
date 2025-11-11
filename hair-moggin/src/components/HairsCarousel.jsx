import React, { useCallback, useEffect, useRef } from 'react'
import useEmblaCarousel from 'embla-carousel-react'
import {
  NextButton,
  PrevButton,
  usePrevNextButtons
} from './EmblaCarouselArrowButtons.jsx'
import './HairsCarousel.css'

const TWEEN_FACTOR_BASE = 0.52

const numberWithinRange = (number, min, max) =>
  Math.min(Math.max(number, min), max)

const HairsCarousel = ({ hairCuts = [], options = {} }) => {
  const [emblaRef, emblaApi] = useEmblaCarousel(options)
  const tweenFactor = useRef(0)
  const tweenNodes = useRef([])


  const {
    prevBtnDisabled,
    nextBtnDisabled,
    onPrevButtonClick,
    onNextButtonClick
  } = usePrevNextButtons(emblaApi)

  const setTweenNodes = useCallback((emblaApi) => {
    tweenNodes.current = emblaApi.slideNodes().map((slideNode) => {
      return slideNode.querySelector('.embla__slide__number')
    })
  }, [])

  const setTweenFactor = useCallback((emblaApi) => {
    tweenFactor.current = TWEEN_FACTOR_BASE * emblaApi.scrollSnapList().length
  }, [])

  const tweenScale = useCallback((emblaApi, eventName) => {
    const engine = emblaApi.internalEngine()
    const scrollProgress = emblaApi.scrollProgress()
    const slidesInView = emblaApi.slidesInView()
    const isScrollEvent = eventName === 'scroll'

    emblaApi.scrollSnapList().forEach((scrollSnap, snapIndex) => {
      let diffToTarget = scrollSnap - scrollProgress
      const slidesInSnap = engine.slideRegistry[snapIndex]

      slidesInSnap.forEach((slideIndex) => {
        if (isScrollEvent && !slidesInView.includes(slideIndex)) return

        if (engine.options.loop) {
          engine.slideLooper.loopPoints.forEach((loopItem) => {
            const target = loopItem.target()

            if (slideIndex === loopItem.index && target !== 0) {
              const sign = Math.sign(target)

              if (sign === -1) {
                diffToTarget = scrollSnap - (1 + scrollProgress)
              }
              if (sign === 1) {
                diffToTarget = scrollSnap + (1 - scrollProgress)
              }
            }
          })
        }

        const tweenValue = 1 - Math.abs(diffToTarget * tweenFactor.current)
        const scale = numberWithinRange(tweenValue, 0, 1).toString()
        const tweenNode = tweenNodes.current[slideIndex]
        // tweenNode.style.transform = `scale(${scale})`
      })
    })
  }, [])

  useEffect(() => {
    if (!emblaApi) return

    setTweenNodes(emblaApi)
    setTweenFactor(emblaApi)
    tweenScale(emblaApi)

    emblaApi
      .on('reInit', setTweenNodes)
      .on('reInit', setTweenFactor)
      .on('reInit', tweenScale)
      .on('scroll', tweenScale)
      .on('slideFocus', tweenScale)
  }, [emblaApi, tweenScale])

  console.log("carosel", hairCuts)

  return (
    <div className="embla">
     <PrevButton onClick={onPrevButtonClick} disabled={prevBtnDisabled} />
      <div className="embla__viewport" ref={emblaRef}>
        <div className="embla__container">
          {hairCuts.map((hairCut, index) => {
            const medals = ['🥇', '🥈', '🥉']
            const medal = index < 3 ? medals[index] : ''

            return (
              <div className="embla__slide" key={hairCut}>
                <div className="embla__slide-image">
                  <img src={`http://127.0.0.1:5000/image/${hairCut}`} alt={hairCut} className="haircuts_img"/>
                  <div className="embla__slide-label">{medal} {hairCut.slice(0, -4).replaceAll("_", " ")} {medal}</div>
                </div>
              </div>
            )
          })}
        </div>
      </div>
      <NextButton onClick={onNextButtonClick} disabled={nextBtnDisabled} />

    </div>
  )
}

export default HairsCarousel;