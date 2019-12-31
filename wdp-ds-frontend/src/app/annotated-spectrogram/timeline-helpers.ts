import { TimelineModel } from '../entities/timeline-model'

const getRandomColor = () => {
  const letters = 'BCDEF'.split('')
  let color = '#'
  for (let i = 0; i < 6; i++) {
    color += letters[Math.floor(Math.random() * letters.length)]
  }
  return color
}

export const calcWidth = (zoomValue, audioLength) =>
  10 * zoomValue * zoomValue * audioLength

export const formatTimelineIntervals = pxPerSec => {
  var retval = 1
  if (pxPerSec >= 25 * 100) {
    retval = 0.01
  } else if (pxPerSec >= 25 * 40) {
    retval = 0.025
  } else if (pxPerSec >= 25 * 10) {
    retval = 0.1
  } else if (pxPerSec >= 25 * 4) {
    retval = 0.25
  } else if (pxPerSec >= 25) {
    retval = 1
  } else if (pxPerSec * 5 >= 25) {
    retval = 5
  } else if (pxPerSec * 15 >= 25) {
    retval = 15
  } else {
    retval = Math.ceil(0.5 / pxPerSec) * 60
  }
  return retval
}

export const assignColorsToTimelineRegions = (
  timeline: TimelineModel,
  audioLength: number
): TimelineModel => {
  const colorMap = new Map()
  timeline.regions.forEach(region => {
    const assignedColor = colorMap.get(region.label)
    if (!assignedColor) {
      colorMap.set(region.label, getRandomColor())
    }
    region.percentStart = (region.start / audioLength) * 100
    region.percentWidth = ((region.end / audioLength) * 100) - region.percentStart
    region.showTooltip =  region.percentWidth < 6
    region.color = colorMap.get(region.label)
  })

  return timeline
}

export const assignPositionsToAnnotations = (
  annotations: any,
  audioLength: number
) => {
  return annotations.map(annotation => {
    const percentTime = (annotation.time / audioLength) * 100
    return {
      ...annotation,
      percentTime,
      leftAlign: percentTime < 95,
    }
  })
}
