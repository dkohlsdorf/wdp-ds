export class RegionModel {
  start: number
  end: number
  label: number
  percentWidth: number
  percentStart: number
  showTooltip: boolean;
  color: any

  cssStyles: Object
  highlighted: Boolean

  constructor({start, end, label}: { start: any, end: any, label: any }) {
    this.start = start
    this.end = end
    this.label = label
  }

}

export class TimelineModel {
  name: string
  regions: Array<RegionModel>

  constructor(name) {
    this.name = name
    this.regions = new Array<RegionModel>()
  }


  addRegion(region: RegionModel) {
    this.regions.push(region)
  }
}
