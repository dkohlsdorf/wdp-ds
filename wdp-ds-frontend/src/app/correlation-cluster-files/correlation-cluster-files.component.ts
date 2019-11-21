import { Component, OnInit, ElementRef, ViewChild, AfterContentInit } from '@angular/core';
import * as d3 from 'd3';
import { ActivatedRoute } from '@angular/router';

@Component({
  selector: 'app-correlation-cluster-files',
  templateUrl: './correlation-cluster-files.component.html',
  styleUrls: ['./correlation-cluster-files.component.css']
})
export class CorrelationClusterFilesComponent implements OnInit, AfterContentInit {
  @ViewChild('chart', { static: true }) chartElement: ElementRef;

  cluster_name: string;

  constructor(private route: ActivatedRoute) {
    route.params.subscribe(
      params => {
        this.cluster_name = params['cluster_name'];
      });
  }

  heatmap() {
    var margin = { top: 10, right: 10, bottom: 50, left: 50 };
    var width = 800 - margin.left - margin.right;
    var height = 600 - margin.top - margin.bottom;
    var svg = d3.select(this.chartElement.nativeElement)
      .append("svg")
      .attr("width", width + margin.left + margin.right)
      .attr("height", height + margin.top + margin.bottom)
      .append("g")
      .attr("transform", "translate(" + margin.left + "," + margin.top + ")");
    let url = `https://wdp-ds.appspot.com/wdp/ds/correlate/${this.cluster_name}`;
    console.log(url);

    d3.json(url).then(function (data) {
      var encodings = Array.from(new Set(data.map(cooc => Number(cooc.encoding)))).sort();
      var clusters = Array.from(new Set(data.map(cooc => Number(cooc.cluster_id)))).sort();
      var x = d3.scaleBand()
        .range([0, width])
        .domain(clusters)
        .padding(0.01);
      svg.append("g")
        .attr("transform", "translate(0," + height + ")")
        .call(d3.axisBottom(x));
      var y = d3.scaleBand()
        .range([height, 0])
        .domain(encodings)
        .padding(0.01);
      svg.append("g")
        .call(d3.axisLeft(y));
      var colors = d3.scaleLinear()
        .range(["white", "#69b3a2"])
        .domain([1, 100]);
      svg.selectAll()
        .data(data, function (cooc) { return cooc.cluster_id + ':' + cooc.encoding; })
        .enter()
        .append("rect")
        .attr("x", function (cooc) { return x(cooc.cluster_id) })
        .attr("y", function (cooc) { return y(cooc.encoding) })
        .attr("width", x.bandwidth())
        .attr("height", y.bandwidth())
        .style("fill", function (cooc) { return colors(cooc.n) })
    });
  }
  ngOnInit() { }
  ngAfterContentInit() {
    this.heatmap();
  }
}
