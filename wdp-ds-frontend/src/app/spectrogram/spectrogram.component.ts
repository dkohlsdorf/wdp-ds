import { Component, OnInit, AfterViewInit } from '@angular/core';
import * as WaveSurfer from 'wavesurfer.js';
import SpectrogramPlugin from 'wavesurfer.js/src/plugin/spectrogram'

import { ActivatedRoute } from '@angular/router';

@Component({
  selector: 'app-spectrogram',
  templateUrl: './spectrogram.component.html',
  styleUrls: ['./spectrogram.component.css']
})

export class SpectrogramComponent implements OnInit {

  cluster_name: string;
  asset_name: string;
  wavesurfer: any;

  constructor(private route: ActivatedRoute) {
    route.params.subscribe(
      params => {
        this.cluster_name = params['cluster_name'];
        this.asset_name = params['asset_name'];
      });

  }

  ngOnInit() {
    this.wavesurfer = WaveSurfer.create({
      container: '#wavcontainer',
      waveColor: '#ddd',
      progressColor: '#333',
      plugins: [
        SpectrogramPlugin.create({
          container: '#spectrogram',
          labels: true,
        }),
      ],
    })      
    let url = `https://wdp-ds.appspot.com/wdp/asset/${this.cluster_name}/${this.asset_name}`;
    this.wavesurfer.load(url);
  }
  
  handleClick(value: string) {
    console.log(value)
    if (value === "play") {
      console.log(">> PLAY <<")
      this.wavesurfer.play();
    } 
    if(value === "pause") {
      console.log(">> PAUSE <<")
      this.wavesurfer.pause();
    }
  } 
}

