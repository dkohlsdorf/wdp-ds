import { Component, OnInit, AfterViewInit, OnDestroy } from '@angular/core';
import * as WaveSurfer from 'wavesurfer.js';
import SpectrogramPlugin from 'wavesurfer.js/src/plugin/spectrogram'
import { Http, Headers } from '@angular/http'
import { map } from "rxjs/operators";
import { ActivatedRoute } from '@angular/router';

@Component({
  selector: 'app-spectrogram',
  templateUrl: './spectrogram.component.html',
  styleUrls: ['./spectrogram.component.css']
})

export class SpectrogramComponent implements OnInit, OnDestroy {

  cluster_name: string;
  asset_name: string;
  wavesurfer: any;
  other_files: Array<File> = [];

  constructor(private route: ActivatedRoute, private http: Http) {
    route.params.subscribe(
      params => {
        this.cluster_name = params['cluster_name'];
        this.asset_name = params['asset_name'];
      });
  }

  get() {
    let url = `/wdp/asset/cluster_files/${this.cluster_name}`;
    let headers = new Headers();
    headers.set('Accept', 'text/json');
    this.http
      .get(url, { headers })
      .pipe(map(resp => resp.json()))
      .subscribe(
        other_files => {
          this.other_files = other_files;
        },
        err => {
          console.error(err);
        }
      );
  }

  ngOnInit() {
    this.get();
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
    });
    let url = `https://wdp-ds.appspot.com/wdp/asset/${this.cluster_name}/${this.asset_name}`;
    this.wavesurfer.load(url);
    this.wavesurfer.zoom(50);
    this.wavesurfer.on('ready', () => {
      document.getElementById('loading').style.display = 'none';
    });
  }  

  ngOnDestroy() {
    this.wavesurfer.stop();
  }
  
  handleClick(value: string) {
    console.log(value)
    if (value === "play") {
      console.log(">> PLAY <<")
      this.wavesurfer.play();
    }
    if (value === "pause") {
      console.log(">> PAUSE <<")
      this.wavesurfer.pause();
    }
  }
}

