import { Component, OnInit, AfterContentInit } from '@angular/core';
import { PVL } from '../entities/pvl';
import { Encoding } from '../entities/encoding';
import { Http, Headers } from '@angular/http';
import { map } from "rxjs/operators";
import { ActivatedRoute } from '@angular/router';
import SpectrogramPlugin from 'wavesurfer.js/src/plugin/spectrogram';
import * as WaveSurfer from 'wavesurfer.js';
import RegionPlugin from 'wavesurfer.js/src/plugin/regions'
import { Cluster } from '../entities/cluster';

@Component({
  selector: 'app-annotated-spectrogram',
  templateUrl: './annotated-spectrogram.component.html',
  styleUrls: ['./annotated-spectrogram.component.css']
})
export class AnnotatedSpectrogramComponent implements OnInit {

  clusters: Array<Cluster> = [];
  pvls: Array<PVL> = [];
  encodings: Array<Encoding> = [];
  encoding: string;
  wavesurfer: any;
  cluster_name: string;

  constructor(private route: ActivatedRoute, private http: Http) { 
    route.params.subscribe(
      params => {
        this.encoding = params['encoding'];
        this.cluster_name = params['cluster_name'];
      });
  }

  getClusters() {
    let url = `https://wdp-ds.appspot.com/wdp/ds/${this.cluster_name}/${this.encoding}`;
    let headers = new Headers();
    headers.set('Accept', 'text/json');
    this.http
      .get(url, { headers })
      .pipe(map(resp => resp.json()))
      .subscribe(
        clusters => {
          this.clusters = clusters[0];
        },
        err => {
          console.error(err);
        }
      );
  }  

  getEncodings() {
    let url = "https://wdp-ds.appspot.com/wdp/encodings";
    let headers = new Headers();
    headers.set('Accept', 'text/json');
    this.http
      .get(url, { headers })
      .pipe(map(resp => resp.json()))
      .subscribe(
        encodings => {
          this.encodings = encodings;
        },
        err => {
          console.error(err);
        }
      );
  }

  getAnnotations() {
    let url = `https://wdp-ds.appspot.com/wdp/pvl/encoding/${this.encoding}`;
    let headers = new Headers();
    headers.set('Accept', 'text/json');
    this.http
      .get(url, { headers })
      .pipe(map(resp => resp.json()))
      .subscribe(
        pvls => {
          this.pvls = pvls;
        },
        err => {
          console.error(err);
        }
      );
  }

  setupSpectrogram() {
    this.wavesurfer = WaveSurfer.create({
      container: '#wavcontainer',
      waveColor: '#ddd',
      progressColor: '#333',
      plugins: [
        RegionPlugin.create(),
        SpectrogramPlugin.create({
          container: '#spectrogram',
          labels: true,
        }),
      ],
    });
    let url = `https://wdp-ds.appspot.com/wdp/wav/${this.encoding}`;
    this.wavesurfer.load(url);
    this.wavesurfer.zoom(50);

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

  createRegionsFromClusters() {
    this.clusters.forEach(cluster => {
        console.log(cluster);
        
        this.wavesurfer.addRegion({
          start: cluster.start / 48000,
          end:   cluster.stop  / 48000,
          color: 'rgba(0,255,0,0.2)',
          drag: false,
          data: {
            annotation: `cluster_${cluster.cluster_id}`,
          },
        });
      });    
  }

  createRegionsFromAnnotations() {
    this.pvls.forEach(pvl => {
      console.log(pvl);
        this.wavesurfer.addRegion({
          start: pvl.timecode,
          end:   pvl.timecode + 1,
          color: 'rgba(255,0,0,0.2)',
          drag: false,
          data: { annotation: pvl.description, soundtype: pvl.sound_type },
        });
    });
  }

  ngOnInit() {    
    this.getClusters();
    this.getEncodings();
    this.getAnnotations();
    this.setupSpectrogram();  
    this.wavesurfer.on('ready', () => {
      document.getElementById('loading').style.display = 'none';
      this.createRegionsFromAnnotations();
      this.createRegionsFromClusters();
    });
    this.wavesurfer.on('region-click', function(region, e) {
      e.stopPropagation();
      if (region.data.annotation.startsWith('cluster_')) {
        (<HTMLInputElement>document.getElementById("cluster")).value = region.data.annotation;
      } else {
        (<HTMLInputElement>document.getElementById("description")).value = region.data.annotation;
        (<HTMLInputElement>document.getElementById("soundtype")).value = region.data.soundtype;
      }
      region.play();
  });
  this.wavesurfer.on('region-in', region => {
    if (region.data.annotation.startsWith('cluster_')) {
      (<HTMLInputElement>document.getElementById("cluster")).value = region.data.annotation;
    } else {
      (<HTMLInputElement>document.getElementById("description")).value = region.data.annotation;
      (<HTMLInputElement>document.getElementById("soundtype")).value = region.data.soundtype;
    }
  });
  
  }
}
