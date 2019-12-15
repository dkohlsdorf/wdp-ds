import { Component, OnInit, AfterContentInit, ViewChild, ElementRef } from '@angular/core';
import { PVL } from '../entities/pvl';
import { Encoding } from '../entities/encoding';
import { Http, Headers, ResponseContentType } from '@angular/http';
import { map, switchMap, flatMap } from "rxjs/operators";
import { ActivatedRoute } from '@angular/router';
import SpectrogramPlugin from 'wavesurfer.js/src/plugin/spectrogram';
import * as WaveSurfer from 'wavesurfer.js';
import RegionPlugin from 'wavesurfer.js/src/plugin/regions'
import { Cluster } from '../entities/cluster';
import * as Colors from '../entities/colors';
import * as Timing from "../spectrogram/timing_utils";
import { Spectrogram } from "wdp-ds-spectrogram";

const SPEC_STEP = 256;
const SPEC_WIN = 512;
const WIN_SIZE = 8.5;
const N_TICKS = 10;

@Component({
  selector: 'app-annotated-spectrogram',
  templateUrl: './annotated-spectrogram.component.html',
  styleUrls: ['./annotated-spectrogram.component.css']
})
export class AnnotatedSpectrogramComponent implements OnInit, AfterContentInit {
  @ViewChild('drawing', { static: true }) canvas: ElementRef;

  // app
  cluster_name: string;
  asset_name: string;
  other_files: Array<File> = [];

  // audio 
  audioContext: AudioContext;
  audioSource: AudioBufferSourceNode;
  audioBuffer: AudioBuffer;

  // animation
  ctx: CanvasRenderingContext2D;
  rate = null;
  end = null;
  spectrogram = null;
  spectrogram_tmp = null;
  stopped = true;
  window_size = null;
  started = 0;
  last_slice = 0;
  offset = 0;

  clusters: Array<Cluster> = [];
  pvls: Array<PVL> = [];
  encodings: Array<Encoding> = [];
  encoding: string;
  wavesurfer: any;

  constructor(private route: ActivatedRoute, private http: Http) { 
    route.params.subscribe(
      params => {
        this.encoding = params['encoding'];
        this.cluster_name = params['cluster_name'];
      });
  }

  log(s) {
    console.log(`Frame ${s}`);
  }

  setup_playback() {
    this.audioSource = this.audioContext.createBufferSource();
    this.audioSource.buffer = this.audioBuffer;
    this.audioSource.connect(this.audioContext.destination);
  }

  spectrogram_seek(slice_i, fft_win, fft_step, end) {
    console.log(this, this.audioBuffer);
    const start = Timing.slice2raw(slice_i, this.window_size, fft_step, end);
    const mid = Timing.slice2raw(slice_i + 1, this.window_size, fft_step, end);
    const stop = Timing.slice2raw(slice_i + 2, this.window_size, fft_step, end);
    if (this.spectrogram == null) {
      this.spectrogram = Spectrogram.from_audio(this.audioBuffer.getChannelData(0).slice(start, mid), fft_win, fft_step);
    } else {
      this.spectrogram = this.spectrogram_tmp;
    }
    if (stop > mid) {
      var _this = this;
      setTimeout(function () {
        this.spectrogram_tmp = Spectrogram.from_audio(this.audioBuffer.getChannelData(0).slice(mid, stop), fft_win, fft_step);
      }.bind(this), 1);
    }
  }

  ticks(slice_i) {
    const start_raw = Timing.slice2raw(slice_i, this.window_size, SPEC_STEP, this.end);
    const start = Timing.spec_frame(start_raw, SPEC_STEP);
    const stop = Timing.spec_frame(Timing.slice2raw(slice_i + 1, this.window_size, SPEC_STEP, this.end), SPEC_STEP);
    const start_sec = start_raw / this.rate;
    const tick = (stop - start) / N_TICKS;
    const n_sec = (tick * SPEC_STEP) / this.rate;
    for (var i = 0; i < N_TICKS; i++) {
      const sec = Timing.fmtMSS(Math.round(start_sec + i * n_sec));
      this.ctx.fillStyle = "#FF0000";
      this.ctx.fillText(`${sec}`, i * tick, 50);
    }
    const n_bins = SPEC_WIN / (2 * N_TICKS);
    const freq_bin = (this.rate / 2) / (SPEC_WIN / 2);
    for (var i = 0; i < N_TICKS; i++) {
      if (256 - i * n_bins > 55) {
        this.ctx.fillStyle = "#FF0000";
        this.ctx.fillText(`${i * freq_bin * n_bins}`, 10, 256 - i * n_bins);
      }
    }
  }

  playback_head(in_slice) {
    this.ctx.beginPath();
    this.ctx.moveTo(in_slice, 0);
    this.ctx.lineTo(in_slice, 256);
    this.ctx.stroke();
  }

  plot(slice, in_slice) {
    this.ctx.clearRect(0, 0, this.canvas.nativeElement.width, this.canvas.nativeElement.height);
    this.spectrogram.plot(this.ctx);
    this.ctx.strokeStyle = "#FF0000";
    this.ctx.lineWidth = 5;
    this.playback_head(in_slice);
    this.ticks(slice);
  }

  loop() {
    console.log(this);
    var playback = Timing.playback_sec(this.audioContext.currentTime, this.started);
    if (this.stopped) {
      playback = 0;
    }
    let seconds = playback + this.offset;
    let samples = Timing.raw_samples(seconds, this.rate);
    let frames = Timing.spec_frame(samples, SPEC_STEP);
    let slice = Timing.slice_i(frames, this.window_size);
    let in_slice = Timing.slice_t(frames, this.window_size);
    this.log(seconds)
    if (!this.stopped && this.last_slice != slice) {
      this.spectrogram_seek(slice, SPEC_WIN, SPEC_STEP, this.end);
      this.last_slice = slice;
    }
    this.plot(slice, in_slice);
    if (seconds < this.end && !this.stopped) {
      requestAnimationFrame(this.loop.bind(this));
    }
  }

  loadSound(url) {
    let headers = new Headers();
    headers.set('Accept', 'text');
    this.http.get(url, { headers })
      .pipe(
        map(r => r.text()),
        switchMap(url => {
          return this.http.get(url, { responseType: ResponseContentType.ArrayBuffer });
        }),
        flatMap(r => this.audioContext.decodeAudioData(r.arrayBuffer()))
      ).subscribe(buffer => {
        console.log("Assign Buffer", this);
        this.audioBuffer = buffer;
        this.end = this.audioBuffer.getChannelData(0).length;
        this.rate = this.audioBuffer.sampleRate;
        this.window_size = Timing.sec2frames(WIN_SIZE, this.rate, SPEC_STEP);     
        this.spectrogram_seek(0, SPEC_WIN, SPEC_STEP, this.end);
        requestAnimationFrame(this.loop.bind(this));       
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

  handleClick(value: string) {
    console.log(value)
    if (value === "play") {
      if (this.stopped) {
        this.setup_playback();
        this.stopped = false;
        this.last_slice = Timing.slice_i(Timing.spec_frame(Timing.raw_samples(this.offset, this.rate), SPEC_STEP), this.window_size);
        this.started = this.audioContext.currentTime;
        this.audioSource.start(0, this.offset);
        requestAnimationFrame(this.loop.bind(this));
      }
    }
    if (value === "pause") {
      if (!this.stopped) {
        this.audioSource.stop();
        this.stopped = true;
      }
    }
    if (value === "next") {
      let slice = Timing.slice_i(Timing.spec_frame(Timing.raw_samples(this.offset, this.rate), SPEC_STEP), this.window_size) + 1;
      this.offset = slice * WIN_SIZE;
      this.spectrogram_tmp = null;
      this.spectrogram = null;
      this.stopped = true;
      this.spectrogram_seek(slice, SPEC_WIN, SPEC_STEP, this.end);
      requestAnimationFrame(this.loop.bind(this));
    }
  }

  /**
  createRegionsFromClusters() {
    this.clusters.forEach(cluster => {
        console.log(cluster);        
        this.wavesurfer.addRegion({
          start: cluster.start / 48000,
          end:   cluster.stop  / 48000,
          color: Colors.COLORS[cluster.cluster_id],
          drag: false,
          data: {
            annotation: `cluster_${cluster.cluster_id}`, url: `https://wdp-ds.appspot.com/wdp-app/spectrogram/${this.cluster_name}/cluster_${cluster.cluster_id}.wav`
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
  } */

  ngAfterContentInit(): void {
    this.ctx = (<HTMLCanvasElement>this.canvas.nativeElement).getContext('2d');
    this.canvas.nativeElement.width = screen.width;
    this.canvas.nativeElement.height = SPEC_WIN / 2;
  }

  ngOnInit() {    
    this.getClusters();
    this.getEncodings();
    this.getAnnotations();
    this.audioContext = new (window['AudioContext'] || window['webkitAudioContext'])();
    let url = `wdp/wav/${this.encoding}`;
    this.loadSound(url);
  }
}
