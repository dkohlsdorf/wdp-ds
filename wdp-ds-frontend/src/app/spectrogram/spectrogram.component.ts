import { Component, OnInit, AfterContentInit, OnDestroy, ViewChild, ElementRef, AfterViewInit, Injectable } from '@angular/core';
import { Http, Headers, RequestOptions, ResponseContentType } from '@angular/http'
import { map, concatMap, flatMap, switchMap } from "rxjs/operators";
import { ActivatedRoute } from '@angular/router';
import * as Timing from "./timing_utils";
import { Spectrogram } from "wdp-ds-spectrogram";

const SPEC_STEP = 256;
const SPEC_WIN = 512;
const WIN_SIZE = 8.5;
const N_TICKS = 10;

@Component({
  selector: 'app-spectrogram',
  templateUrl: './spectrogram.component.html',
  styleUrls: ['./spectrogram.component.css']
})
export class SpectrogramComponent implements OnInit, OnDestroy, AfterContentInit {
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

  constructor(private route: ActivatedRoute, private http: Http) {
    route.params.subscribe(
      params => {
        this.cluster_name = params['cluster_name'];
        this.asset_name = params['asset_name'];
      });
  }

  ngOnInit() {
    this.audioContext = new (window['AudioContext'] || window['webkitAudioContext'])();
    console.log(this.audioContext);
    let url = `wdp/asset_url/${this.cluster_name}/${this.asset_name}`;
    this.loadSound(url);
    this.get();
  }

  ngAfterContentInit(): void {
    this.ctx = (<HTMLCanvasElement>this.canvas.nativeElement).getContext('2d');
    this.canvas.nativeElement.width = screen.width;
    this.canvas.nativeElement.height = SPEC_WIN / 2;
  }

  ngOnDestroy() { }

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
}

