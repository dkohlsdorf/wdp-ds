import { Spectrogram } from "wdp-ds-spectrogram";

function playback_sec(t, started) {
  return t - started;
}

function raw_samples(t_sec, rate) {
  return t_sec * rate;
} 

function spec_frame(raw_t, spec_step) {
  return raw_t / spec_step;
}

function slice_i(spec_t, window_size) {
  return Math.floor(spec_t / window_size);
}

function slice_t(spec_t, window_size) {
  return spec_t % window_size;
}

function sec2frames(seconds, rate, spec_step) {
  return spec_frame(raw_samples(seconds, rate), spec_step)
}

function slice2raw(slice, window_size, spec_step, end) {
  return Math.min(end, slice * window_size * spec_step);
}

// web audio
window.AudioContext = window.AudioContext || window.webkitAudioContext;
var context = new AudioContext();
var audio_source = null;
var audio_buffer = null;

// spec params
const spec_step = 256;
const spec_win  = 512;
const window_size_sec = 8.5;

// animation params
const canvas = document.getElementById('drawing');
const ctx = canvas.getContext('2d');
const n_ticks = 10;
document.getElementById("drawing").width = screen.width;
document.getElementById("drawing").height = spec_win / 2;
var rate = null;
var end  = null;
var spectrogram = null;
var spectrogram_tmp = null;
var stopped = true;
var window_size = null;
var started = 0;
var last_slice = 0;
var offset = 0;

const callback_func = function(sec) {console.log(sec)};

// Animations Controller
document.getElementById("play").onclick = function () {
  if(stopped) { 
    setup_playback();
    stopped    = false;
    last_slice = slice_i(spec_frame(raw_samples(offset, rate), spec_step), window_size);
    started    = context.currentTime;
    audio_source.start(0, offset);  
    animate(callback_func);
  }
}

function setup_playback() {
  audio_source = context.createBufferSource();
  audio_source.buffer = audio_buffer;
  audio_source.connect(context.destination);
}

document.getElementById("pause").onclick = function () {  
  if (!stopped) {
    audio_source.stop();
    stopped = true;
  }
}

document.getElementById("next").onclick = function () {   
  const slice = slice_i(spec_frame(raw_samples(offset, rate), spec_step), window_size) + 1;  
  offset = slice * window_size_sec;
  spectrogram_tmp = null;
  spectrogram = null;
  stopped = true;  
  spectrogram_seek(slice, spec_win, spec_step, end);
  animate(callback_func);
}

// Animations Visualizer
function spectrogram_seek(slice_i, fft_win, fft_step, end) {    
  const start  = slice2raw(slice_i, window_size, spec_step, end);
  const mid    = slice2raw(slice_i + 1, window_size, spec_step, end); 
  const stop   = slice2raw(slice_i + 2, window_size, spec_step, end);
  if (spectrogram == null) {    
    spectrogram = Spectrogram.from_audio(audio_buffer.getChannelData(0).slice(start, mid), fft_win, fft_step);
  } else {
    spectrogram = spectrogram_tmp;
  }
  if (stop > mid) {
    setTimeout(function() {
      spectrogram_tmp = Spectrogram.from_audio(audio_buffer.getChannelData(0).slice(mid, stop), fft_win, fft_step);    
    }, 1);
  }
}

function fmtMSS(s){return(s-(s%=60))/60+(9<s?':':':0')+s}

function ticks(slice_i) {
  const start_raw = slice2raw(slice_i, window_size, spec_step, end);
  const start  = spec_frame(start_raw, spec_step);
  const stop   = spec_frame(slice2raw(slice_i + 1, window_size, spec_step, end), spec_step); 
  const start_sec = start_raw / rate;
  const tick  = (stop - start)/n_ticks;   
  const n_sec = (tick * spec_step) / (rate);
  for(var i = 0; i < n_ticks; i++) {
    const sec = fmtMSS(Math.round(start_sec + i * n_sec));
    ctx.fillStyle = "#FF0000";     
    ctx.fillText(`${sec}`, i * tick, 50);
  }
  const n_bins = spec_win / (2 * n_ticks);
  const freq_bin = (rate / 2) / (spec_win / 2);
  for(var i = 0; i < n_ticks; i++) {
    if( 256 - i * n_bins > 55) {
      ctx.fillStyle = "#FF0000";         
      ctx.fillText(`${i * freq_bin * n_bins}`, 10, 256 - i * n_bins);
    }
  }
}

function plot(slice, in_slice) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  spectrogram.plot(ctx);     
  ctx.strokeStyle = "#FF0000";     
  ctx.lineWidth = 5;
  playback_head(in_slice);
  ticks(slice);
}

function playback_head(in_slice) { 
  ctx.beginPath();
  ctx.moveTo(in_slice, 0);
  ctx.lineTo(in_slice, 256);
  ctx.stroke();
}

function animate(callback) {
  function loop() {
    var playback  = playback_sec(context.currentTime, started);
    if (stopped) {
      var playback = 0;
    }
    let seconds   = playback + offset;
    let samples   = raw_samples(seconds, rate);
    let frames    = spec_frame(samples, spec_step);
    let slice     = slice_i(frames, window_size);
    let in_slice  = slice_t(frames, window_size);
    callback(seconds)
    if(!stopped && last_slice != slice) {
      spectrogram_seek(slice, spec_win, spec_step, end);
      last_slice = slice;
    }
    plot(slice, in_slice);
    if (seconds < end && !stopped) {
      requestAnimationFrame(loop);
    }
  }
  requestAnimationFrame(loop)
}

function loadSound(url) {
  var request = new XMLHttpRequest();
  request.open('GET', url, true);
  request.responseType = 'arraybuffer';
  request.onload = function() {      
    context.decodeAudioData(request.response, function(buffer) {    
        audio_buffer = buffer;
        end  = audio_buffer.getChannelData(0).length;
        rate = audio_buffer.sampleRate;
        window_size = sec2frames(window_size_sec, rate, spec_step);
        spectrogram_seek(0, spec_win, spec_step, end);
        animate(callback_func);
    }, function(e) {console.log(e);});
  }
  request.send();
}

loadSound("demo.wav");
