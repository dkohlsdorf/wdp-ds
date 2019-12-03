import { Spectrogram } from "wdp-ds-spectrogram";

// audio plaback
window.AudioContext = window.AudioContext || window.webkitAudioContext;
var context = new AudioContext();
var audio_source = null;
var audio_buffer = null;
var rate = null;

// double buffered spectrograms
var spectrogram = null;
var spectrogram_tmp = null;

// playback logic
var T = 1;
var end  = null;
var start = null;
var stopped = true;
var play_start = 0;

// spectrogram constants
const WIN  = 512;
const STEP = 256;
const VIZ_WIN = 8.5;

// spectrogram canvas
const canvas = document.getElementById('drawing');
const ctx = canvas.getContext('2d');
document.getElementById("drawing").width = screen.width;
document.getElementById("drawing").height = WIN / 2;

document.getElementById("next").onclick = function () { 
  if ((T + 1) * VIZ_WIN * rate < end) {
    spectrogram_seek();
    stopped = true;
    play_start = 0;
  }  
  window.requestAnimationFrame(loop);
}

document.getElementById("play").onclick = function () {
  if(stopped) { 
    setup_playback(play_start);
    stopped = false;
    window.requestAnimationFrame(loop);
    start = context.currentTime;
    const t_start = (play_start * STEP) / rate +  (T - 1) * VIZ_WIN;
    audio_source.start(0, t_start);    
  }
}

document.getElementById("pause").onclick = function () {  
  stopped = true;
  audio_source.stop();
  window.requestAnimationFrame(loop);
}

document.getElementById("drawing").onclick = function (e) {
  if(stopped) {  
    play_start = e.x;
    window.requestAnimationFrame(loop);
  }
}

function fmtMSS(s){return(s-(s%=60))/60+(9<s?':':':0')+s}

function ticks(offset) {
  const len = audio_buffer.getChannelData(0).length;
  const n = Math.min(len - 1, VIZ_WIN * rate);
  const LEN = (n * rate) / STEP - WIN;
  const TICK = LEN / VIZ_WIN;   
  const N_SEC = n / VIZ_WIN;
  for(var i = 0; i < VIZ_WIN; i++) {
    const sec = fmtMSS(Math.round((offset + i * N_SEC) / rate));
    ctx.fillStyle = "#FF0000";     
    ctx.fillText(`${sec}`, i * TICK / rate, 50);
  }
}

function setup_playback(offset) {
  audio_source = context.createBufferSource();
  audio_source.buffer = audio_buffer;
  audio_source.connect(context.destination);
}

function playback_head(t) { 
  var n = Math.min(play_start, end / STEP );
  if(!stopped) {
    n = ((t * rate) / STEP);
  }  
  ctx.beginPath();
  ctx.moveTo(n, 0);
  ctx.lineTo(n, 256);
  ctx.stroke();
  return n;
}

function spectrogram_seek() {
  if (spectrogram == null) {
    end = audio_buffer.getChannelData(0).length;
    spectrogram = Spectrogram.from_audio(audio_buffer.getChannelData(0), 0, Math.min(end - 1, VIZ_WIN * rate), WIN, STEP);
    setTimeout(function() {
      spectrogram_tmp = Spectrogram.from_audio(audio_buffer.getChannelData(0), T * VIZ_WIN * rate, (T + 1) * VIZ_WIN * rate, WIN, STEP);    
    }, 1);
  } else {
    spectrogram = spectrogram_tmp;  
    setTimeout(function() {
      spectrogram_tmp = Spectrogram.from_audio(audio_buffer.getChannelData(0), T * VIZ_WIN * rate, (T + 1) * VIZ_WIN * rate, WIN, STEP);    
    }, 1);
    T += 1;
  }
}

function plot(t) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  spectrogram.plot(ctx);     
  ctx.strokeStyle = "#FF0000";     
  ctx.lineWidth = 5;
  ticks((T - 1) * VIZ_WIN * rate);
  return playback_head(t);
}

function loop() {
  const t_start = (play_start * STEP) / rate;
  const t = context.currentTime - start + t_start;
  console.log(t, t_start + (T - 1) * VIZ_WIN);
  plot(t);
  if (t >= VIZ_WIN) {
    if ((T + 1) * VIZ_WIN * rate < end) {
      spectrogram_seek();
      play_start = 0;
    }
  } 
  if (t * rate >= end) {
    stopped = true;
  }
  if(!stopped) {
    setTimeout(function(){}, 2000);
    window.requestAnimationFrame(loop);
  } else {
    play_start = ((t * rate) / STEP);
  }
}

function loadSound(url) {
  var request = new XMLHttpRequest();
  request.open('GET', url, true);
  request.responseType = 'arraybuffer';
  request.onload = function() {      
    context.decodeAudioData(request.response, function(buffer) {
        audio_buffer = buffer;
        rate = audio_buffer.sampleRate;
        spectrogram_seek();
        window.requestAnimationFrame(loop);
    }, function(e) {console.log(e);});
  }
  request.send();
}

loadSound("demo2.wav");
