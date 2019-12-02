import { Spectrogram } from "wdp-ds-spectrogram";

window.AudioContext = window.AudioContext || window.webkitAudioContext;
var context = new AudioContext();
var source = null;
var audio_buffer = null;
var spectrogram = null;
var spectrogram_tmp = null;
var end = null;
var rate = null;
var T = 1;
var start = null;
var stopped = true;
var play_start = 0;

const WIN  = 512;
const STEP = 256;
const VIZ_WIN = 8.5;

document.getElementById("drawing").width = screen.width;
document.getElementById("drawing").height = WIN / 2;
const canvas = document.getElementById('drawing');
const ctx = canvas.getContext('2d');

document.getElementById("next").onclick = function () { 
  if ((T + 1) * VIZ_WIN * rate < end) {
    spectrogram_seek();
    stopped = true;
  }
  window.requestAnimationFrame(loop);
}

document.getElementById("play").onclick = function () {
  if(stopped) { 
    setup_playback(play_start);
    stopped = false;
    window.requestAnimationFrame(loop);
    start = context.currentTime;
    source.start();    
  }
}

document.getElementById("stop").onclick = function () {  
  T = 1;
  stopped = true;
  source.stop();
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
  source = context.createBufferSource();
  source.buffer = audio_buffer;
  source.connect(context.destination);
  // seek to offset
}

function playback_head(t) { 
  var n = Math.min(play_start, end / STEP );
  if(!stopped) {
    n = ((t * rate) / STEP) - (((T - 1) * VIZ_WIN * rate) / STEP);
  }
  ctx.beginPath();
  ctx.moveTo(n, 0);
  ctx.lineTo(n, 256);
  ctx.stroke();
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
  playback_head(t);
}

function loop() {
  const t_start = (play_start * STEP) / rate + (T - 1) * VIZ_WIN;
  console.log(t_start);
  const t = context.currentTime - start + t_start;
  plot(t);
  setTimeout(function(){}, 2000);
  if (t * rate >= T * VIZ_WIN * rate) {
    if ((T + 1) * VIZ_WIN * rate < end) {
      spectrogram_seek();
    }
  } 
  if (t * rate >= end) {
    stopped = true;
  }
  if(!stopped) {
    window.requestAnimationFrame(loop);
  } else {
    play_start = ((t * rate) / STEP) - (((T - 1) * VIZ_WIN * rate) / STEP);
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

loadSound("222.wav");
