extern crate web_sys;

use wasm_bindgen::prelude::*;
use rustfft::num_complex::Complex;
use rustfft::FFTplanner;
use rustfft::num_traits::Zero;
use web_sys::{CanvasRenderingContext2d, ImageData};
use wasm_bindgen::Clamped;

#[cfg(feature = "wee_alloc")]
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

#[wasm_bindgen]
pub struct Spectrogram {
    win_size: usize,
    spectrogram: Vec<u8>
}

#[wasm_bindgen]
impl Spectrogram {

    pub fn plot(&mut self, ctx: &CanvasRenderingContext2d) -> Result<(), JsValue> {        
        let height = self.win_size / 2;
        let width  = self.spectrogram.len() / (4 * height);
        let data = ImageData::new_with_u8_clamped_array_and_sh(Clamped(&mut self.spectrogram), width as u32, height as u32)?;
        ctx.put_image_data(&data, 0.0, 0.0)    
    }

    pub fn from_audio(raw_audio: &[f32], win_size: usize, win_step: usize) -> Result<Spectrogram, JsValue> {            
        let hamming = Spectrogram::hamming(win_size);
        let samples: Vec<Complex<f32>> = raw_audio
            .iter()
            .map(|x| Complex::new(f32::from(*x), 0.0))
            .collect();
        let mut planner_dft = FFTplanner::new(false);
        let fft = planner_dft.plan_fft(win_size);
        let n = samples.len();
        let mut spectrogram: Vec<Vec<f32>> = Vec::new();
        for i in (win_size..n).step_by(win_step) {
            let start = i - win_size;
            let stop = i;
            let mut output: Vec<Complex<f32>> = vec![Complex::zero(); win_size];
            let mut input: Vec<Complex<f32>> = samples[start..stop]
                .iter()
                .enumerate()
                .map(|(i, x)| x * hamming[i])
                .collect();
            fft.process(&mut input[..], &mut output);
            let mut result: Vec<f32> = output
                .iter()
                .map(|complex| f32::sqrt(complex.norm_sqr()))               
                .collect();
            let mut min = std::f32::INFINITY;
            let mut max = std::f32::NEG_INFINITY;
            for i in 0 .. result.len() {
                if result[i] < min {
                    min = result[i];
                }
                if result[i] > max {
                    max = result[i];
                }                
            }
            for i in 0 .. result.len() {
                result[i] = (result[i] - min) / (max - min);
            }
            spectrogram.push(Vec::from(&result[win_size/2..win_size]));
        }
        let t = spectrogram.len();
        let d = win_size / 2;
        let mut result: Vec<u8> = Vec::new();
        for j in 0 .. d {
            for i in 0 .. t {
                let value = 255.0 - spectrogram[i][j] * 255.0;
                result.push(value as u8);                              
                result.push(value as u8);                              
                result.push(value as u8);                              
                result.push(255);                                  
            }
        }
        Ok(Spectrogram {
            win_size: win_size,
            spectrogram: result
        })    
    }
    
    pub fn hamming(len: usize) -> Vec<f32> {
        let mut hamming = Vec::new();
        for i in 0..len {
            hamming.push(0.54 + 0.46 * f32::cos((2.0 * std::f32::consts::PI * i as f32) / len as f32));
        }
        hamming
    }
}