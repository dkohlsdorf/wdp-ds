
/**
 * Compute the file position in seconds from playback timestamp and a start timestamp
 *
 * @param t current clock
 * @param started clock when started
 */
export function playback_sec(t:number, started:number): number {
    return t - started;
}

/**
 * Compute the sample position given a playback time
 * 
 * @param t_sec playback time
 * @param rate sample rate of audio file
 */
export function raw_samples(t_sec: number, rate: number): number {
    return t_sec * rate;
}

/**
 * Compute the spectrogram frame from a sample position
 * 
 * @param raw_t the sample position
 * @param spec_step the step size used for spectrogram computation
 */
export function spec_frame(raw_t: number, spec_step: number): number {
    return raw_t / spec_step;
}

/**
 * Index a spectrogram slice given a spectrogram sample
 * 
 * @param spec_t spectrogram sample
 * @param window_size size of a spectrogram slice
 */
export function slice_i(spec_t: number, window_size: number): number {
    return Math.floor(spec_t / window_size);
}

/**
 * Index in a spectrogram slice
 * 
 * @param spec_t spectrogram sample
 * @param window_size size of a spectrogram slice
 */
export function slice_t(spec_t: number, window_size: number): number {
    return spec_t % window_size;
}

/**
 * Compute the spectrogram frame for a given timestamp
 * 
 * @param seconds timestamp
 * @param rate sample rate of audio
 * @param spec_step the step size used for spectrogram computation
 */
export function sec2frames(seconds: number, rate: number, spec_step: number): number {
    return spec_frame(raw_samples(seconds, rate), spec_step)
}

/**
 * Compute the start timestamp of a spectrogram slice
 * 
 * @param slice the id of the spectrogram slice
 * @param window_size the size of a spectrogram slice
 * @param spec_step the spectrogram step size
 * @param end length of the file
 */
export function slice2raw(slice: number, window_size: number, spec_step: number, end: number): number {
    return Math.min(end, slice * window_size * spec_step);
}

export function fmtMSS(s: number){return(s-(s%=60))/60+(9<s?':':':0')+s}
