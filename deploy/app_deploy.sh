cd wdp-ds-spectrogram/
wasm-pack build --release -t bundler
cd ../wdp-ds-frontend/
ng build #--prod
cd ../app
gcloud app deploy
cd ..