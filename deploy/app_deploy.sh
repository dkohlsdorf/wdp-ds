cd wdp-ds-spectrogram/
wasm-pack build --release
cd ../wdp-ds-frontend/
ng build --prod
cd ../app
gcloud app deploy
cd ..