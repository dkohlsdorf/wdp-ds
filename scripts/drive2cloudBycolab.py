# Execture this in colab
from google.colab import drive
from google.colab import auth

drive.mount('/content/drive')
auth.authenticate_user()

project_id = 'wdp-ds'
!gcloud config set project {project_id}
!gsutil -m cp -r /content/drive/My\ Drive/Wild\ Dolphin\ Project\ acoustics/2011\ wav\ extraction/SP\ mpeg/*.m4a gs://wdp-data/audio_files/2011/
