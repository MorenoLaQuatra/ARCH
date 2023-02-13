mkdir audio_datasets
cd audio_datasets

# -------------------------- ESC-50 --------------------------
echo "Downloading ESC-50 dataset..."
wget https://github.com/karoldvl/ESC-50/archive/master.zip
unzip master.zip
rm master.zip
mv ESC-50-master esc50

# --------------------------UrbanSound8K --------------------------
echo "Downloading UrbanSound8K dataset..."
wget https://zenodo.org/record/1203745/files/UrbanSound8K.tar.gz
tar -xvzf UrbanSound8K.tar.gz
rm UrbanSound8K.tar.gz

# -------------------------- FSD50K --------------------------
echo "Downloading FSD50K dataset..."
mkdir fsd50k
cd fsd50k
wget https://zenodo.org/record/4060432/files/FSD50K.dev_audio.z01
wget https://zenodo.org/record/4060432/files/FSD50K.dev_audio.z02
wget https://zenodo.org/record/4060432/files/FSD50K.dev_audio.z03
wget https://zenodo.org/record/4060432/files/FSD50K.dev_audio.z04
wget https://zenodo.org/record/4060432/files/FSD50K.dev_audio.z05
wget https://zenodo.org/record/4060432/files/FSD50K.dev_audio.zip

wget https://zenodo.org/record/4060432/files/FSD50K.eval_audio.z01
wget https://zenodo.org/record/4060432/files/FSD50K.eval_audio.zip

wget https://zenodo.org/record/4060432/files/FSD50K.ground_truth.zip
wget https://zenodo.org/record/4060432/files/FSD50K.metadata.zip


7z x FSD50K.dev_audio.zip
rm FSD50K.dev_audio.z*

7z x FSD50K.eval_audio.zip
rm FSD50K.eval_audio.z*

unzip FSD50K.ground_truth.zip
unzip FSD50K.metadata.zip

rm FSD50K.ground_truth.zip
rm FSD50K.metadata.zip
cd ..

# -------------------------- VIVAE --------------------------
echo "Downloading VIVAE dataset..."
wget https://zenodo.org/record/4066235/files/VIVAE.zip
unzip VIVAE.zip
rm VIVAE.zip


# -------------------------- FMA-small --------------------------
echo "Downloading FMA-small dataset..."
mkdir fma_small
cd fma_small
wget https://os.unil.cloud.switch.ch/fma/fma_small.zip
wget https://os.unil.cloud.switch.ch/fma/fma_metadata.zip
7z x fma_small.zip
7z x fma_metadata.zip
rm fma_small.zip
rm fma_metadata.zip
cd ..

# -------------------------- MagnaTagATune --------------------------
echo "Downloading MagnaTagATune dataset..."
mkdir magnatagatune
wget https://mirg.city.ac.uk/datasets/magnatagatune/clip_info_final.csv
wget https://mirg.city.ac.uk/datasets/magnatagatune/mp3.zip.001
wget https://mirg.city.ac.uk/datasets/magnatagatune/mp3.zip.002
wget https://mirg.city.ac.uk/datasets/magnatagatune/mp3.zip.003
wget https://mirg.city.ac.uk/datasets/magnatagatune/annotations_final.csv
cat mp3.zip.* > mp3.zip
unzip mp3.zip
rm mp3.zip
rm mp3.zip.*

wget https://raw.githubusercontent.com/jordipons/musicnn-training/master/data/index/mtt/test_gt_mtt.tsv # download the test split
wget https://raw.githubusercontent.com/jordipons/musicnn-training/master/data/index/mtt/train_gt_mtt.tsv # download the train split
wget https://raw.githubusercontent.com/jordipons/musicnn-training/master/data/index/mtt/val_gt_mtt.tsv # download the validation split
cd ..


# -------------------------- IRMAS --------------------------
echo "Downloading IRMAS dataset..."
mkdir irmas
cd irmas

wget https://zenodo.org/record/1290750/files/IRMAS-TrainingData.zip
wget https://zenodo.org/record/1290750/files/IRMAS-TestingData-Part1.zip
wget https://zenodo.org/record/1290750/files/IRMAS-TestingData-Part2.zip
wget https://zenodo.org/record/1290750/files/IRMAS-TestingData-Part3.zip

unzip IRMAS-TrainingData.zip
unzip IRMAS-TestingData-Part1.zip
unzip IRMAS-TestingData-Part2.zip
unzip IRMAS-TestingData-Part3.zip

rm IRMAS-TrainingData.zip
rm IRMAS-TestingData-Part1.zip
rm IRMAS-TestingData-Part2.zip
rm IRMAS-TestingData-Part3.zip
cd ..

# -------------------------- Medley-Solos-DB --------------------------
echo "Downloading MedleyDB dataset..."
mkdir medleydb
cd medleydb
wget https://zenodo.org/record/3464194/files/Medley-solos-DB.tar.gz
wget https://zenodo.org/record/2582103/files/Medley-solos-DB_metadata.csv
mkdir audio
cd audio
mv ../Medley-solos-DB.tar.gz .
tar -xvzf Medley-solos-DB.tar.gz
rm Medley-solos-DB.tar.gz
cd ..
cd ..


# -------------------------- RAVDESS --------------------------
echo "Downloading RAVDESS dataset..."
wget https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip
mkdir ravdess
unzip Audio_Speech_Actors_01-24.zip -d ravdess
rm Audio_Speech_Actors_01-24.zip
mv Audio_Speech_Actors_01-24 ravdess

# -------------------------- SLURP --------------------------
echo "Downloading SLURP dataset..."
mkdir slurp
cd slurp
wget https://zenodo.org/record/4274930/files/slurp_real.tar.gz
wget https://zenodo.org/record/4274930/files/slurp_synth.tar.gz
tar -xvzf slurp_real.tar.gz
tar -xvzf slurp_synth.tar.gz
rm slurp_real.tar.gz
rm slurp_synth.tar.gz
wget https://raw.githubusercontent.com/pswietojanski/slurp/master/dataset/slurp/devel.jsonl
wget https://raw.githubusercontent.com/pswietojanski/slurp/master/dataset/slurp/test.jsonl
wget https://raw.githubusercontent.com/pswietojanski/slurp/master/dataset/slurp/train.jsonl
cd ..

# -------------------------- EMOVO --------------------------
echo "Downloading EMOVO dataset..."
mkdir emovo
cd emovo
gdown 1SUtaKeA-LYnKaD3qv87Y5wYgihJiNJAo
unzip emovo.zip
rm emovo.zip
cd ..

echo "Done!"
# print the current directory
echo "Current directory:"
pwd