python evaluate_hf_models.py --model facebook/wav2vec2-base --device cuda > results/wav2vec2-base.txt
python evaluate_hf_models.py --model facebook/wav2vec2-large-robust --device cuda > results/wav2vec2-large-robust.txt
python evaluate_hf_models.py --model facebook/wav2vec2-large-xlsr-53 --device cuda > results/wav2vec2-large-xlsr-53.txt

python evaluate_hf_models.py --model microsoft/wavlm-base --device cuda > results/wavlm-base.txt
python evaluate_hf_models.py --model microsoft/wavlm-base-plus --device cuda > results/wavlm-base-plus.txt
python evaluate_hf_models.py --model microsoft/wavlm-large --device cuda > results/wavlm-large.txt
