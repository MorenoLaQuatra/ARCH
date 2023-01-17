python evaluate_hf_models.py --model facebook/wav2vec2-base --device cuda > results/wav2vec2-base.txt
python evaluate_hf_models.py --model facebook/wav2vec2-large-robust --device cuda > results/wav2vec2-large-robust.txt
python evaluate_hf_models.py --model facebook/wav2vec2-large-xlsr-53 --device cuda > results/wav2vec2-large-xlsr-53.txt

python evaluate_hf_models.py --model microsoft/wavlm-base --device cuda > results/wavlm-base.txt
python evaluate_hf_models.py --model microsoft/wavlm-base-plus --device cuda > results/wavlm-base-plus.txt
python evaluate_hf_models.py --model microsoft/wavlm-large --device cuda > results/wavlm-large.txt

python evaluate_hf_models.py --model facebook/hubert-base-ls960 --device cuda > results/hubert-base-ls960.txt
python evaluate_hf_models.py --model facebook/hubert-large-ll60k --device cuda > results/hubert-large-ll60k.txt
python evaluate_hf_models.py --model facebook/hubert-xlarge-ll60k --device cuda > results/hubert-xlarge-ll60k.txt

python evaluate_hf_models.py --model facebook/data2vec-audio-base --device cuda > results/data2vec-audio-base.txt
python evaluate_hf_models.py --model facebook/data2vec-audio-large --device cuda > results/data2vec-audio-large.txt
