
python evaluate_hf_models.py --model facebook/wav2vec2-base --device cuda
python evaluate_hf_models.py --model facebook/wav2vec2-large-robust --device cuda

python evaluate_hf_models.py --model microsoft/wavlm-base --device cuda
python evaluate_hf_models.py --model microsoft/wavlm-base-plus --device cuda
python evaluate_hf_models.py --model microsoft/wavlm-large --device cuda

python evaluate_hf_models.py --model facebook/hubert-base-ls960 --device cuda
python evaluate_hf_models.py --model facebook/hubert-large-ll60k --device cuda
python evaluate_hf_models.py --model facebook/hubert-xlarge-ll60k --device cuda

python evaluate_hf_models.py --model facebook/data2vec-audio-base --device cuda
python evaluate_hf_models.py --model facebook/data2vec-audio-large --device cuda

python evaluate_hf_models.py --model facebook/wav2vec2-xls-r-300m --device cuda
python evaluate_hf_models.py --model facebook/wav2vec2-xls-r-1b --device cuda
python evaluate_hf_models.py --model facebook/wav2vec2-xls-r-2b --device cuda