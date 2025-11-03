python calculate_f1.py \
    --input results/Qwen3-8B/squad_without_pretrain/squad.json \
    --output results/Qwen3-8B/squad_without_pretrain/squad_f1_score.txt

python calculate_f1.py \
    --input results/Qwen3-8B/squad_without_pretrain/squad_no_metanet.json \
    --output results/Qwen3-8B/squad_without_pretrain/squad_no_metanet_f1_score.txt

python calculate_f1.py \
    --input results/Qwen3-8B/squad_without_pretrain/squad_only_question.json \
    --output results/Qwen3-8B/squad_without_pretrain/squad_only_question_f1_score.txt