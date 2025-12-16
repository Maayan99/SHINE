python calculate_f1.py \
    --input checkpoints/results/Qwen3-8B/4_layer_1450/squad/squad.json \
    --output checkpoints/results/Qwen3-8B/4_layer_1450/squad/squad_f1_score.txt

python calculate_f1.py \
    --input checkpoints/results/Qwen3-8B/4_layer_1450/squad/squad_no_metanet.json \
    --output checkpoints/results/Qwen3-8B/4_layer_1450/squad/squad_no_metanet_f1_score.txt

python calculate_f1.py \
    --input checkpoints/results/Qwen3-8B/4_layer_1450/squad/squad_only_question.json \
    --output checkpoints/results/Qwen3-8B/4_layer_1450/squad/squad_only_question_f1_score.txt