# model_name='meta-llama/Llama-3.2-1b'
# model_name='./fine_tuned_llama'
# model_name='./modified_llama'
model_name='./enhanced_model_target0_0.0005'
python eval.py \
        --model_args pretrained=${model_name},dtype=float16 \
        --tasks arc_challenge,arc_easy \
        --device cuda:1 \
        --batch_size 16 \
        --output_path outputs/${model_name}/search/${tasks}_restore_and_scale \
