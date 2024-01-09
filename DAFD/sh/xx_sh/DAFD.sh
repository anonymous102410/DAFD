GPU_ID=0
data_dir=pathtodata

task=A2W
source_domain=amazon
target_domain=webcam
python main.py --config DAFD/DAFD.yaml --data_dir $data_dir --src_domain $source_domain --tgt_domain $target_domain --task $task
