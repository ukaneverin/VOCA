Example running pipeline:

python generate_task_map.py --dataset $YourDataset$ --output_path $output_path$
python train.py --cv $CrossValidationFold$
python detection_validation.py
python detection_whole_slide.py --id $slide_id$

