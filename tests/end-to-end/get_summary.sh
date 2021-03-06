#!/bin/bash

#
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
source $utils_dir/utils.sh
echo ""
echo "Summarize peak statistics..."
echo ""
python $root_dir/peaksummary.py \
    --peakbw $out_dir/inference_latest/HSC_infer.peaks.bw \
    --trackbw $out_dir/inference_latest/HSC_infer.track.bw \
    --out_dir $out_dir/inference_latest \
    --prefix denoise.output.summary \
    --minlen 50

echo ""
echo "Verifying output model against expected model"
python $utils_dir/verify_diff.py --result_path $out_dir/inference_latest/denoise.output.summary.bed \
	                        --expected_path $expected_results_dir/inference_latest/denoise.output.summary.bed \
		                --format "general_diff"
check_status $? "Summary bed files do not match!"
