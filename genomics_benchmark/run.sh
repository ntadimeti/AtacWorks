#Set variables
testintervals=get_intervals/genome_intervals.bed
model=epoch23_checkpoint.pth.tar
label_file=bw2h5/Erythro-15.1000000.2.wg.50000.5000.h5
outdir=benchmark_dir

mkdir $outdir
#Generate intervals
mkdir get_intervals; python scripts/get_intervals.py --sizes /home/ntadimeti/AtacWorks/example/reference/hg19.auto.sizes --intervalsize 50000 --out_dir ./get_intervals --wg

#convert to h5
mkdir ./bw2h5; python scripts/bw2h5.py --noisybw ./Erythro-15.1000000.2.cutsites.smoothed.200.bw --intervals ./get_intervals/genome_intervals.bed --pad 5000 --out_dir ./bw2h5/ --nolabel --prefix Erythro-15.1000000.2.wg.50000.5000

#Run inference
start=`date +%s`
python $root_dir/main.py infer --infer_files $label_file --weights_path $model \
--out_home $out_dir --label inference --result_fname output.h5 --model resnet \
--nblocks 5 --nfilt 15 --width 51 --dil 8 --nblocks_cla 2 --nfilt_cla 15 --width_cla 51 --dil_cla 8 \
--task both --pad 5000  --bs 512 --num_workers <num_workers> --batches_per_worker 16 \
--sizes_file <path to sizes file> --intervals_file $testintervals
end=`date +%s`
runtime=$((end-start))

echo "Total run time for inference: $runtime"
