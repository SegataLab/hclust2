hclust2.py \
    -i HMP.species.txt \
    -o HMP.sqrt_scale.png \
    --skip_rows 1 \
    --ftop 50 \
    --f_dist_f correlation \
    --s_dist_f braycurtis \
    --cell_aspect_ratio 9 \
    -s --fperc 99 \
    --flabel_size 4 \
    --metadata_rows 2,3,4 \
    --legend_file HMP.sqrt_scale.legend.png \
    --max_flabel_len 100 \
    --metadata_height 0.075 \
    --minv 0.01 \
    --no_slabels \
    --dpi 300 \
    --slinkage complete