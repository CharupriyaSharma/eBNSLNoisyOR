for filename in CSV/*.csv; do
                
                col=$(head -1 $filename | sed 's/[^,]//g' | wc -c)
               
                dataset="${filename##*/}"
                dataset="${dataset%.*}"
                echo $dataset $col
        
                for((i=0; i<$col; i++ )); do   
                       
                        python3 NoisyORScore.py  $filename $i $dataset 150  &
           		wait
			echo $dataset $i 
                done
		
done
