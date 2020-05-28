#!/bin/bash
for filename in insurance/insurance_100.csv; do
                
                nvar=$(head -1 $filename | sed 's/[^,]//g' | wc -c)
               
                dataset="${filename##*/}"
                dataset="${dataset%.*}"
                echo $dataset $nvar
        
                for((i=0; i<21; i++ )); do   
                	for bf in 20; do 
			
				sbatch -o ./log_runtime/$dataset.$i.$bf -J $i.$dataset noisyOR.BIC.sh "$filename" "$i" "$dataset" "$bf"
			done
			
                done
		
done

