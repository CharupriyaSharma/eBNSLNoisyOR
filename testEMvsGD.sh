for i in 8 7 6 5 4 3  
do
	x=1
	while [ $x -le 20 ]
	do
		
		python3 generateNoisyOR.py $i 1000 f
		cp "f_"$i"_1000.bif"  "f_"$i"_100.bif" 
		cp "f_"$i"_1000.bif"  "f_"$i"_500.bif"
		cp "f_"$i"_1000_noise"  "f_"$i"_100_noise" 
		cp "f_"$i"_1000_noise"  "f_"$i"_500_noise"
	        cat "f_"$i"_1000.csv"  | tail -100 > "f_"$i"_100.csv"
		cat "f_"$i"_1000.csv"  | tail -500 > "f_"$i"_500.csv"
		cat "f_"$i"_1000.vcsv"  | tail -100 > "f_"$i"_100.vcsv"
		cat "f_"$i"_1000.vcsv"  | tail -500 > "f_"$i"_500.vcsv"	
		for j in 100 500 1000 
		
		do
			dataset="f_"$i"_"$j
			#python3 generateNoisyOR.py $i $j f
			echo $dataset $x
			python3 bifFixer.py $dataset".bif"
			sed -i "" 's/1.0, 0.0/0.99999, 0.00001/g' $dataset".bifclean"
			sed -i "" 's/0.0, 1.0/0.00001, 0.99999/g' $dataset".bifclean"
			sed -i "" 's/1.0, 0.0/0.99999, 0.00001/g' $dataset".bifclean"
			sed -i "" 's/0.0, 1.0/0.00001, 0.99999/g' $dataset".bifclean"
			python3 score1.py $dataset".csv" 0 $dataset 3
			python3 NoisyORtoBIFv3.py $dataset null $i 1 $dataset".bifclean"
			cp $dataset"_enet_0" $dataset"_enet_gd"
			sed -i "" 's/1.0, 0.0/0.99999, 0.00001/g' $dataset"_enet_gd"
			sed -i "" 's/0.0, 1.0/0.00001, 0.99999/g' $dataset"_enet_gd"
			./n-or $dataset".vcsv" $dataset".csv" try.txt > $dataset"_0_3_noise"
			#cat $dataset"_0_3_noise"
			python3 reporterror.py f $i $j
			python3 NoisyORtoBIFv3.py $dataset null $i 1 $dataset".bifclean"
			cp $dataset"_enet_0" $dataset"_enet_em"
			sed -i "" 's/1.0, 0.0/0.99999, 0.00001/g' $dataset"_enet_em"
			sed -i "" 's/0.0, 1.0/0.00001, 0.99999/g' $dataset"_enet_em"
			python3 ../../Downloads/logreg2-3/ConditionalKLDivergenceClass.py  $dataset"_enet_gd" $dataset".bifclean" $dataset".csv" "CKL_GD_"$i"_"$j
			python3 ../../Downloads/logreg2-3/ConditionalKLDivergenceClass.py  $dataset"_enet_em" $dataset".bifclean" $dataset".csv" "CKL_EM_"$i"_"$j
		done
		x=$(( $x + 1 ))
	
	done
done
