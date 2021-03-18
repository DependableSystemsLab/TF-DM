for ((j=1; j<=9; j++))
do
    cp sample.yaml d"$j"0.yaml
done

for ((j=1; j<=9; j++))
do
    sed -i 's/: 30/: '$j'0/g' d"$j"0.yaml
done
