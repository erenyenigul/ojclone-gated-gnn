for d in source/OJClone/*; do 
    mkdir compiled/$d
    for f in $d/*; do
        clang -S -emit-llvm $f -o compiled/$f.ll
    done
done

