
export CUDA_VISIBLE_DEVICES=0
data=cifar10
arch='ConvNet'
batchsize=32

# Note: Depending on your setup It might make sense to first train all checkpoints and afterwards attack them
# For this simply comment out the corresponding line in the inner loop.

for steps_ in 1 5 10 20 50; # Sequential
do
{
    for aug_list in '21-13-3+7-4-15'; # Generally executed in parallel
    do
    {
        echo $aug_list
        #python3 -u benchmark/cifar_train.py --data=$data --arch=$arch --epochs="-1" --steps=$steps_ --batch_size=$batchsize --aug_list=$aug_list --mode=aug 2> /dev/null
        python3 -u benchmark/cifar_attack.py --data=$data --arch=$arch --epochs="-1" --steps=$steps_ --batch_size=$batchsize --aug_list=$aug_list --mode=aug --optim='inversed' 
    }&
    done
}
done