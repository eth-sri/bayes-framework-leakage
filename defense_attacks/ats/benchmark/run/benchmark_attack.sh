
export CUDA_VISIBLE_DEVICES=0
data=cifar10
arch='ConvNet'

for epoch_ in 0;
do
{
    for aug_list in '' '21-13-3' '7-4-15' '21-13-3+7-4-15';
    do
    {
        echo $aug_list
        #python3 -u benchmark/cifar_train.py --data=$data --arch=$arch --epochs=$epoch_ --aug_list=$aug_list --mode=aug 2> /dev/null
        python3 -u benchmark/cifar_attack.py --data=$data --arch=$arch --epochs=$epoch_ --aug_list=$aug_list --mode=aug --optim='inversed'
    }&
    done
    wait
}
done