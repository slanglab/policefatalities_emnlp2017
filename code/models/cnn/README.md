### Overall notes

- This code is a modification of Yoon Kim's original CNN https://github.com/yoonkim/CNN_sentence
- There are many places where some of arguments to Kim's original functions are not used. There are also many places where extraneous data is passed as arguments. This legacy code could be much improved by refactoring, but this was not a priority for this project.
- We ran this code using ami-813110b1 on AWS, which comes with Theano 0.7, Python 2.7 and GPU drivers already configured
- We observed wide variability in runtimes on AWS boxes. On p2.8x large boxes the CNN sometimes ran in as little as 20 minutes per epoch and other times ran in as long as many hours hour per epoch. We do not understand this wide variability in performance and did not investigate. While running the experiments in the paper, the AWS machine ran in about 20 minutes per epoch on a g2.8xlarge machine.

### Details
- When making the AMI, create a .pem file called cnn.pem which is the key to connect to the AWS box
- fill in your DNS variable in `connect_aws.sh`, `send_to_gpu.sh` and `collect_from_gpu.sh` to the public DNS address visible in the ec2 console under the describion of your virtual machine. The USER variable in these scripts should be set to ubuntu
- run `./connect_aws.sh` to connect to your AWS machine.
- on your AMI, run the `ami_setup.sh` to create relevant directories. You will have to scp the `ami_setup.sh` to the amazon box when you create it.
- then run `./hard_cnn.sh 1` to run the cnn in the hard setting using GPU number 1. If you have multiple GPUs and want to run experiments in parallel you can also run `./hard_cnn.sh 2` to run on GPU number 2 etc.
- if you want to run the CNN in the soft setting use `./em_cnn.sh 1` to run on GPU 1.
- once the CNN completes, run `exporter.py` on the AWS box to create exported files.
- run `./collect_from_gpu.sh`. If you ran the CNN for E epochs using G gpus, collect_from_gpu will download E * G files labeled `test_processed/out$e-$g.json` where `$e` is the epoch number and `$g` is the gpu number.
- run `cat test_processed/out$e-$g.json | python eval/evaluation.py` to see the scores of gpu `$g` during epoch `$e`
