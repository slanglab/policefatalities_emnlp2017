DNS=ec2-34-223-231-11.us-west-2.compute.amazonaws.com
rm test_processed/*
scp -i cnn.pem ubuntu@$DNS:/mnt/test_processed/* test_processed
