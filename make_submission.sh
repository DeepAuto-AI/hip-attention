mkdir -p submission

git checkout-index -a -f --prefix=./submission/

cd third_party

cd vllm-hip
git checkout-index -a -f --prefix=../../submission/third_party/vllm-hip/

cd ../LongBench-hip
git checkout-index -a -f --prefix=../../submission/third_party/LongBench-hip/

cd ../lm-evaluation-harness
git checkout-index -a -f --prefix=../../submission/third_party/lm-evaluation-harness/

cd ../../

# zip the submission
timestamp=$(date +'%m_%d_%Y')
zip -r "submission_${timestamp}.zip" submission/*

echo done