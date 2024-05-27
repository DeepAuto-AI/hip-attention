mkdir -p submission

git checkout-index -a -f --prefix=./submission/

cd third_party

cd vllm-timber
git checkout-index -a -f --prefix=../../submission/third_party/vllm-timber/

cd ../LongBench-timber
git checkout-index -a -f --prefix=../../submission/third_party/LongBench-timber/

cd ../lm-evaluation-harness
git checkout-index -a -f --prefix=../../submission/third_party/lm-evaluation-harness/

cd ../../

# zip the submission
timestamp=$(date +'%m_%d_%Y')
zip -r "submission_${timestamp}.zip" submission/*

echo done