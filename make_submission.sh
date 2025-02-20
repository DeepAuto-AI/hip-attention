mkdir -p submission

git checkout-index -a -f --prefix=./submission/

cd third_party

cd vllm
git checkout-index -a -f --prefix=../../submission/third_party/vllm/

cd ../sglang
git checkout-index -a -f --prefix=../../submission/third_party/sglang/

cd ../RULER-hip
git checkout-index -a -f --prefix=../../submission/third_party/RULER-hip/

cd ../LongBench-hip
git checkout-index -a -f --prefix=../../submission/third_party/LongBench-hip/

cd ../hip-jina
git checkout-index -a -f --prefix=../../submission/third_party/hip-h2o/

cd ../quick-extend
git checkout-index -a -f --prefix=../../submission/third_party/hip-training/

cd ../../

rm submission/third_party/hip-h2o/notebook/h2o_5_6_cmp.ipynb

# zip the submission
timestamp=$(date +'%m_%d_%Y')
zip -r "submission_${timestamp}.zip" submission/*

echo done
