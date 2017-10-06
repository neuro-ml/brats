rm -Rf deep_pipe
rm -f configs/{train.config,val.config,test.config}
rm -Rf exp
rm -Rf val_submission
rm -Rf test_submission
docker rmi recreate_brats2017_sub 2>/dev/null