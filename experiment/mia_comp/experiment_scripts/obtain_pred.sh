# modify this to set up directory:
DATA_DIR="data"

# This script is used to obtain the predictions of the attack on the target models
seed=0

if [ $# -eq 1 ]; then  # if the number of arguments is 1, the argument is the seed
    seed=$1
fi

echo "obtain_pred.sh seed = $seed"

data_dir="${DATA_DIR}/miae_standard_exp/target"

preds_dir="${DATA_DIR}/miae_standard_exp/preds_sd${seed}"

target_model_path="$data_dir/target_models"

prepare_path="${preds_dir}/prepare_sd${seed}"

mkdir -p "$preds_dir"


#datasets=("purchase100" "texas100")
 datasets=("cifar10" "cifar100" "cinic10")
# datasets=("cifar10")
 archs=("resnet56" "wrn32_4" "vgg16" "mobilenet")
# archs=("resnet56")
#archs=("mlp_for_texas_purchase")
 mias=("lira" "reference" "shokri" "losstraj" "calibration" "yeom" "aug")



for dataset in "${datasets[@]}"; do
  # if assign different num_epoch for different dataset
  if [ "$dataset" == "cifar10" ]; then
    num_epoch=60
  elif [ "$dataset" == "cifar100" ]; then
    num_epoch=100
  elif [ "$dataset" == "cinic10" ]; then
    num_epoch=60
  elif [ "$dataset" == "purchase100" ]; then
    num_epoch=30
  elif [ "$dataset" == "texas100" ]; then
    num_epoch=30
  else
    echo "Error: Unknown dataset $dataset"
    exit 1
fi


    for arch in "${archs[@]}"; do
      # for a given dataset and architecture, save the predictions
      mkdir -p "$preds_dir/$dataset/$arch"

      # prepare a directory for lira shadow models so lira and other attacks (RMIA, reference) could share the same shadow models
      lira_shadow_dir="$preds_dir/$dataset/$arch/lira_shadow_ckpts"
      mkdir -p "$preds_dir/$dataset/$arch/lira_shadow_ckpts"

        for mia in "${mias[@]}"; do
            result_dir="$preds_dir/$dataset/$arch/${mia}"
            # if the predictions are already saved, skip
            if [ -f "$result_dir/pred_$mia.npy" ]; then
                echo "Predictions already saved for $dataset $arch $mia at $result_dir/pred_$mia.npy"
                continue
            else
                echo "Predictions not saved for $dataset $arch $mia at $result_dir/pred_$mia.npy"
            fi

            # if the preparation directory is not empty, delete it
            if [ -d "$prepare_path" ] ; then
                rm -r "$prepare_path"
            fi

            mkdir -p "$result_dir"
            prepare_dir="$prepare_path"

            echo "Running $dataset $arch $mia"
            target_model_save_path="$target_model_path/$dataset/$arch"

            python3 obtain_pred.py \
            --dataset "$dataset"\
            --target_model "$arch"\
            --attack "$mia"\
            --result_path "$result_dir"\
            --seed "$seed"\
            --delete-files "True" \
            --preparation_path "$prepare_dir" \
            --data_aug "False"  \
            --target_model_path "$target_model_save_path" \
            --attack_epochs "$num_epoch" \
            --target_epochs "$num_epoch" \
            --data_path "$data_dir" \
            --device "cuda:0" \
            --dataset_file_root="$data_dir" \
            --lira_shadow_path "$lira_shadow_dir"

            rm -r "$prepare_path"
        done
    done
done
