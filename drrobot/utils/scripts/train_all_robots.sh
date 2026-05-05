# Define a function to train a robot with different configurations
train_robot() {
    local robot_name=$1
    local device_id=$2

    CUDA_VISIBLE_DEVICES=$device_id DISPLAY=:0.0 python train.py \
        --experiment_name "${robot_name}_0707" \
        --dataset_path data/${robot_name}
}

# Train all robots with their respective configurations
# train_robot "google_robot" 2
# train_robot "universal_robots_ur5e_robotiq" 2
# train_robot "unitree_go1" 3
# train_robot "trossen_vx300s" 2
# train_robot "ufactory_xarm7" 3
# train_robot "unitree_go2" 4
# train_robot "franka_emika_panda" 5
# train_robot "shadow_hand" 6
train_robot "universal_robots_ur5e" 7
# train_robot "unitree_h1" 0
# train_robot "unitrlee_g1" 1