generate_data() {
  local gpu=$(( $1 % 8 ))
  local display=":$gpu.0"
  local model_xml_path=$2
  local camera_distance_factor=$3
  local dataset_name=$4

  extra_args=""
  [ -n "$dataset_name" ] && extra_args="--dataset_name $dataset_name"

  CUDA_VISIBLE_DEVICES=$gpu DISPLAY=$display python generate_robot_data.py --model_xml_dir $model_xml_path --camera_distance_factor $camera_distance_factor $extra_args&
}

models=(
  "mujoco_menagerie/shadow_hand 0.5 shadow_hand"
  "mujoco_menagerie/unitree_h1 1.25 unitree_h1"
  "mujoco_menagerie/unitree_g1 1.25 unitree_g1"
  "mujoco_menagerie/unitree_go1 0.75 unitree_go1"
  "mujoco_menagerie/trossen_vx300s 0.75 trossen_vx300s"
  "mujoco_menagerie/ufactory_xarm7 1.0 ufactory_xarm7"
  "mujoco_menagerie/universal_robots_ur5e 1.0 universal_robots_ur5e"
  "mujoco_menagerie/universal_robots_ur5e_robotiq 1.0 universal_robots_ur5e_robotiq"
  "mujoco_menagerie/franka_emika_panda 1.0 franka_emika_panda"
  "mujoco_menagerie/unitree_go2 0.75 unitree_go2"
  "mujoco_menagerie/google_robot 1.5 google_robot"
)

for i in "${!models[@]}"; do
  generate_data $i ${models[$i]}
done
