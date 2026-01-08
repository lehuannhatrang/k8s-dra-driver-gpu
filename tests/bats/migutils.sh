


# show_all_mig_devices_all_nodes() {
#   for node in $(kubectl get nodes -o=jsonpath='{.items[*].metadata.name}'); do
#     nvmm "$node" nvidia-smi -L
#     #mig -lgi
#   done
# }

# show_processes_on_migs_all_nodes() {
#   for node in $(kubectl get nodes -o=jsonpath='{.items[*].metadata.name}'); do
#     nvmm "$node" sh -c 'nvidia-smi mig -lgi | grep MIG; nvidia-smi | grep -A10 Processes | grep -E '[0-9]+''
#   done
# }

# show_mig_mode_all_gpus_all_nodes() {
#   for node in $(kubectl get nodes -o jsonpath='{.items[*].metadata.name}'); do
#      nvmm "$node" sh -c 'nvidia-smi --query-gpu=index,mig.mode.current --format=csv'
#   done
# }


# show_utilization_all_gpus_all_nodes() {
#   for node in $(kubectl get nodes -o jsonpath='{.items[*].metadata.name}'); do
#      nvmm "$node" nvidia-smi --query-gpu=memory.used,temperature.gpu --format=csv
#   done
# }

show_all_mig_devices_all_nodes() {
  for node in $(cat "$MU_NODES_NAMES_FILE_PATH"); do
    nvmm "$node" nvidia-smi -L
  done
}

show_processes_on_migs_all_nodes() {
  for node in $(kubectl get nodes -o=jsonpath='{.items[*].metadata.name}'); do
    nvmm "$node" sh -c 'nvidia-smi mig -lgi | grep MIG; nvidia-smi | grep -A10 Processes | grep -E '[0-9]+''
  done
}

show_mig_mode_all_gpus_all_nodes() {
  for node in $(kubectl get nodes -o jsonpath='{.items[*].metadata.name}'); do
     nvmm "$node" sh -c 'nvidia-smi --query-gpu=index,mig.mode.current --format=csv'
  done
}


show_utilization_all_gpus_all_nodes() {
  for node in $(kubectl get nodes -o jsonpath='{.items[*].metadata.name}'); do
     nvmm "$node" nvidia-smi --query-gpu=memory.used,temperature.gpu --format=csv
  done
}


if [ -z "$1" ]; then
  echo "Usage: argument missing"
  return 1
fi

# show_all_mig_devices_all_nodes
# Run one of the commands
$1