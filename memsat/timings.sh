#!/bin/bash

LOGFILE="/tmp/dra-driver-dbg_plugins_$(date -u +'%Y-%m-%dT%H%M%SZ').log"

kubectl logs \
    -n nvidia-dra-driver-gpu \
    -l nvidia-dra-driver-gpu-component=kubelet-plugin \
    --prefix --all-containers --timestamps --tail=-1 > "${LOGFILE}"

#for metric in "t_unprep_total" "t_unprep_lock_acq" "t_unprep"


# cmd/gpu-kubelet-plugin/device_state.go:t_prep_state_lock_acq %.3f s",
# cmd/gpu-kubelet-plugin/device_state.go:t_prep_gcp %.3f s",
# cmd/gpu-kubelet-plugin/device_state.go:t_prep_gcp %.3f s",
# cmd/gpu-kubelet-plugin/device_state.go:t_prep_core %.3f s (claim %s)", time.Since(tprep0).Seconds(),
# cmd/gpu-kubelet-plugin/device_state.go:t_prep_create_mig_dev %.3f s (claim %s)", time.Since(tcmig0).Seconds(),
# cmd/gpu-kubelet-plugin/driver.go:t_prep_lock_acq %.3f s",
# cmd/gpu-kubelet-plugin/driver.go:t_prep %.3f s (claim %s)", time.Since(tprep0).Seconds(),
# cmd/gpu-kubelet-plugin/driver.go:t_prep_total %.3f s (claim %s)", time.Since(t0).Seconds(),
# cmd/gpu-kubelet-plugin/nvlib.go:t_prep_create_mig_dev_preamble %.3f s",
# cmd/gpu-kubelet-plugin/nvlib.go:t_prep_create_mig_dev_cigi %.3f s",
# cmd/gpu-kubelet-plugin/nvlib.go:t_prep_create_mig_dev_walkdevs %.3f s",

# "t_prep_lock_acq"

for metric in "t_prep_state_lock_acq" "t_prep_core" "t_prep_ucp" "t_prep_ucp2" \
        "t_prep_gcp" "t_prep_ccsf" "t_gen_write_cdi_spec" \
        "t_prep_create_mig_dev_preamble" "t_prep_create_mig_dev_new_dev" \
        "t_prep_create_mig_dev_init" "t_prep_create_mig_dev_get_dev_handle" \
        "t_prep_create_mig_dev_check_mig_enabled" \
        "t_enable_mig" \
        "t_prep_create_mig_dev_cigi" \
        "t_cdi_get_specs_for_uuid" \
        "t_cdi_get_common_edits" \
        "t_prep_total" "t_prep" \

do
    echo "## $metric"
    cat "$LOGFILE" | grep -Po "$metric \K[0-9]+\.[0-9]+" | uplot hist --nbins 7 --xlabel "count" --ylabel "time(s)"
    echo -e "\n\n"
done
