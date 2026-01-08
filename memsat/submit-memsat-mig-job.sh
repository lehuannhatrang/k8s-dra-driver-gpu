#!/bin/bash
set -o nounset


memchoice() {
    # Numbers in GB (not GiB).
    python3 -c \
        'import random; print(random.choice([16, 17, 18, 18, 20, 21, 40, 42, 41, 40, 45, 70]))'
    }

gen_and_submit() {
    local PODIDX=$(printf "%04d" "$1")
    local MEM_GPU_REQUIRED_GB=$(memchoice)

    # local REQ_DEVICE="mig"
    # local REQ_DEVICE="gpu"
    local RCT_NAME="rct-mig-${MEM_GPU_REQUIRED_GB}gb"
    local REQ_NAME="req-mig-min-${MEM_GPU_REQUIRED_GB}gb"

    # Device capacity is announced in 'proper units', i.e. translation to GiB is
    # required. Using `bc` without `-l`: scale=0, so all divisions truncate to
    # integer results -- which is what we want here.
    local  MEM_GPU_REQUIRED_GIB=$(echo "($MEM_GPU_REQUIRED_GB)*10^9/1024^3" | bc)

    # 'cannot use generate name with apply'; so we roll our own.
    local RNDSFX=$(tr -dc a-z0-9 < /dev/urandom | head -c 7)
    local POD_NAME="pod-memsat-${PODIDX}-${REQ_DEVICE}-$(printf "%03d" "$MEM_GPU_REQUIRED_GB")gb-${RNDSFX}"

    # inject local variables into environment of just this `envsubst` child
    # process. note that if shell option errexit is set then 'already exists'
    # here terminates the function.
    RCT_NAME="$RCT_NAME" REQ_NAME="$REQ_NAME" REQ_DEVICE="$REQ_DEVICE" MEM_GPU_REQUIRED_GIB="$MEM_GPU_REQUIRED_GIB" \
        envsubst < gpu-rc.tmpl.yaml | kubectl apply -f - 2>&1 | grep -v 'already exists'

    # when called at large-ish concurrency, this might silently fail (absence of
    # 'created' in output however is revealing that problem)
    # output=$(envsubst < gpu-memsat.tmpl.yaml | kubectl apply -f -)
    local output=""

    while true; do
        output=$(
            POD_NAME="$POD_NAME" RCT_NAME="$RCT_NAME" MEM_GPU_REQUIRED_GIB="$MEM_GPU_REQUIRED_GIB" \
                envsubst < gpu-memsat.tmpl.yaml | kubectl apply -f -
            )

        if [[ "$output" != *"created"* ]]; then
            echo "throttling"
            sleep 2
            continue
        fi

        echo "submitted $POD_NAME"
        break
    done
}


for X in $(seq 1 $1); do
    gen_and_submit $X &
    if (( X % 10 == 0 )); then
        wait
    fi
done

wait

# Any out of error messages?
# Did anything fail?
# kubectl logs -l app=jp-mig-gen --prefix | grep OutOf
# kubectl get pods --namespace=default --field-selector=status.phase=Failed
# Some textual output reflecting the valuable compute work
# kubectl logs -l app=jp-mig-gen --prefix | grep duration | sort