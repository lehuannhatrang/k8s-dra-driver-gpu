#/bin/bash
set -o nounset
set -o errexit


while true; do
    count="$(kubectl get pods -l app=jp-mig-gen --no-headers | grep Completed | wc -l)"
    echo "done: $count"
    if [ "$count" -ge "$1" ]; then
        break
    fi
    sleep 2
done
