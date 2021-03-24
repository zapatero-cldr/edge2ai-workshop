#!/bin/bash
# Works on MacOS only
set -o errexit
set -o nounset
BASE_DIR=$(cd $(dirname $0); pwd -L)
export NO_DOCKER_EXEC=1
source $BASE_DIR/common.sh

function cleanup() {
  kill $CHILD_PID
}

if [ $# != 2 ]; then
  echo "Syntax: $0 <namespace> <cluster_number>"
  show_namespaces
  exit 1
fi
NAMESPACE=$1
CLUSTER_ID=$2
load_env $NAMESPACE

PROXY_PORT=8157
PUBLIC_DNS=$(try_in_docker $NAMESPACE public_dns $CLUSTER_ID)

ssh -o StrictHostKeyChecking=no -i $TF_VAR_ssh_private_key -CND $PROXY_PORT $TF_VAR_ssh_username@$PUBLIC_DNS &
CHILD_PID=$!

$BASE_DIR/browse-cluster.sh $NAMESPACE $CLUSTER_ID $PROXY_PORT
