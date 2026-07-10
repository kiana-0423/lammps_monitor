#!/usr/bin/env bash
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_ROOT=$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)
VERSIONS_FILE=${PHAL_VERSIONS_FILE:-"${SCRIPT_DIR}/versions.yaml"}
OUTPUT_DIR=${PHAL_CONTAINER_OUTPUT_DIR:-"${PROJECT_ROOT}/build/containers"}
IMAGE_NAMESPACE=${PHAL_IMAGE_NAMESPACE:-"phal"}
TARGETS="base md train cp2k"

yaml_value() {
    key=$1
    awk -F ':' -v key="${key}" '
        $1 == key {
            value = substr($0, index($0, ":") + 1)
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", value)
            gsub(/^"|"$/, "", value)
            print value
            exit
        }
    ' "${VERSIONS_FILE}"
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || {
        echo "required command not found: $1" >&2
        exit 2
    }
}

validate_target() {
    case " ${TARGETS} " in
        *" $1 "*) ;;
        *) echo "unknown target: $1 (expected: ${TARGETS})" >&2; exit 2 ;;
    esac
}

version_context() {
    UBUNTU_VERSION=$(yaml_value ubuntu)
    IMAGE_TAG=$(yaml_value image_tag)
    if [ -z "${UBUNTU_VERSION}" ] || [ "${UBUNTU_VERSION}" = "null" ]; then
        echo "containers/versions.yaml must pin ubuntu" >&2
        exit 2
    fi
    if [ -z "${IMAGE_TAG}" ] || [ "${IMAGE_TAG}" = "null" ] || [ "${IMAGE_TAG}" = "latest" ]; then
        echo "containers/versions.yaml must define a non-latest image_tag" >&2
        exit 2
    fi
    BASE_IMAGE="ubuntu:${UBUNTU_VERSION}"
}

image_ref() {
    echo "${IMAGE_NAMESPACE}/$1:${IMAGE_TAG}"
}

dockerfile_for() {
    echo "${SCRIPT_DIR}/Dockerfile.$1"
}

print_plan() {
    target=$1
    validate_target "${target}"
    echo "target=${target}"
    echo "dockerfile=$(dockerfile_for "${target}")"
    echo "base_image=${BASE_IMAGE}"
    echo "image=$(image_ref "${target}")"
    echo "versions=${VERSIONS_FILE}"
    echo "runtime_mount=${PROJECT_ROOT}/runtime:/runtime"
}

build_image() {
    target=$1
    validate_target "${target}"
    require_command docker
    docker build \
        --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
        --build-arg "IMAGE_TAG=${IMAGE_TAG}" \
        --file "$(dockerfile_for "${target}")" \
        --tag "$(image_ref "${target}")" \
        "${SCRIPT_DIR}"
}

tag_image() {
    target=$1
    destination_tag=$2
    validate_target "${target}"
    if [ -z "${destination_tag}" ] || [ "${destination_tag}" = "latest" ]; then
        echo "destination tag must be explicit and must not be latest" >&2
        exit 2
    fi
    require_command docker
    docker tag "$(image_ref "${target}")" "${IMAGE_NAMESPACE}/${target}:${destination_tag}"
}

export_oci() {
    target=$1
    validate_target "${target}"
    require_command docker
    mkdir -p "${OUTPUT_DIR}"
    archive="${OUTPUT_DIR}/phal-${target}-${IMAGE_TAG}.oci.tar"
    docker buildx build \
        --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
        --build-arg "IMAGE_TAG=${IMAGE_TAG}" \
        --file "$(dockerfile_for "${target}")" \
        --output "type=oci,dest=${archive}" \
        "${SCRIPT_DIR}"
    echo "wrote ${archive}"
}

build_sif() {
    target=$1
    validate_target "${target}"
    require_command apptainer
    mkdir -p "${OUTPUT_DIR}"
    source_uri=${PHAL_OCI_URI:-"docker-daemon://$(image_ref "${target}")"}
    output="${OUTPUT_DIR}/phal-${target}-${IMAGE_TAG}.sif"
    apptainer build "${output}" "${source_uri}"
    echo "wrote ${output}"
}

usage() {
    cat <<'EOF'
Usage: containers/build.sh COMMAND [TARGET] [ARG]

Commands:
  plan TARGET            Print the resolved Milestone 1 build plan.
  build TARGET           Build one minimal OCI/Docker image template.
  build-all              Build base, md, train, and cp2k templates.
  tag TARGET TAG         Add an explicit, non-latest Docker tag.
  oci TARGET             Export an OCI archive with Docker Buildx.
  sif TARGET             Convert an OCI image to SIF with Apptainer.
  help                    Show this message.

Environment overrides:
  PHAL_VERSIONS_FILE
  PHAL_CONTAINER_OUTPUT_DIR
  PHAL_IMAGE_NAMESPACE
  PHAL_OCI_URI            e.g. docker://registry.example/phal/md:m1

Milestone 1 Dockerfiles contain no package installation commands.
EOF
}

version_context
command_name=${1:-help}

case "${command_name}" in
    plan) print_plan "${2:?TARGET is required}" ;;
    build) build_image "${2:?TARGET is required}" ;;
    build-all)
        for target in ${TARGETS}; do
            build_image "${target}"
        done
        ;;
    tag) tag_image "${2:?TARGET is required}" "${3:?TAG is required}" ;;
    oci) export_oci "${2:?TARGET is required}" ;;
    sif) build_sif "${2:?TARGET is required}" ;;
    help|-h|--help) usage ;;
    *) usage >&2; exit 2 ;;
esac
