#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_ROOT=$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)
VERSIONS_FILE=${PHAL_VERSIONS_FILE:-"${SCRIPT_DIR}/versions.yaml"}
OUTPUT_DIR=${PHAL_CONTAINER_OUTPUT_DIR:-"${PROJECT_ROOT}/build/containers"}
EVIDENCE_DIR=${PHAL_EVIDENCE_DIR:-"${PROJECT_ROOT}/build/evidence"}
IMAGE_NAMESPACE=${PHAL_IMAGE_NAMESPACE:-"phal"}
REGISTRY=${PHAL_REGISTRY:-}
TARGETS="base md train train-probe cp2k"
SCAFFOLD_TARGETS="base md train cp2k"

yaml_path_value() {
    path=$1
    awk -v target="${path}" '
        function trim(value) {
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", value)
            gsub(/^"|"$/, "", value)
            return value
        }
        /^[[:space:]]*(#|$)/ { next }
        {
            match($0, /^[ ]*/)
            level = int(RLENGTH / 2) + 1
            content = substr($0, RLENGTH + 1)
            colon = index(content, ":")
            if (colon == 0) next
            key = trim(substr(content, 1, colon - 1))
            value = trim(substr(content, colon + 1))
            stack[level] = key
            for (i = level + 1; i <= 12; i++) delete stack[i]
            current = stack[1]
            for (i = 2; i <= level; i++) current = current "." stack[i]
            if (current == target && value != "") {
                print value
                exit
            }
        }
    ' "${VERSIONS_FILE}"
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || {
        echo "required command not found: $1" >&2
        exit 2
    }
}

require_value() {
    name=$1
    value=$2
    if [[ -z "${value}" || "${value}" == "null" ]]; then
        echo "required value is not configured: ${name}" >&2
        exit 2
    fi
}

reject_floating_ref() {
    name=$1
    value=$2
    case "${value}" in
        latest|stable|main|*:latest|*:stable|*:main)
            echo "${name} must use a fixed tag or digest, got: ${value}" >&2
            exit 2
            ;;
    esac
}

validate_target() {
    case " ${TARGETS} " in
        *" $1 "*) ;;
        *) echo "unknown target: $1 (expected: ${TARGETS})" >&2; exit 2 ;;
    esac
}

version_context() {
    target=$1
    PLATFORM=${PHAL_PLATFORM:-$(yaml_path_value platform.default_oci_platform)}
    if [[ "${target}" == "train-probe" ]]; then
        candidate_base_image=$(yaml_path_value candidates.train_stack_a.base_image)
        BASE_IMAGE_DIGEST=$(yaml_path_value candidates.train_stack_a.base_image_digest)
        BASE_IMAGE=${PHAL_BASE_IMAGE:-${candidate_base_image}}
        if [[ -z "${PHAL_BASE_IMAGE:-}" && -n "${BASE_IMAGE_DIGEST}" && "${BASE_IMAGE_DIGEST}" != "null" ]]; then
            BASE_IMAGE="${BASE_IMAGE}@${BASE_IMAGE_DIGEST}"
        fi
        IMAGE_TAG=${PHAL_IMAGE_TAG:-$(yaml_path_value candidates.train_stack_a.image_tag)}
        NEQUIP_VERSION=$(yaml_path_value candidates.train_stack_a.explicit.nequip)
        ALLEGRO_VERSION=$(yaml_path_value candidates.train_stack_a.explicit.allegro)
    else
        BASE_IMAGE=${PHAL_BASE_IMAGE:-$(yaml_path_value scaffold_defaults.base_image)}
        IMAGE_TAG=${PHAL_IMAGE_TAG:-$(yaml_path_value scaffold_defaults.image_tag)}
        NEQUIP_VERSION=""
        ALLEGRO_VERSION=""
        BASE_IMAGE_DIGEST=""
    fi

    require_value PHAL_PLATFORM "${PLATFORM}"
    require_value PHAL_BASE_IMAGE "${BASE_IMAGE}"
    require_value PHAL_IMAGE_TAG "${IMAGE_TAG}"
    reject_floating_ref PHAL_BASE_IMAGE "${BASE_IMAGE}"
    reject_floating_ref PHAL_IMAGE_TAG "${IMAGE_TAG}"
    if [[ "${target}" == "train-probe" ]]; then
        require_value nequip "${NEQUIP_VERSION}"
        require_value allegro "${ALLEGRO_VERSION}"
    fi
}

image_ref() {
    repository="${IMAGE_NAMESPACE}/$1"
    if [[ -n "${REGISTRY}" ]]; then
        repository="${REGISTRY%/}/${repository}"
    fi
    echo "${repository}:${IMAGE_TAG}"
}

dockerfile_for() {
    echo "${SCRIPT_DIR}/Dockerfile.$1"
}

platform_slug() {
    echo "${PLATFORM//\//-}"
}

print_plan() {
    target=$1
    validate_target "${target}"
    version_context "${target}"
    echo "target=${target}"
    echo "platform=${PLATFORM}"
    echo "dockerfile=$(dockerfile_for "${target}")"
    echo "build_context=${PROJECT_ROOT}"
    echo "base_image=${BASE_IMAGE}"
    if [[ -n "${BASE_IMAGE_DIGEST}" ]]; then
        echo "base_image_digest=${BASE_IMAGE_DIGEST}"
    fi
    echo "image=$(image_ref "${target}")"
    echo "versions=${VERSIONS_FILE}"
    echo "runtime_mount=${PROJECT_ROOT}/runtime:/runtime"
    echo "evidence_dir=${EVIDENCE_DIR}"
    if [[ "${target}" == "train-probe" ]]; then
        echo "nequip=${NEQUIP_VERSION}"
        echo "allegro=${ALLEGRO_VERSION}"
    fi
}

build_args() {
    BUILD_ARGS=(
        --build-arg "BASE_IMAGE=${BASE_IMAGE}"
        --build-arg "IMAGE_TAG=${IMAGE_TAG}"
    )
    if [[ "${target}" == "train-probe" ]]; then
        BUILD_ARGS+=(
            --build-arg "NEQUIP_VERSION=${NEQUIP_VERSION}"
            --build-arg "ALLEGRO_VERSION=${ALLEGRO_VERSION}"
        )
    fi
}

build_image() {
    target=$1
    validate_target "${target}"
    version_context "${target}"
    require_command docker
    build_args
    docker build \
        --platform "${PLATFORM}" \
        "${BUILD_ARGS[@]}" \
        --file "$(dockerfile_for "${target}")" \
        --tag "$(image_ref "${target}")" \
        "${PROJECT_ROOT}"
}

push_image() {
    target=$1
    validate_target "${target}"
    version_context "${target}"
    require_command docker
    if [[ -z "${REGISTRY}" ]]; then
        echo "PHAL_REGISTRY is required for push" >&2
        exit 2
    fi
    mkdir -p "${EVIDENCE_DIR}"
    metadata_file=${PHAL_METADATA_FILE:-"${EVIDENCE_DIR}/${target}-build-metadata.json"}
    build_args
    docker buildx build \
        --platform "${PLATFORM}" \
        "${BUILD_ARGS[@]}" \
        --file "$(dockerfile_for "${target}")" \
        --tag "$(image_ref "${target}")" \
        --metadata-file "${metadata_file}" \
        --push \
        "${PROJECT_ROOT}"
}

tag_image() {
    target=$1
    destination_tag=$2
    validate_target "${target}"
    version_context "${target}"
    reject_floating_ref destination_tag "${destination_tag}"
    require_command docker
    repository=$(image_ref "${target}")
    docker tag "${repository}" "${repository%:*}:${destination_tag}"
}

export_oci() {
    target=$1
    validate_target "${target}"
    version_context "${target}"
    require_command docker
    mkdir -p "${OUTPUT_DIR}"
    archive="${OUTPUT_DIR}/phal-${target}-${IMAGE_TAG}-$(platform_slug).oci.tar"
    build_args
    docker buildx build \
        --platform "${PLATFORM}" \
        "${BUILD_ARGS[@]}" \
        --file "$(dockerfile_for "${target}")" \
        --output "type=oci,dest=${archive}" \
        "${PROJECT_ROOT}"
    echo "wrote ${archive}"
}

build_sif() {
    target=$1
    validate_target "${target}"
    version_context "${target}"
    source_uri=${PHAL_OCI_URI:-}
    if [[ -z "${source_uri}" ]]; then
        echo "PHAL_OCI_URI is required for sif; Miyabi must pull from a registry and cannot use a local Docker daemon" >&2
        exit 2
    fi
    reject_floating_ref PHAL_OCI_URI "${source_uri}"
    require_command apptainer
    mkdir -p "${OUTPUT_DIR}"
    output="${OUTPUT_DIR}/phal-${target}-${IMAGE_TAG}-$(platform_slug).sif"
    apptainer build "${output}" "${source_uri}"
    echo "wrote ${output}"
}

usage() {
    cat <<'EOF'
Usage: containers/build.sh COMMAND [TARGET] [ARG]

Commands:
  plan TARGET            Print the resolved build plan without external tools.
  build TARGET           Build one local Docker image for the requested platform.
  push TARGET            Build with Buildx and push to PHAL_REGISTRY.
  build-all              Build the four unchanged Milestone 1 scaffold images.
  tag TARGET TAG         Add an explicit, non-floating Docker tag.
  oci TARGET             Export a platform-specific OCI archive with Buildx.
  sif TARGET             Convert PHAL_OCI_URI from a registry to a SIF image.
  help                    Show this message.

Environment overrides:
  PHAL_PLATFORM           Defaults to linux/arm64.
  PHAL_BASE_IMAGE         Overrides the target's manifest candidate.
  PHAL_IMAGE_TAG          Overrides the target's explicit image tag.
  PHAL_REGISTRY           Registry prefix used by image tags and push.
  PHAL_IMAGE_NAMESPACE    Defaults to phal.
  PHAL_OCI_URI            Required by sif, e.g. docker://registry/phal/train-probe:tag.
  PHAL_VERSIONS_FILE
  PHAL_CONTAINER_OUTPUT_DIR
  PHAL_EVIDENCE_DIR       Defaults to build/evidence.
  PHAL_METADATA_FILE      Buildx metadata output for push.

No command installs Docker or Apptainer. Docker/Buildx builds run on an external
builder; SIF conversion runs on Miyabi with the site Apptainer module loaded.
EOF
}

command_name=${1:-help}

case "${command_name}" in
    plan) print_plan "${2:?TARGET is required}" ;;
    build) build_image "${2:?TARGET is required}" ;;
    push) push_image "${2:?TARGET is required}" ;;
    build-all)
        for target in ${SCAFFOLD_TARGETS}; do
            build_image "${target}"
        done
        ;;
    tag) tag_image "${2:?TARGET is required}" "${3:?TAG is required}" ;;
    oci) export_oci "${2:?TARGET is required}" ;;
    sif) build_sif "${2:?TARGET is required}" ;;
    help|-h|--help) usage ;;
    *) usage >&2; exit 2 ;;
esac
