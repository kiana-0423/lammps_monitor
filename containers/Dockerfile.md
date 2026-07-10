# syntax=docker/dockerfile:1
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG IMAGE_TAG
LABEL org.opencontainers.image.title="PHAL MD template" \
      org.opencontainers.image.description="Milestone 1 MLIP-MD container scaffold; no software is installed" \
      org.opencontainers.image.version="${IMAGE_TAG}" \
      org.opencontainers.image.source="containers/Dockerfile.md" \
      org.phal.versions-file="containers/versions.yaml" \
      org.phal.stage="md"

WORKDIR /opt/phal

ENV PHAL_HOME=/opt/phal \
    PHAL_CONFIG_DIR=/opt/phal/configs \
    PHAL_RUNTIME_DIR=/runtime
