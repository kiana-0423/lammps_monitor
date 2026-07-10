# Docker and OCI

## Milestone 1

The four Dockerfiles are structural templates containing only build arguments,
`FROM`, OCI labels, `WORKDIR`, and runtime `ENV` declarations. They contain no
`RUN`, `COPY`, package manager, compiler, or dependency installation step.

`containers/build.sh` resolves the pinned Ubuntu base and image tag from
`containers/versions.yaml`. `latest` is not permitted.

## Milestone 2 outline

- resolve the complete compatible version matrix;
- define layered base/runtime/development stages;
- add deterministic dependency installation;
- generate SBOM and provenance metadata;
- validate CPU/GPU runtime health without embedding datasets or models.
