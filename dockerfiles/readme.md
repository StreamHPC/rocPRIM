# Dockerfiles

Benchmark results that are required for generating configs via `/scripts/autotune/create_optimization.py` can be easily gathered with `autotune.dockerfile`. It is recommended to run this via `docker compose` with the following configuration:

```yaml
# compose.yaml
services:
    tune:
        # Expose the GPU to the docker environment.
        devices:
        - "/dev/kfd"
        - "/dev/dri"
        # If we want to export the resulting artifacts to the host,
        # we'll want to be the same user as host.
        user: "${UID}:${GID}"
        build:
            dockerfile: "autotune.dockerfile"
            args:
                # The repository that will be cloned
                REPO_URL: "https://github.com/ROCm/rocPRIM.git"
                # The benchmark targets that need to be compiled.
                BENCHMARK_TARGETS: "benchmark_config_tuning"
                # The GPU architecture that is being targetting during build.
                # When building for multiple targets, a single GPU_TARGET needs
                # to be selected for the runtime tuning by adding it to the environment.
                GPU_TARGET: "gfx1030;gfx942"
        volumes:
        # Make the auto tune results available on host
        - "./artifacts:/workspace/autotune_results"
        environment:
            # Write the auto tune results the folder that we expose to host
            AUTOTUNE_RESULT_DIR: "/workspace/autotune_results"
            AUTOTUNE_FILENAME_REGEX: "^benchmark"
            AUTOTUNE_ALGORITHM_REGEX: ""
            AUTOTUNE_SIZE: ""
            AUTOTUNE_TRIALS: ""
            # The GPU target we run the tuning for.
            # If ommitted, the GPU_TARGET from the build arguments is used.
            GPU_TARGET: "gfx942"
```

A different branch and repository can be selected by changing `REPO_URL`, which is passed to [an `ADD` instruction](https://docs.docker.com/reference/dockerfile/#add).
Private repositories [can be accessed with `ssh`](https://docs.docker.com/reference/dockerfile/#adding-files-from-a-git-repository) or [via a `GIT_AUTH_TOKEN`](https://docs.docker.com/build/building/secrets/#git-authentication-for-remote-contexts).

A specific tuning benchmark can be built by changing `BENCHMARK_TARGETS`, for example:

```yaml
BENCHMARK_TARGETS: "benchmark_device_reduce"
```
