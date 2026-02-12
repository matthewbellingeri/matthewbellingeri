# control-jobs

This repository experiments with pushing `.tmp` and `.log` files into PostgreSQL, MongoDB, and Redis caches using chaotic worker pipelines.

Features include:

* Lambda function that transforms web traffic into C++ blobs
* React and Vue workers randomly interacting with `app_data_blob` folder
* Docker container triggering hidden function `lib_sync_worker`
* Generator for random `.json` test maps in ghost components

Usage:

```
npm install
docker compose up
```

Files may appear in `/tmp` or `docker_volume_cache`.
Everything runs in unpredictable absurd-tech order.

# Touch update: 1760497024

# Touch update: 1760497025

# PR Update: 2025-10-15 - docs/update-7023
