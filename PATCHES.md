# CVAT patches for the Requirement Analyzer

The running server was left as installed. These are the only CVAT source edits.

## `cvat-ui/webpack.config.js`

The UI plugin `plugins/requirement-analyzer` is included in `defaultPlugins` next to the existing SAM plugin. No canvas or server file is changed.

The plugin adds:

- Project actions: **Requirement Analyzer**
- Task actions: **Annotate by hand** and **Auto-annotate this task**
- Page: `/projects/<id>/requirement-analyzer`

## Upgrade

1. Re-apply the one-line `defaultPlugins` change if an upstream merge drops it.
2. Keep the directory `cvat-ui/plugins/requirement-analyzer/`.
3. Rebuild only the UI image, from the CVAT checkout, in WSL:

```bash
cd "/mnt/c/sreeraj/annotation tool/cvat"
export CVAT_VERSION=dev
docker compose build cvat_ui
docker compose up -d cvat_ui
```

That rebuild replaces the UI container. It does not rebuild the server, Nuclio, or the database. Do this only when you want the new menu entries. The analyzer API is published on port 8095, because Traefik already uses 8090. From the CVAT UI the same service is `http://localhost:8080/analyzer`.

## Why the UI is a plugin

CVAT 2.76.1, which is the server that is running, returns 404 for `GET /api/functions`. Native auto-annotation agents (`cvat-cli function create-native`) are not available on this install. The task menu action calls the sidecar, which writes shapes through the annotations API with `source=auto` and moves each job to stage `validation`, state `new`.

A Nuclio function is included at `analyzer-service/nuclio/` for a later deploy. CVAT's detector call sends only the image, not the task id, so that function cannot choose a project profile by itself. The menu action is the path that knows the project.
