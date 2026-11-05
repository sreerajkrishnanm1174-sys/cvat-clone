"""Thin CVAT REST adapter. Prototype logic does not call this module."""

from __future__ import annotations

import time
from pathlib import Path

import httpx


class CvatError(RuntimeError):
    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status = status
        self.detail = detail


class CvatClient:
    def __init__(self, base_url: str, token: str, scheme: str = "Bearer", organization: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.organization = organization
        headers = {"Authorization": f"{scheme} {token}" if token else ""}
        if organization is not None:
            headers["X-Organization"] = organization
        self._client = httpx.Client(base_url=self.base_url, headers=headers, timeout=60)

    def close(self) -> None:
        self._client.close()

    def _request(self, method: str, path: str, **kwargs) -> httpx.Response:
        response = self._client.request(method, path, **kwargs)
        if response.status_code >= 400:
            raise CvatError(response.status_code, response.text[:500])
        return response

    def list_projects(self) -> list[dict]:
        data = self._request("GET", "/api/projects", params={"page_size": 50}).json()
        return data.get("results", [])

    def create_project(self, name: str) -> dict:
        return self._request("POST", "/api/projects", json={"name": name}).json()

    def list_tasks(self, project_id: int | None = None) -> list[dict]:
        params = {"page_size": 50}
        if project_id is not None:
            params["project_id"] = project_id
        data = self._request("GET", "/api/tasks", params=params).json()
        return data.get("results", [])

    def create_task(self, name: str, project_id: int | None = None) -> dict:
        body = {"name": name}
        if project_id is not None:
            body["project_id"] = project_id
        return self._request("POST", "/api/tasks", json=body).json()

    def upload_task_images(self, task_id: int, files: list[tuple[str, bytes]]) -> None:
        """Upload with CVAT's start / multiple / finish steps, then wait until import finishes."""
        if not files:
            return
        path = f"/api/tasks/{task_id}/data"
        self._request("POST", path, headers={"Upload-Start": "true"})
        for batch in _file_batches(files):
            payload = [
                (f"client_files[{index}]", (name, content, _content_type(name)))
                for index, (name, content) in enumerate(batch)
            ]
            self._request(
                "POST",
                path,
                data={"image_quality": "70"},
                files=payload,
                headers={"Upload-Multiple": "true"},
                timeout=300,
            )
        started = self._request(
            "POST",
            path,
            json=_import_body(files),
            headers={"Upload-Finish": "true"},
            timeout=120,
        )
        body = started.json() if started.content else {}
        rq_id = body.get("rq_id")
        if not rq_id:
            raise CvatError(400, "CVAT did not start the image import.")
        self._wait_request(str(rq_id))
        task = self.get_task(task_id)
        if not int(task.get("size") or 0):
            raise CvatError(400, "CVAT finished the import, but the task still has no images.")

    def _wait_request(self, rq_id: str, timeout: float = 600) -> dict:
        deadline = time.monotonic() + timeout
        while True:
            data = self._request("GET", f"/api/requests/{rq_id}").json()
            status = str(data.get("status") or "")
            if status == "finished":
                return data
            if status == "failed":
                raise CvatError(400, data.get("message") or "CVAT could not import the images.")
            if time.monotonic() > deadline:
                raise CvatError(408, "CVAT is still importing the images. Open the task after it finishes, then annotate.")
            time.sleep(2)

    def get_project(self, project_id: int) -> dict:
        return self._request("GET", f"/api/projects/{project_id}").json()

    def get_labels(self, project_id: int) -> list[dict]:
        response = self._request("GET", "/api/labels", params={"project_id": project_id, "page_size": 1000})
        data = response.json()
        return data.get("results", data if isinstance(data, list) else [])

    def patch_project_labels(self, project_id: int, labels: list[dict]) -> dict:
        return self._request("PATCH", f"/api/projects/{project_id}", json={"labels": labels}).json()

    def patch_project(self, project_id: int, payload: dict) -> dict:
        return self._request("PATCH", f"/api/projects/{project_id}", json=payload).json()

    def delete_project(self, project_id: int) -> None:
        self._request("DELETE", f"/api/projects/{project_id}")

    def patch_task(self, task_id: int, payload: dict) -> dict:
        return self._request("PATCH", f"/api/tasks/{task_id}", json=payload).json()

    def delete_task(self, task_id: int) -> None:
        self._request("DELETE", f"/api/tasks/{task_id}")

    def get_task(self, task_id: int) -> dict:
        return self._request("GET", f"/api/tasks/{task_id}").json()

    def get_task_meta(self, task_id: int) -> dict:
        return self._request("GET", f"/api/tasks/{task_id}/data/meta").json()

    def get_job(self, job_id: int) -> dict:
        return self._request("GET", f"/api/jobs/{job_id}").json()

    def list_project_tasks(self, project_id: int) -> list[dict]:
        rows = []
        page = 1
        while page < 50:
            data = self._request(
                "GET",
                "/api/tasks",
                params={"project_id": project_id, "page_size": 100, "page": page},
            ).json()
            rows.extend(data.get("results") or [])
            if not data.get("next"):
                break
            page += 1
        return rows

    def get_jobs(self, task_id: int | None = None, project_id: int | None = None) -> list[dict]:
        params = {"page_size": 1000}
        if task_id is not None:
            params["task_id"] = task_id
        if project_id is not None:
            params["project_id"] = project_id
        data = self._request("GET", "/api/jobs", params=params).json()
        return data.get("results", [])

    def get_annotations(self, kind: str, resource_id: int) -> dict:
        return self._request("GET", f"/api/{kind}/{resource_id}/annotations").json()

    def put_annotations(self, kind: str, resource_id: int, payload: dict) -> None:
        self._request("PUT", f"/api/{kind}/{resource_id}/annotations", json=payload)

    def create_annotations(self, kind: str, resource_id: int, payload: dict) -> None:
        self._request(
            "PATCH",
            f"/api/{kind}/{resource_id}/annotations",
            params={"action": "create"},
            json=payload,
        )

    def update_job(self, job_id: int, payload: dict) -> dict:
        return self._request("PATCH", f"/api/jobs/{job_id}", json=payload).json()

    def list_issues(self, job_id: int) -> list[dict]:
        data = self._request("GET", "/api/issues", params={"job_id": job_id, "page_size": 1000}).json()
        if isinstance(data, list):
            return data
        return data.get("results") or []

    def delete_issue(self, issue_id: int) -> None:
        self._request("DELETE", f"/api/issues/{issue_id}")

    def clear_review_issues(self, job_id: int) -> int:
        removed = 0
        for issue in self.list_issues(job_id):
            if str(issue.get("message") or "").startswith("Review first:"):
                self.delete_issue(int(issue["id"]))
                removed += 1
        return removed

    def create_webhook(self, project_id: int, target_url: str, secret: str, events: list[str]) -> dict:
        return self._request(
            "POST",
            "/api/webhooks",
            json={
                "target_url": target_url,
                "description": "Requirement Analyzer",
                "project_id": project_id,
                "type": "project",
                "content_type": "application/json",
                "secret": secret,
                "is_active": True,
                "enable_ssl": False,
                "events": events,
            },
        ).json()

    def get_frame(self, task_id: int, frame: int) -> bytes:
        response = self._request(
            "GET",
            f"/api/tasks/{task_id}/data",
            params={"type": "frame", "number": frame, "quality": "original"},
        )
        return response.content

    def lambda_functions(self) -> list[dict]:
        data = self._request("GET", "/api/lambda/functions").json()
        return data if isinstance(data, list) else data.get("results", [])


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".mpeg", ".mpg", ".m4v", ".wmv", ".3gp"}


def is_video_name(name: str) -> bool:
    return Path(name).suffix.lower() in VIDEO_EXTENSIONS


def _content_type(name: str) -> str:
    if is_video_name(name):
        return "video/mp4" if Path(name).suffix.lower() == ".mp4" else "application/octet-stream"
    return "application/octet-stream"


def _import_body(files: list[tuple[str, bytes]]) -> dict:
    body = {"image_quality": 70}
    if any(is_video_name(name) for name, _content in files):
        # Match a manual CVAT video upload: decode frames into cached chunks.
        # A raw video chunk is what the player fails to show.
        body["use_zip_chunks"] = True
        body["use_cache"] = True
    return body


def _file_batches(files: list[tuple[str, bytes]], limit: int = 80 * 1024 * 1024) -> list[list[tuple[str, bytes]]]:
    batches: list[list[tuple[str, bytes]]] = []
    current: list[tuple[str, bytes]] = []
    size = 0
    for item in files:
        item_size = len(item[1])
        if current and size + item_size > limit:
            batches.append(current)
            current = []
            size = 0
        current.append(item)
        size += item_size
    if current:
        batches.append(current)
    return batches
