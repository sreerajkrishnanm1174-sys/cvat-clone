import React, { useEffect, useState } from 'react';
import { Route, useParams } from 'react-router';
import Button from 'antd/lib/button';
import Modal from 'antd/lib/modal';
import Upload from 'antd/lib/upload';
import { getCore } from 'cvat-core-wrapper';
import { PluginEntryPoint } from 'components/plugins-entrypoint';

const core = getCore();

function analyzerBase(): string {
    return window.localStorage.getItem('analyzerBase') || '/analyzer';
}

async function cvatToken(): Promise<string> {
    const saved = window.sessionStorage.getItem('analyzerToken');
    if (saved) {
        return saved;
    }
    const csrf = decodeURIComponent((document.cookie.match(/(?:^|; )csrftoken=([^;]+)/) || [])[1] || '');
    const response = await fetch('/api/auth/access_tokens', {
        method: 'POST',
        credentials: 'same-origin',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': csrf,
        },
        body: JSON.stringify({ name: 'Requirement Analyzer' }),
    });
    if (!response.ok) {
        throw new Error('Create a CVAT access token in your profile, then save it with localStorage.setItem("analyzerToken", token).');
    }
    const created = await response.json();
    const value = created.value as string;
    window.sessionStorage.setItem('analyzerToken', value);
    return value;
}

async function analyzerFetch(path: string, options: RequestInit = {}): Promise<any> {
    const token = await cvatToken();
    const headers = new Headers(options.headers || {});
    headers.set('Authorization', `Bearer ${token}`);
    if (options.body && !(options.body instanceof FormData) && !headers.has('Content-Type')) {
        headers.set('Content-Type', 'application/json');
    }
    const response = await fetch(`${analyzerBase()}${path}`, { ...options, headers });
    if (!response.ok) {
        const detail = await response.text();
        throw new Error(detail || response.statusText);
    }
    return response.json();
}

function AnalyzerPage(): JSX.Element {
    const params = useParams<{ id: string }>();
    const projectId = params.id;
    const [spec, setSpec] = useState<any>(null);
    const [diff, setDiff] = useState<any>(null);
    const [learning, setLearning] = useState<any>(null);
    const [message, setMessage] = useState('');

    const refresh = async (): Promise<void> => {
        const [nextSpec, nextLearning] = await Promise.all([
            analyzerFetch(`/projects/${projectId}/spec`),
            analyzerFetch(`/projects/${projectId}/learning`),
        ]);
        setSpec(nextSpec);
        setLearning(nextLearning);
    };

    useEffect(() => {
        refresh().catch((error: Error) => setMessage(error.message));
    }, [projectId]);

    const upload = async (file: File): Promise<void> => {
        const body = new FormData();
        body.append('file', file);
        const next = await analyzerFetch(`/projects/${projectId}/requirements`, { method: 'POST', body });
        setSpec({ spec: next.spec, version: next.version });
        setMessage('Read the questions, then preview the labels.');
    };

    return (
        <div style={{ padding: 24, maxWidth: 860 }}>
            <h2>Requirement Analyzer</h2>
            <p>{learning?.mode_text}</p>
            <p>{learning?.summary}</p>
            <Upload beforeUpload={(file) => { upload(file as File).catch((error: Error) => setMessage(error.message)); return false; }}>
                <Button type='primary'>Upload requirement document</Button>
            </Upload>
            {(spec?.spec?.questions || []).map((question: { id: string; text: string }) => (
                <div key={question.id} style={{ marginTop: 12 }}>
                    <div>{question.text}</div>
                    <Button onClick={() => {
                        const answer = window.prompt(question.text) || '';
                        analyzerFetch(`/projects/${projectId}/spec/answers`, {
                            method: 'POST',
                            body: JSON.stringify({ question_id: question.id, answer }),
                        }).then(refresh).catch((error: Error) => setMessage(error.message));
                    }}
                    >
                        Answer
                    </Button>
                </div>
            ))}
            <div style={{ marginTop: 16 }}>
                <Button onClick={() => analyzerFetch(`/projects/${projectId}/labels/preview`, { method: 'POST' }).then(setDiff)}>
                    Preview label changes
                </Button>
                <Button
                    type='primary'
                    style={{ marginLeft: 8 }}
                    onClick={() => analyzerFetch(`/projects/${projectId}/labels/apply`, {
                        method: 'POST',
                        body: JSON.stringify({ confirm_type_changes: false }),
                    }).then(() => setMessage('Labels now match the requirement.'))}
                >
                    Apply labels
                </Button>
            </div>
            {diff && (
                <pre>{JSON.stringify({
                    create: (diff.create || []).map((item: { name: string }) => item.name),
                    update: (diff.update || []).map((item: { name: string }) => item.name),
                    blocked: diff.blocked,
                }, null, 2)}
                </pre>
            )}
            <h3>What the agent has learned</h3>
            <ul>{(learning?.habits || []).map((habit: string) => <li key={habit}>{habit}</li>)}</ul>
            <div>
                <Button onClick={() => analyzerFetch(`/projects/${projectId}/learning/settings`, {
                    method: 'POST', body: JSON.stringify({ annotation_mode: 'human_first' }),
                }).then(refresh)}
                >
                    Annotate by hand
                </Button>
                <Button
                    style={{ marginLeft: 8 }}
                    onClick={() => analyzerFetch(`/projects/${projectId}/learning/settings`, {
                        method: 'POST', body: JSON.stringify({ annotation_mode: 'auto_first' }),
                    }).then(refresh)}
                >
                    The agent annotates first
                </Button>
                <Button
                    style={{ marginLeft: 8 }}
                    onClick={() => analyzerFetch(`/projects/${projectId}/learning/settings`, {
                        method: 'POST', body: JSON.stringify({ annotation_mode: 'mixed' }),
                    }).then(refresh)}
                >
                    Mix both
                </Button>
            </div>
            <div style={{ marginTop: 12 }}>
                <Button onClick={() => analyzerFetch(`/projects/${projectId}/learning/settings`, {
                    method: 'POST',
                    body: JSON.stringify({ learn_enabled: !learning?.learn_enabled }),
                }).then(refresh)}
                >
                    {learning?.learn_enabled ? 'Switch learning off' : 'Switch learning on'}
                </Button>
                <Button
                    style={{ marginLeft: 8 }}
                    onClick={() => analyzerFetch(`/projects/${projectId}/learning/settings`, {
                        method: 'POST',
                        body: JSON.stringify({ compare_with_agent: !learning?.compare_with_agent }),
                    }).then(refresh)}
                >
                    {learning?.compare_with_agent ? 'Stop comparing with the agent' : 'Compare with the agent'}
                </Button>
                <Button
                    danger
                    style={{ marginLeft: 8 }}
                    onClick={() => Modal.confirm({
                        title: 'Forget everything learned for this project?',
                        onOk: () => analyzerFetch(`/projects/${projectId}/learning/reset`, { method: 'POST' }).then(refresh),
                    })}
                >
                    Forget everything for this project
                </Button>
            </div>
            {message && <p>{message}</p>}
        </div>
    );
}

function projectMenu({ targetProps }: { targetProps: { projectInstance: { id: number } } }) {
    const id = targetProps.projectInstance.id;
    return {
        key: 'requirement-analyzer',
        label: <a href={`/projects/${id}/requirement-analyzer`}>Requirement Analyzer</a>,
    };
}

function handAnnotate({ targetProps }: { targetProps: { taskInstance: { id: number; projectId: number } } }) {
    const task = targetProps.taskInstance;
    return {
        key: 'annotate-by-hand',
        label: 'Annotate by hand',
        onClick: async () => {
            if (task.projectId) {
                await analyzerFetch(`/projects/${task.projectId}/learning/settings`, {
                    method: 'POST',
                    body: JSON.stringify({ annotation_mode: 'human_first', learn_enabled: true }),
                });
            }
            const page = await core.jobs.get({ taskID: task.id });
            const job = (page.results || page)[0];
            window.location.assign(`/tasks/${task.id}/jobs/${job.id}`);
        },
    };
}

function autoAnnotate({ targetProps }: { targetProps: { taskInstance: { id: number; projectId: number } } }) {
    const task = targetProps.taskInstance;
    return {
        key: 'auto-annotate-task',
        label: 'Auto-annotate this task',
        onClick: async () => {
            const result = await analyzerFetch(`/projects/${task.projectId}/annotate`, {
                method: 'POST',
                body: JSON.stringify({ task_id: task.id, mode: 'auto_first' }),
            });
            window.alert(result.message || `Automatic annotation ${result.status}. Review starts with the flagged objects.`);
        },
    };
}

const builder: PluginEntryPoint = ({ dispatch, actionCreators }) => {
    const page = () => (
        <Route exact path='/projects/:id/requirement-analyzer' component={AnalyzerPage} />
    );
    dispatch(actionCreators.addUIComponent('router', page, { weight: 50, shouldBeRendered: () => true }));
    dispatch(actionCreators.addUIComponent('projectActions.items', projectMenu, { weight: 45 }));
    dispatch(actionCreators.addUIComponent('taskActions.items', handAnnotate, { weight: 25 }));
    dispatch(actionCreators.addUIComponent('taskActions.items', autoAnnotate, { weight: 26 }));
    return {
        name: 'Requirement Analyzer',
        destructor: () => undefined,
    };
};

window.addEventListener('plugins.ready', () => {
    const register = (window as unknown as { cvatUI?: { registerComponent: (fn: PluginEntryPoint) => void } }).cvatUI;
    register?.registerComponent(builder);
}, { once: true });

export default builder;
