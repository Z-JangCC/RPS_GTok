# Anonymization Audit

This repository has been scrubbed for anonymous review.

Checked categories:

- author names and user names;
- local absolute filesystem paths;
- emails and mail links;
- external experiment tracking identifiers;
- machine names, job-scheduler fragments, SSH keys, API keys, and credential material;
- committed logs, checkpoints, raw run directories, caches, notebook outputs;
- embedded Git metadata or repository history.

Intentional non-identifying references:

- public dataset names and loader names;
- repository-relative output paths under `runs/`.

Local processed-dataset cache paths are written as `<repo-local-cache>` in
checked-in reports and regenerated metadata.
