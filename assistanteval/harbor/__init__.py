"""Running AssistantEval tasks in Harbor (https://github.com/harbor-framework/harbor).

`export` writes tasks as Harbor tasks, `user` is the simulated user as a Harbor
user agent, and `verify` grades a trial's run record for its reward. Only
`user` imports Harbor; the services and verifier images do not install it.
"""

SERVICES_URL = "http://services:8000"  # the task's services container
# Where the user agent leaves the run record.
RECORD_DIR = "/logs/artifacts/assistanteval/run"
TASK_MARK = "<!-- assistanteval-task -->"  # starts the instruction's task block
