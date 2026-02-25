# Mention Hook

A webhook-driven bot that listens for `@mentions` in Jira and GitLab comments, then responds using AI-powered skills. Mention a skill by name in any issue or merge request comment and the bot replies with an AI-generated response.

Supports two modes per skill:
- **R2R mode** -- retrieves context from an [R2R](https://github.com/SciPhi-AI/R2R) knowledge base before answering (retrieval-augmented generation).
- **Normal AI mode** -- sends the issue/MR context directly to the LLM with no external retrieval.

## How It Works

1. Jira or GitLab sends a webhook when a comment is created.
2. The gateway parses the event and looks for an `@skill_name` mention.
3. The matched skill gathers context (from R2R, the platform API, or both), builds a prompt, and calls an LLM via a LiteLLM proxy.
4. The response is posted back as a comment.

## Skills

### `@support` (R2R mode)

Searches your R2R knowledge base for relevant documents, then answers the user's question using that context. Good for internal docs, runbooks, FAQs, etc.

**Usage** -- comment on any Jira issue or GitLab issue/MR:

```
@support How do I reset my VPN credentials?
```

**Config** (`config.yaml`):

```yaml
skills:
  support:
    mention_name: "support"        # the @name users type in comments
    prompt_file: "support"         # loads prompts/support.md
    model: "gpt-4o"
    r2r_search_limit: 5           # number of R2R chunks to retrieve
    max_context_chars: 16000       # truncate retrieved context beyond this
```

**Required env vars:** `R2R_BASE_URL`, `R2R_ADMIN_EMAIL`, `R2R_ADMIN_PASSWORD`

### `@review` (Normal AI mode)

Reviews a GitLab merge request diff and posts feedback. No R2R -- the MR diff and description are sent directly to the LLM.

**Usage** -- comment on a GitLab merge request:

```
@review Please check for security issues
```

**Config** (`config.yaml`):

```yaml
skills:
  review:
    mention_name: "review"
    prompt_file: "review"
    model: "gpt-4o"
    max_diff_chars: 32000
    ignore_patterns: ["*.lock", "*.min.js", "*.min.css", "package-lock.json", "yarn.lock"]
```

### `@summarize` (Normal AI mode)

Summarizes the full discussion thread of a Jira issue or GitLab issue/MR. Fetches all comments from the platform API and sends them to the LLM.

**Usage** -- comment on any issue or MR:

```
@summarize
```

**Config** (`config.yaml`):

```yaml
skills:
  summarize:
    mention_name: "summarize"
    prompt_file: "summarize"
    model: "gpt-4o"
    max_context_chars: 24000
```

## R2R vs Normal AI

| | R2R skills (`@support`) | Normal AI skills (`@review`, `@summarize`) |
|---|---|---|
| **Context source** | R2R knowledge base search | Platform API (comments, diffs) |
| **Use case** | Answer questions from your docs | Analyze issue/MR content directly |
| **Requires R2R?** | Yes | No |
| **Adding a new one** | Register class as `R2RSkill` | Subclass `BaseSkill` with custom logic |

To add another R2R-backed skill (e.g. `@it-help`), register it in `skills.py` and add a config entry:

```python
# skills.py
SKILL_REGISTRY: dict[str, type[BaseSkill]] = {
    "support": R2RSkill,
    "it-support": R2RSkill,   # same class, different mention & prompt
    ...
}
```

```yaml
# config.yaml
skills:
  it-support:
    mention_name: "it-help"        # users type @it-help
    prompt_file: "it-support"      # loads prompts/it-support.md
    model: "gpt-4o"
    r2r_search_limit: 5
```

Then create `prompts/it-support.md` with the system prompt. The `mention_name` controls what users type, while `prompt_file` controls which prompt template is loaded -- they don't have to match.

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your actual values
```

Key variables:

| Variable | Required | Description |
|---|---|---|
| `LITELLM_BASE_URL` | Yes | LiteLLM proxy URL (e.g. `http://localhost:4000/v1`) |
| `LITELLM_API_KEY` | Yes | API key for LiteLLM |
| `JIRA_BASE_URL` | For Jira | Your Jira instance URL |
| `JIRA_USER_EMAIL` | For Jira | Bot account email |
| `JIRA_API_TOKEN` | For Jira | Jira API token |
| `GITLAB_BASE_URL` | For GitLab | Your GitLab instance URL |
| `GITLAB_API_TOKEN` | For GitLab | GitLab personal access token |
| `R2R_BASE_URL` | For R2R skills | R2R server URL |
| `R2R_ADMIN_EMAIL` | For R2R skills | R2R admin email |
| `R2R_ADMIN_PASSWORD` | For R2R skills | R2R admin password |

### 3. Edit `config.yaml`

Each skill has two key fields:
- `mention_name` -- the `@name` users type in comments to trigger the skill.
- `prompt_file` -- the filename (without `.md`) loaded from `prompts/` as the system prompt.

Set the model name and tuning parameters for each skill. You can use any model supported by your LiteLLM proxy. Add bot account usernames to `gateway.bot_accounts` to prevent self-reply loops.

### 4. Run

```bash
uvicorn gateway:app --host 0.0.0.0 --port 8000
```

Or with Docker:

```bash
docker build -t mention-hook .
docker run --env-file .env -p 8000:8000 mention-hook
```

### 5. Configure webhooks

- **Jira**: Add a webhook pointing to `https://your-host/webhook/jira` triggered on "Issue > Comment > Created".
- **GitLab**: Add a project or group webhook pointing to `https://your-host/webhook/gitlab` with the "Comments" trigger enabled.
