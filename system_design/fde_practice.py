###############################################
#                                             #
#              SYSTEM DESIGN FDE              #
#                                             #
###############################################
"""
There are two important phases in a system design interview: 
- The discovery phase: where you ask clarifying questions to understand the problem statement, the users, the data, the constraints, and the success metrics.
- The design phase: where you propose a high-level architecture and justify your choices.
"""



###############################################
#                                             #
#                DISCOVERY PHASE              #
#                                             #
###############################################


"""
Problem Statement
- My understanding:
    - Financial company
    - Various data source incl. sensitive data
    - Automate email drafting based on market research + internal data analysis
    - Compliance + Auditing

1. Users & Workflow
- Who (internal / external): sales team, account team, operations team, only internal for now
- Current workflow: user -> market trend research + spreadsheet analysis + internal docs -> draft email -> review -> send
- Target workflow: similar
- Scale (today → 2yr): 100 users for now to be scaled to 10k users
- Usage pattern (peaks): Towards the end of fiscal year high usage peaks

2. Tasks & Scope
- Capability 1 —  Email drafting:
    - Input: email history, CRM account refs, market research, internal analysis, internal docs
    - Task: draft an email for xyz customer
    - Output: drafted email (possibly in a email client)
    - Autonomy: [read-only / draft / execute / human-in-loop] -> HIL
    - Latency: <10s
- Capability 2 — Market trend research:
    - Input: approved public sources
    - Task: analyse market trends for xyz customer
    - Output: insights from market trend
    - Autonomy: read-only
    - Latency: < 5 - 10 mins
- Capability 3 — Spreadsheet analyst:
    - Input: Spreadsheets
    - Task: Analyze the data for xyz customer
    - Output: stats
    - Autonomy: read + conditional write back (user confirmation required)
    - Latency: < 1 min

3. Data & Systems
- Internal docs (unstructured): Internal docs in Sharepoint 
- Structured data: Spreadsheets, Postgres/BigQuery
- External sources: Financial news, SEC fillings
- Email / CRM: Gmail/Outlook, Salesforce
- Existing AI infra: Greenfield / Azure / GCP / AWS

4. Security & Compliance
- Sensitive data: Internal docs, comp data, client portfolio, PII across all sources
- Permissions: RBAC - sales_rep, analyst, ops_staff, admin - enforced at gateway + data layer
- Isolation / residency: only auth based / EU
- Regulations: GDPR + MiFID II
- Audit / retention: required / 7 years

5. Success Metrics
- Business: time_saved, cost_saved, efficiency_gain
- Quality: draft_edit_rate, draft_acceptance_rate
- Safety: hallucinations, omissions, data_leakage
- Operations:
    - Latency per step
    - Tokens per step
    - Cost per step

Assumptions & MVP Scope
- HIL for all email send - no autonomous external comm
- Market research agent runs as scheduled batch, no-on-demand
- Postgres stay on-prem, access via private network tunnel
- Comm guidelines PDF ingested via RAG pipeline
- GDPR + MiFID II apply - EU Data residency, 7 year temper-proof audit log (Azure Blob WORM policy)
- MVP: single tenant,  Container apps
- Enterprise: multi-tenant, AKS; multi-region
"""


###############################################
#                                             #
#                DESIGN PHASE                 #
#                                             #
###############################################


"""
MVP Scope
- In scope: email drafting, market trends, spreadsheet analysis 
- Out of scope: external users, mobile, fine-tuning, autonomous send
- Pilot: 400 users —> path to 10k+ users

High level Flow
User
- Channel (UI / Email - Web app / Gmail / Outlook)
- API Gateway / RBAC Auth Policies (Azure API Management)
- Request Safety (middleware - DLP/Presidio PII Scan, Prompt Injections - Azure Content Satefy)
    - blocked here if unsafe
- Orchestrator [Azure Foundry / Vertex AI Agent Builder, Router, Planning, Tool registry, Session state / memory]
- Task Agents
        - Agent 1 — Email drafting 
        - Agent 2 — Market Research
        - Agent 3 — Analyst
        - RAG Service - called by agent when needed
- Data Layer 
    - Internal docs: Sharepoint
    - Structured: Sheets, Postgres (on-prem)
    - External: financial news, SEC Filings 
    - CRM/Email: Gmail/Outlook, Salesforce
- Output governance [Compliance service - always sequential, enforced by orchestrator]
- Human Review [draft + sources, approve/feedback/reject]
- Outcome [Gmail draft saved, audit logs updated - Azure Blob WORM]

Task Agents:
- Agent 1 — Email drafting 
    - Model: GPT-4o
    - Tools: Gmail/Outlook API, SalesforceM API, RAG Service
    - Workflow: User prompt via Gmail add-on -> fetch email history + CRM context + comm guidelines -> Draft an email -> Compliance -> human review -> Gmail draft
    - Autonomy: draft-only + human in loop, max. 3 drafts iterations
    - Latency: < 10s
- Agent 2 — Market Research
    - Model: GPT-4o + web grounding
    - Tools: SerpAPI, market data API, Google Drive API
    - Workflow: Schedule trigger (Azure logic apps) -> Fetch approved public sources + internal reports -> Synthesize -> compliance service -> push to Google Drive
    - Autonomy: read-only, fully automated
    - Latency: < 5 -10 mins (async batch)
- Agent 3 — Analyst
    - Model: GPT-4o + Python executor
    - Tools: Sheet API, Postgres API, Python Sandbox
    - Workflow: User triggers sheet add-on -> fetch sheet + query postgres (RBAC enforced at DB level) -> Analyze in sandbox environment -> Compliance service -> Summary + write-back 
    - Autonomy: read + conditional write-back (user confirmation)
    - Latency: < 1 min
- Agent Guardrails
    - Email: max 3 redraft iterations, no autonomous send
    - Market research: approved domains whitelist, read-only
    - Analyst: sandboxed execution, no DROP/DELETE, confirm write-back

Shared services
- Compliance Agent
    - Position: sequential gate after every task agent output
    - Enforced by: orchestrator — mandatory, non-skippable
    - PII scan: Presidio + Azure Content Safety
    - Tone/policy: RAG over comms guidelines + prohibited phrases check
    - Groundedness: LLM-as-judge, citation check - Actions: pass / redact / block
- RAG Service 
    - Ingest: docs -> chunk (paragraph) -> embed -> store + metadata
    - Metadata: doc_id, sensitivity_label, allowed_groups, …
    - Query: ACL filter -> hybrid search (vector + BM25) -> rerank
    - Vector store: Azure AI Search
    - Embedding model: text-embedding-3-small
    - Called-by: email drafting agent, compliance service

Observability & Success Metrics
- Business: time_saved, cost_saved, efficiency_gain
- Quality: draft_acceptance_rate, draft_edit_rate, redraft_rate
- Safety: hallucination_rate, omission_rate, data_leakage_rate
- Operational: latency_per_agent, tokens_per_step, cost_per_request, error_rate, queue_depth
- Infrastructure:
    - Dashboards: Azure Monitor per agent
    - Alerting: PagerDuty / Slack on threshold breach
    - Tracing: App Insights distributed trace per request
    - Feedback loop: approve/reject/redraft logged → weekly review

Tradeoffs:
- GPT-4o for all agents: Consistency in MVP optimize later
- Compliance as shared service: Reusable across all agents, single enforcement point
- Single tenant in MVP: Stronger isolation, higher ops overhead vs multi-tenant
- Container apps over AKS: simpler ops, sufficient for current scale
- ExpressRoute over VPN: no public internet for sensitive DB

—— PROBE ZONE - Only if asked ——

Security & Compliance
- Identity & Access
    - Internal: Azure AD SSO
    - External (future): Azure AD B2C
    - RBAC: sales_rep, analyst, ops_staff, admin
    - Enforced: gateway + DB level (defense in depth)
    - Postgres: column-level access control
- PII Detection
    - Tool: Presidio + Azure Content Safety 
    - Input: Scan before LLM context
    - Output: scan before shown to user
    - Logs: redact before writing to audit log
- Audit logging
    - Entry: timestamp, user_id, role, agent, tools, input, output, action
    - Hot: Azure Monitor -> queryable, dashboards
    - Cold: Azure Blob WORM -> temper-proof, 7 years retention
    - PII redacted before logging
- Data Residency
    - All services: EU region
    - Postgres: on-prem EU, ExpressRoute tunnel
    - No public endpoints

AI/ML Engineering
- LLM: GPT-4o for all agents (MVP)
- Embeddings: text-embedding-3-small
- RAG: paragraph chunking, hybrid search, ACL filter at query time
- Structured data: row/col level access, no SELECT *, provenance returned
- Routing: deterministic rule-based in MVP
- Fine-tuning: avoid MVP, consider later for tone/style

Operational Excellence 
- Hallucination: LLM-as-judgement groundedness score per output
- Feedback: approve/reject/redarft rate tracked over time
- Latency: per agent + per tool call (trace level)
- Errors: alerts on spike, circuit breaker per agent
- Drift: output quality monitored weekly

Scalability
- 400 users: Container apps, auto-scale
- 1000+ users: increase replicas, Postgres replica
- 10k+ users: AKS, horizontal pod autoscaling
- Spikes: Service Bus queue absorbs burst
- External clients: single-tenant per client

Cost & Optimization
- Token tracking: per agent, per user, daily total
- Batch scheduling: market research not on-demand
- Model tiering: cheaper model for routing/classification
- Caching: frequent RAG queries cached
- Budget alerts: per agent spend threshold

Infrastructure
- Compute: Container Apps (auto-scale) → AKS at scale
- Secrets: Azure Key Vault, injected at runtime
- Async: Azure Service Bus for batch jobs
- DB connectivity: ExpressRoute, private endpoint, no public IP
- Scheduling: Azure Logic Apps for market research trigger
"""