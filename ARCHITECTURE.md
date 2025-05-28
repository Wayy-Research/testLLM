# testLLM Complete Architecture

This document outlines the complete 3-part architecture for the testLLM project.

## Current Directory: `/testLLM/` - Framework Only

This directory contains **only the Python framework** that users install via pip.

### What's Implemented:
- ✅ Core testLLM Python library
- ✅ pytest integration  
- ✅ YAML test definitions
- ✅ Assertion library (semantic testing)
- ✅ Input/output focused testing (works with any agent)
- ✅ Freemium license with data collection rights
- ✅ Telemetry system (ready for Supabase integration)

### Framework Structure:
```
testLLM/
├── testllm/
│   ├── __init__.py          # Main exports
│   ├── core.py              # AgentUnderTest, ConversationTest, etc
│   ├── assertions.py        # All assertion types
│   ├── pytest_plugin.py    # pytest integration
│   ├── telemetry.py         # Data collection (sends to Supabase)
│   └── reporting.py         # HTML/JSON reports
├── examples/
│   ├── basic_greeting.yaml  # Example test files
│   ├── weather_query.yaml
│   └── test_agent.py        # Example pytest usage
├── setup.py                 # Package configuration
├── LICENSE                  # Custom freemium license
└── README.md               # Framework documentation
```

## Next Phase: Parent Directory Structure

When you reopen in the parent directory, create this structure:

```
wayy-research/
├── testLLM/                    # (Current - Framework only)
├── testllm-dashboard/          # React frontend
│   ├── src/
│   ├── components/
│   ├── pages/
│   ├── package.json
│   └── README.md
├── testllm-backend/            # Supabase configuration
│   ├── supabase/
│   ├── migrations/
│   ├── functions/
│   └── schema.sql
└── docs/                       # Shared documentation
```

## 2. React Frontend (`testllm-dashboard/`)

**Purpose**: $5/month premium dashboard for test inspection

### Features to Build:
- **Authentication**: Login/signup with subscription management
- **Test History**: View all test executions for a user
- **Test Details**: Drill down into inputs, outputs, assertions
- **Analytics**: Test performance trends, failure patterns
- **Export**: Download test data in various formats
- **Settings**: Manage subscription, data retention preferences

### Key Pages:
- `/login` - Authentication
- `/dashboard` - Overview with recent tests
- `/tests` - Paginated test history
- `/tests/:id` - Detailed test inspection
- `/analytics` - Charts and trends
- `/settings` - Account management

### Tech Stack:
- React 18+ with TypeScript
- Supabase Auth for authentication
- Supabase JS client for database
- Tailwind CSS for styling
- Recharts for analytics
- React Query for data fetching

## 3. Supabase Backend (`testllm-backend/`)

**Purpose**: PostgreSQL database to store all test data

### Database Schema:
```sql
-- Users and subscriptions
users (
  id uuid primary key,
  email text unique,
  subscription_status text, -- 'free', 'premium', 'cancelled'
  subscription_expires_at timestamp,
  created_at timestamp
)

-- Test executions
test_executions (
  id uuid primary key,
  user_id uuid references users(id),
  test_id text,
  description text,
  passed boolean,
  execution_time float,
  created_at timestamp,
  framework_version text,
  platform text
)

-- Test conversations  
test_conversations (
  id uuid primary key,
  execution_id uuid references test_executions(id),
  conversation_name text,
  turn_order integer,
  role text, -- 'user' or 'agent'
  content text,
  created_at timestamp
)

-- Assertions
test_assertions (
  id uuid primary key,
  conversation_id uuid references test_conversations(id),
  assertion_type text,
  passed boolean,
  expected_value text,
  actual_value_hash text, -- Privacy: only store hash
  message text,
  created_at timestamp
)
```

### API Endpoints (Supabase Functions):
- `POST /api/telemetry` - Receive data from framework
- `GET /api/tests` - List user's tests (premium only)
- `GET /api/tests/:id` - Get test details (premium only)
- `DELETE /api/tests/:id` - Delete test (premium only)
- `POST /api/subscribe` - Handle subscription payments

## Data Flow

```
testLLM Framework (user's machine)
    ↓ (telemetry via HTTPS)
Supabase Database
    ↓ (query via API)
React Dashboard (premium users)
```

## Business Model

### Free Tier:
- Use framework unlimited
- All test data collected automatically
- No access to historical data
- Basic local HTML reports only

### Premium Tier ($5/month):
- Everything in free tier
- Web dashboard access to all historical test data
- Advanced analytics and trends
- Data export capabilities
- Priority support

## Integration Points

### Framework → Supabase:
- Telemetry system in `telemetry.py` sends to Supabase Edge Functions
- Authentication via API key embedded in framework
- Automatic data collection on every test run

### Supabase → React:
- Supabase Auth for user management
- Real-time subscriptions for live updates
- Row Level Security (RLS) to ensure users only see their data

## Next Steps (When Reopened):

1. **Test the current framework** in this directory first
2. **Create parent directory structure**
3. **Set up Supabase project** and database schema
4. **Build React dashboard** with authentication
5. **Connect framework telemetry** to Supabase endpoints
6. **Add subscription management** (Stripe integration)
7. **Deploy and test** end-to-end flow

## Testing the Framework

Before moving to parent directory, test this framework:

```bash
cd /home/rcgalbo/wayy-research/testLLM
pip install -e .
pytest --testllm examples/test_agent.py -v
```

This should validate that the core framework works before building the full stack.