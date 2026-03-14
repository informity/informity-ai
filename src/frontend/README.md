# Informity AI — React + Vite Frontend

React 19 + Vite 7 frontend for Informity AI. The backend serves the built output from `dist/` when running.

## Quick Start

```bash
# Build for production (output: dist/)
npm run build

# Or from project root:
make frontend-build
```

Then run the backend: `make run` or `make dev`. Open http://127.0.0.1:8420.

## Development

```bash
# Vite dev server with hot reload (port 5173)
npm run dev

# Or from project root:
make frontend
```

Run the backend separately in another terminal: `make run` or `make dev`. The Vite proxy forwards `/api/*` to http://localhost:8420.

## Build and run (single command)

From project root:

```bash
make app   # Builds frontend + runs backend
```

## Structure

- `src/` — React components, pages, API client, utilities
- `src/pages/` — Route pages (ChatPage, FilesPage, DashboardPage, etc.)
- `src/components/` — Reusable components (Layout, Sidebar, ChatView, SourceCard, etc.)
- `src/context/` — React contexts: ChatProvider/useChatContext, ToastProvider/useToast, ConfirmProvider/useConfirm
- `src/styles/shared/` — Shared CSS (filters, tables)
- `dist/` — Build output (gitignored; served by FastAPI when present)
