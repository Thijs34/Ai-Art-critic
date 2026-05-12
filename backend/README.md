---
title: Ai Art Critic
emoji: 🎨
colorFrom: yellow
colorTo: red
sdk: docker
pinned: false
---

## Conversational artwork chat

The backend exposes `POST /api/chat` for multi-turn discussion grounded in the current artwork analysis.

Request body:

```json
{
  "message": "How does the composition affect the mood?",
  "analysis": { "style": { "label": "Impressionism", "confidence": 82 } },
  "session_id": "optional-existing-session"
}
```

The chat layer builds a compact artwork profile, retrieves small style/technique/ArtEmis references, and keeps short in-memory session history. ArtEmis snippets are used as emotional parallels, not factual claims about the uploaded artwork. Set `OPENAI_API_KEY` to enable LLM responses. Optional knobs:

- `ART_CHAT_MODEL`, default `gpt-4o-mini`
- `ARTEMIS_RAG_MAX_ROWS`, default `6000`
- `ART_RAG_USE_EMBEDDINGS`, default `false`; set to `true` to rerank retrieved notes with OpenAI embeddings
- `ART_RAG_EMBED_MODEL`, default `text-embedding-3-small`
- `ART_RAG_EMBED_CACHE_MAX`, default `3000`; cached embeddings are stored lazily in `retrieval_index/art_chat_embedding_cache.json`
