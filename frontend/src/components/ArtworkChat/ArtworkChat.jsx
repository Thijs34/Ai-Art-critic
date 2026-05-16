import { useEffect, useRef, useState } from 'react'
import './ArtworkChat.css'

const API_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:5000'

const STARTERS = [
  'What should I notice first?',
  'How does the composition affect the mood?',
  'Talk to me about the brushwork.',
]

function createMessageId() {
  return globalThis.crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random().toString(16).slice(2)}`
}

function renderInlineFormatting(text, keyPrefix) {
  return text.split(/(\*\*[^*]+\*\*|\*[^*\n]+\*)/g).map((part, index) => {
    const key = `${keyPrefix}-inline-${index}`

    if (part.startsWith('**') && part.endsWith('**')) {
      return <strong key={key}>{part.slice(2, -2)}</strong>
    }

    if (part.startsWith('*') && part.endsWith('*')) {
      return <em key={key}>{part.slice(1, -1)}</em>
    }

    return part
  })
}

function renderFormattedMessage(content) {
  const blocks = String(content || '')
    .replace(/\r\n/g, '\n')
    .split(/\n{2,}/)
    .map((block) => block.trim())
    .filter(Boolean)

  if (blocks.length === 0) return null

  return blocks.map((block, blockIndex) => {
    const lines = block.split('\n').map((line) => line.trim()).filter(Boolean)
    const unorderedItems = lines
      .map((line) => line.match(/^[-*]\s+(.+)$/))
      .filter(Boolean)
    const orderedItems = lines
      .map((line) => line.match(/^\d+[.)]\s+(.+)$/))
      .filter(Boolean)

    if (unorderedItems.length === lines.length) {
      return (
        <ul key={`block-${blockIndex}`} className="artwork-chat__list">
          {unorderedItems.map((match, itemIndex) => (
            <li key={`block-${blockIndex}-item-${itemIndex}`}>
              {renderInlineFormatting(match[1], `block-${blockIndex}-item-${itemIndex}`)}
            </li>
          ))}
        </ul>
      )
    }

    if (orderedItems.length === lines.length) {
      return (
        <ol key={`block-${blockIndex}`} className="artwork-chat__list">
          {orderedItems.map((match, itemIndex) => (
            <li key={`block-${blockIndex}-item-${itemIndex}`}>
              {renderInlineFormatting(match[1], `block-${blockIndex}-item-${itemIndex}`)}
            </li>
          ))}
        </ol>
      )
    }

    return (
      <p key={`block-${blockIndex}`} className="artwork-chat__paragraph">
        {lines.map((line, lineIndex) => (
          <span key={`block-${blockIndex}-line-${lineIndex}`}>
            {lineIndex > 0 && <br />}
            {renderInlineFormatting(line, `block-${blockIndex}-line-${lineIndex}`)}
          </span>
        ))}
      </p>
    )
  })
}

export default function ArtworkChat({ analysis }) {
  const [messages, setMessages] = useState([
    {
      id: createMessageId(),
      role: 'assistant',
      content: 'Ask me about the mood, brushwork, composition, symbolism, technique, or historical context of this piece.',
      visibleContent: 'Ask me about the mood, brushwork, composition, symbolism, technique, or historical context of this piece.',
    },
  ])
  const [input, setInput] = useState('')
  const [sessionId, setSessionId] = useState(null)
  const [isSending, setIsSending] = useState(false)
  const [isFullscreen, setIsFullscreen] = useState(false)
  const [error, setError] = useState(null)
  const inputRef = useRef(null)
  const messagesRef = useRef(null)

  useEffect(() => {
    const streamingMessage = messages.find((message) => (
      message.role === 'assistant'
      && message.isStreaming
      && !message.isPending
      && message.visibleContent !== message.content
    ))

    if (!streamingMessage) return undefined

    const timeout = window.setTimeout(() => {
      setMessages((prev) => prev.map((message) => {
        if (message.id !== streamingMessage.id) return message

        const currentLength = message.visibleContent?.length ?? 0
        const remaining = message.content.slice(currentLength)
        const nextBreak = remaining.search(/[\s.,;:!?)]/)
        const chunkSize = nextBreak >= 0 ? Math.max(nextBreak + 1, 2) : Math.min(remaining.length, 4)
        const nextContent = message.content.slice(0, currentLength + chunkSize)

        return {
          ...message,
          visibleContent: nextContent,
          isStreaming: nextContent.length < message.content.length,
        }
      }))
    }, 28)

    return () => window.clearTimeout(timeout)
  }, [messages])

  useEffect(() => {
    const messageBox = messagesRef.current
    if (!messageBox) return

    messageBox.scrollTo({
      top: messageBox.scrollHeight,
      behavior: 'smooth',
    })
  }, [messages])

  useEffect(() => {
    document.body.classList.toggle('artwork-chat-body--fullscreen', isFullscreen)

    return () => {
      document.body.classList.remove('artwork-chat-body--fullscreen')
    }
  }, [isFullscreen])

  useEffect(() => {
    function handleKeyDown(event) {
      if (event.key === 'Escape') {
        setIsFullscreen(false)
      }
    }

    if (isFullscreen) {
      window.addEventListener('keydown', handleKeyDown)
    }

    return () => {
      window.removeEventListener('keydown', handleKeyDown)
    }
  }, [isFullscreen])

  async function sendMessage(text) {
    const message = text.trim()
    if (!message || isSending) return

    const assistantMessageId = createMessageId()

    setMessages((prev) => [
      ...prev,
      {
        id: createMessageId(),
        role: 'user',
        content: message,
        visibleContent: message,
      },
      {
        id: assistantMessageId,
        role: 'assistant',
        content: '',
        visibleContent: '',
        isPending: true,
        isStreaming: true,
      },
    ])
    setInput('')
    setError(null)
    setIsSending(true)

    try {
      const res = await fetch(`${API_URL}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message,
          artwork_id: analysis.artworkId,
          session_id: sessionId,
        }),
      })

      if (!res.ok) {
        const err = await res.json().catch(() => ({}))
        throw new Error(err.error || `Server error ${res.status}`)
      }

      const data = await res.json()
      setSessionId(data.session_id)
      setMessages((prev) => prev.map((chatMessage) => {
        if (chatMessage.id !== assistantMessageId) return chatMessage

        return {
          ...chatMessage,
          content: data.answer,
          visibleContent: '',
          references: data.references ?? [],
          isPending: false,
          isStreaming: true,
        }
      }))
    } catch (err) {
      console.error('Chat failed:', err)
      setError(err.message)
      setMessages((prev) => prev.map((chatMessage) => {
        if (chatMessage.id !== assistantMessageId) return chatMessage

        const fallback = 'I could not reach the critic voice right now, but the analysis above is still available.'
        return {
          ...chatMessage,
          content: fallback,
          visibleContent: fallback,
          isPending: false,
          isStreaming: false,
        }
      }))
    } finally {
      setIsSending(false)
      inputRef.current?.focus()
    }
  }

  function handleSubmit(event) {
    event.preventDefault()
    sendMessage(input)
  }

  return (
    <section className={`artwork-chat${isFullscreen ? ' artwork-chat--fullscreen' : ''}`} aria-label="Artwork conversation">
      <div className="artwork-chat__header">
        <div>
          <h3 className="artwork-chat__title">Discuss the Artwork</h3>
          <p className="artwork-chat__sub">Grounded in this analysis</p>
        </div>
        <div className="artwork-chat__header-actions">
          <div className="artwork-chat__status" aria-hidden="true">
            {isSending ? 'Live AI' : 'Ready'}
          </div>
          <button
            className={`artwork-chat__fullscreen-toggle${isFullscreen ? ' artwork-chat__fullscreen-toggle--active' : ''}`}
            type="button"
            onClick={() => setIsFullscreen((current) => !current)}
            aria-label={isFullscreen ? 'Exit fullscreen chat' : 'Open fullscreen chat'}
            title={isFullscreen ? 'Exit fullscreen' : 'Fullscreen'}
          >
            <span className="artwork-chat__fullscreen-icon" aria-hidden="true" />
          </button>
        </div>
      </div>

      <div ref={messagesRef} className="artwork-chat__messages" aria-live="polite">
        {messages.map((message, index) => (
          <div
            key={message.id ?? `${message.role}-${index}`}
            className={`artwork-chat__message artwork-chat__message--${message.role}${message.isStreaming ? ' artwork-chat__message--live' : ''}`}
          >
            <div className="artwork-chat__content">
              {message.isPending ? (
                <span className="artwork-chat__typing" aria-label="AI is thinking">
                  <span />
                  <span />
                  <span />
                </span>
              ) : (
                <>
                  {renderFormattedMessage(message.visibleContent ?? message.content)}
                  {message.isStreaming && <span className="artwork-chat__cursor" aria-hidden="true" />}
                </>
              )}
            </div>
            {!message.isStreaming && message.references?.length > 0 && (
              <div className="artwork-chat__refs">
                {message.references.slice(0, 3).map((ref) => (
                  <span key={`${ref.source}-${ref.title}`}>
                    {ref.type ? `${ref.type}: ` : ''}{ref.title}
                  </span>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>

      <div className="artwork-chat__starters" aria-label="Suggested questions">
        {STARTERS.map((starter) => (
          <button
            key={starter}
            type="button"
            className="artwork-chat__starter"
            onClick={() => sendMessage(starter)}
            disabled={isSending}
          >
            {starter}
          </button>
        ))}
      </div>

      <form className="artwork-chat__form" onSubmit={handleSubmit}>
        <input
          ref={inputRef}
          className="artwork-chat__input"
          value={input}
          onChange={(event) => setInput(event.target.value)}
          placeholder="Ask about mood, technique, symbols..."
          disabled={isSending}
        />
        <button className="artwork-chat__send" type="submit" disabled={!input.trim() || isSending}>
          Send
        </button>
      </form>
      {error && <p className="artwork-chat__error">{error}</p>}
    </section>
  )
}
