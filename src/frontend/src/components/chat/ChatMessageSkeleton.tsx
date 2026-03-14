/**
 * Informity AI — Chat message skeleton for loading state
 */
import { Skeleton } from '../Skeleton'
import './ChatMessageSkeleton.css'

interface ChatMessageSkeletonProps {
  role?: 'user' | 'assistant'
}

export function ChatMessageSkeleton({ role = 'assistant' }: ChatMessageSkeletonProps) {
  return (
    <div className={`chat-message-skeleton chat-message-skeleton--${role}`}>
      <div className="chat-message-skeleton__content">
        <Skeleton width="80%" height={14} />
        <Skeleton width="60%" height={14} />
        <Skeleton width="90%" height={14} />
      </div>
    </div>
  )
}
