export interface AgentState {
  step: number;
  operation: 'add' | 'remove' | 'noop';
  reward: number;
  memory_count: number;
  new_message: string;
  memory: string;
  done: boolean;
  timestamp: string;
  task_score: number;
  correct_in_memory: number;
  total_relevant_seen: number;
  memory_capacity: number;
  task_name: string;
}

export interface TrainingLog {
  timestamp: string;
  step?: number;
  loss?: number;
  reward: number;
  fmt_reward?: number;
  env_reward?: number;
  episode: number;
  operation?: 'add' | 'remove' | 'noop';
  memory_count?: number;
}
