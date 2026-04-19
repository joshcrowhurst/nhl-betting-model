export interface Prediction {
  id: number;
  game_id: number;
  game_date: string;
  season: string | null;
  game_type: number | null;
  home_team: string;
  away_team: string;
  home_win_prob: number;
  away_win_prob: number;
  predicted_winner: string | null;
  market_home_prob: number | null;
  home_odds: number | null;
  away_odds: number | null;
  home_ev: number | null;
  away_ev: number | null;
  is_value_bet: boolean | null;
  value_team: string | null;
  value_odds: number | null;
  value_ev: number | null;
  actual_home_win: number | null;
  correct: boolean | null;
  value_bet_correct: boolean | null;
}

export interface DailyBreakdown {
  date: string;
  correct: number;
  total: number;
  vb_correct: number;
  vb_total: number;
}

export interface Performance {
  total_predictions: number;
  resolved_predictions: number;
  correct_predictions: number;
  accuracy: number | null;
  total_value_bets: number;
  value_bets_correct: number;
  value_bet_accuracy: number | null;
  value_bet_roi: number | null;
  total_ev: number | null;
  daily_breakdown: DailyBreakdown[] | null;
  computed_at: string | null;
}

export interface ModelRun {
  id: number;
  run_type: string;
  status: string | null;
  games_processed: number | null;
  started_at: string | null;
  completed_at: string | null;
  details: Record<string, unknown> | null;
}
