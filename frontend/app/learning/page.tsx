import EpisodeJournal from "./episode-journal";
import CheckpointEvaluation from "./checkpoint-evaluation";
import "./learning-report.css";

export default function LearningPage() {
  return <div className="research-page">
    <EpisodeJournal />
    <details className="panel learning-evaluation-section">
      <summary>Held-out evaluation · did performance improve?</summary>
      <CheckpointEvaluation />
    </details>
  </div>;
}
