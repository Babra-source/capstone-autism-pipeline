import type { Emotion } from "./emotion";

const colors: Record<string, string> = {
  Happy: "bg-green-600",
  Neutral: "bg-gray-400",
  Sad: "bg-blue-500",
  Surprise: "bg-yellow-500",
  Angry: "bg-red-500",
};

export function EmotionBars({ emotions }: { emotions: Emotion[] }) {
  return (
    <div className="space-y-1">
      {emotions.map((e) => (
        <div key={e.label} className="flex items-center gap-2">
          <span className="w-12 text-xs text-gray-500">
            {e.label}
          </span>

          <div className="flex-1 h-1.5 bg-gray-100 rounded">
            <div
              className={`${colors[e.label] || "bg-gray-300"} h-1.5 rounded`}
              style={{ width: `${e.value}%` }}
            />
          </div>

          <span className="text-xs text-gray-500 w-8 text-right">
            {e.value}%
          </span>
        </div>
      ))}
    </div>
  );
}