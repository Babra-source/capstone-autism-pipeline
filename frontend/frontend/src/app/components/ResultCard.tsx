import { EmotionBars } from "./EmotionsBar";
import { EyeFeatures } from "./EyeFeatures";
import type { ResultData } from "./emotion";

type Props = {
  title: string;
  data?: ResultData;
};

export default function ResultCard({ title, data }: Props) {
  if (!data) {
    return (
      <div className="border rounded-xl p-4 text-sm text-gray-400">
        {title}: No data
      </div>
    );
  }

  return (
    <div className="border rounded-xl p-4 flex flex-col gap-3">
      
      {/* Title */}
      <p className="text-xs uppercase tracking-wider text-gray-400">
        {title}
      </p>

      {/* Top prediction */}
      <div className="flex items-center gap-2">
        <span className="px-2 py-0.5 rounded-full bg-green-100 text-green-700 text-xs">
          {data.top_emotion}
        </span>
        <span className="text-lg font-medium">
          {data.confidence.toFixed(1)}%
        </span>
      </div>

      {/* Emotion bars */}
      <EmotionBars emotions={data.emotions} />

      {/* Eye features (optional) */}
      {data.eye_features && (
        <>
          <div className="border-t my-2"></div>
          <EyeFeatures data={data.eye_features} />
        </>
      )}
    </div>
  );
}