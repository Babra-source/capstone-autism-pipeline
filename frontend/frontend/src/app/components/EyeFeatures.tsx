import type { EyeFeaturesData } from "./emotion";

export function EyeFeatures({ data }: { data: EyeFeaturesData }) {
  return (
    <div className="grid grid-cols-2 gap-2 text-xs">
      <Feature label="EAR L" value={data.ear_left} />
      <Feature label="EAR R" value={data.ear_right} />
      <Feature label="Gaze" value={data.gaze} />
      <Feature label="Blink" value={data.blink ? "Yes" : "No"} />
    </div>
  );
}

function Feature({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="bg-gray-50 p-2 rounded">
      <p className="text-gray-400">{label}</p>
      <p className="font-mono">{value}</p>
    </div>
  );
}