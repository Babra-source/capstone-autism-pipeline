import type { ResultData } from "./emotion";
import type { Dispatch, SetStateAction } from "react";

type UploadProps = {
  onResult: Dispatch<SetStateAction<ResultData | undefined>>;
};

export default function Upload({onResult}: UploadProps) {
  return (
    <div className="flex flex-col gap-4">
      <div className="border-2 border-dashed rounded-xl p-10 text-center bg-gray-50">
        <div className="w-10 h-10 mx-auto mb-3 rounded-full border flex items-center justify-center">
          ⬆️
        </div>

        <p className="text-sm font-medium">Drop image or video here</p>
        <p className="text-xs text-gray-500 mb-4">
          JPG, PNG, MP4 supported
        </p>

        <button className="border px-4 py-1.5 rounded-md text-xs">
          Browse files
        </button>
      </div>

      <div className="bg-blue-50 border border-blue-200 p-3 rounded-md text-xs text-blue-700">
        If you upload a video, only a short snapshot of frames will be analysed.
      </div>

      <button className="bg-blue-500 text-white py-2 rounded-md text-sm">
        Analyse →
      </button>
    </div>
  );
}