"use client";
import { useState } from "react";
import type { ResultData } from "./emotion";
import type { Dispatch, SetStateAction } from "react";

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:8000";

type UploadProps = {
  onResult: Dispatch<SetStateAction<ResultData | undefined>>;
};

export default function Upload({ onResult }: UploadProps) {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setError(null);
    setFile(event.target.files?.[0] ?? null);
  };

  const uploadAndPredict = async () => {
    if (!file) {
      setError("Please choose an image first.");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(`${BACKEND_URL}/predict`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const text = await response.text();
        throw new Error(`Backend error ${response.status}: ${text}`);
      }

      const result = (await response.json()) as ResultData;
      onResult(result);
    } catch (err) {
      if (err instanceof Error) {
        setError(err.message);
      } else {
        setError("Prediction failed.");
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col gap-4">
      <div className="border-2 border-dashed rounded-xl p-10 text-center bg-gray-50">
        <div className="w-10 h-10 mx-auto mb-3 rounded-full border flex items-center justify-center">
          ⬆️
        </div>

        <p className="text-sm font-medium">Upload an image file</p>
        <p className="text-xs text-gray-500 mb-4">JPG, PNG supported</p>

        <input
          id="upload-file"
          type="file"
          accept="image/png,image/jpeg"
          className="hidden"
          onChange={handleFileChange}
        />
        <label
          htmlFor="upload-file"
          className="inline-flex items-center justify-center border px-4 py-1.5 rounded-md text-xs cursor-pointer hover:bg-gray-100"
        >
          Browse files
        </label>
        {file && <p className="text-xs text-gray-500 mt-3">{file.name}</p>}
      </div>

      <div className="bg-blue-50 border border-blue-200 p-3 rounded-md text-xs text-blue-700">
        Upload an image and the backend will predict the emotion.
      </div>

      <button
        onClick={uploadAndPredict}
        disabled={!file || loading}
        className="bg-blue-500 text-white py-2 rounded-md text-sm disabled:opacity-60"
      >
        {loading ? "Analysing…" : "Analyse →"}
      </button>

      {error && (
        <p className="text-xs text-red-500 bg-red-50 border border-red-100 rounded-md px-3 py-2">
          {error}
        </p>
      )}
    </div>
  );
}
