"use client";
import Image from "next/image";
import Link from "next/link";
import Upload from "./components/Upload";
import Webcam from "./components/Webcam";
import ResultCard from "./components/ResultCard";
import { useState } from "react";
import type { ResultData } from "./components/emotion";

export default function Home() {
  const [result, setResult] = useState<ResultData | undefined>(undefined);

  return (
    <>
      {/* Navbar */}
      <div className="flex w-full h-[50px] items-center px-6 bg-blue-500">
        <div className="px-5 py-2 rounded-full text-sm font-medium bg-gradient-to-br from-[#f9c5c2] to-[#f4a09d] text-[#7a2e2b]">
          Autism Emotion Detection
        </div>
        <div className="flex flex-1 justify-center gap-6">
          <Link className="text-white hover:underline" href="/">Home</Link>
          <Link className="text-white hover:underline" href="/analyze">Analyze</Link>
          <Link className="text-white hover:underline" href="/results">Results</Link>
        </div>
      </div>

      {/* Hero Section */}
      <div className="flex flex-col items-center text-center px-6 pt-16 pb-12">
        <span className="text-xs font-mono uppercase tracking-widest text-blue-400 mb-4">
          Research Demo · ASD FER System
        </span>
        <h1 className="text-4xl font-bold text-gray-900 mb-4 leading-tight">
          Detect Emotions from Videos
        </h1>
        <p className="text-gray-500 max-w-lg text-base mb-8">
          A multimodal system that leverages Facial Expression Recognition
          and Eye Tracking to detect the emotions of people with Autism.
        </p>
      </div>

      {/* Button + Image side by side */}
      <div className="flex flex-col md:flex-row items-center justify-center gap-10 px-10 pb-16">
        <Link href="#demo-section">
          <div className="rounded-full bg-black w-[280px] h-[50px] flex items-center justify-center text-white text-sm font-medium cursor-pointer hover:bg-gray-800 transition">
            Try demo with sample videos →
          </div>
        </Link>
        <Image
          src="/Autismimage.jpg"
          alt="Autism Image"
          width={420}
          height={280}
          className="rounded-2xl shadow-md object-cover"
        />
      </div>

      {/* How it works */}
      <div className="bg-blue-50 border-t border-gray-100 px-6 py-14">
        <div className="max-w-4xl mx-auto text-center mb-10">
          <p className="text-xs font-mono uppercase tracking-widest text-gray-400 mb-2">Pipeline</p>
          <h2 className="text-2xl font-bold text-gray-800">How It Works</h2>
        </div>
        <div className="max-w-4xl mx-auto grid grid-cols-1 sm:grid-cols-4 gap-6">
          {[
            { icon: '📹', title: '1. Input',    desc: 'Upload an image, video, or capture a webcam frame.' },
            { icon: '👁️', title: '2. Eye Tracking', desc: 'MediaPipe extracts gaze direction, blink rate, and EAR features.' },
            { icon: '😊', title: '3. FER',       desc: 'Facial landmarks detect and classify expressions.' },
            { icon: '🧠', title: '4. Prediction', desc: 'FER model classifies the emotion with a confidence score.' },
          ].map(({ icon, title, desc }) => (
            <div key={title} className="bg-white rounded-xl border border-gray-100 p-6 text-center shadow-sm">
              <div className="text-3xl mb-3">{icon}</div>
              <h3 className="font-semibold text-gray-800 mb-2">{title}</h3>
              <p className="text-sm text-gray-500 leading-relaxed">{desc}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Demo section */}
      <div id="demo-section" className="flex justify-center items-center mt-10 px-6">
        <div className="flex flex-wrap justify-center gap-10 mt-10 px-6">
          <Upload onResult={setResult} />
          <Webcam onResult={setResult} />
        </div>
      </div>

      {/* Result cards — only show after a prediction */}
      {result && (
        <div className="max-w-4xl mx-auto px-6 py-10 grid grid-cols-1 sm:grid-cols-3 gap-6">
          <ResultCard title="FER" data={result} />
          <ResultCard title="Eye Tracking" data={result} />
          <ResultCard title="Combined" data={result} />
        </div>
      )}

      {/* Footer */}
      <div className="border-t border-gray-200 mt-10 py-6 text-center text-xs text-gray-400 font-mono">
        AutismEmotionScan · FER + Eye Tracking Research Demo
      </div>
    </>
  );
}