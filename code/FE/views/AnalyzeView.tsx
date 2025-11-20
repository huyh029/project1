import React, { useState } from 'react';
import { WebcamCapture } from '../components/WebcamCapture';
import { apiService } from '../services/api';
import { AnalyzeResponse, AnalysisResult } from '../types';
import { EMOTION_COLORS } from '../constants';
import { Loader2, ScanFace, AlertCircle, User, Smile, Users, Fingerprint, Upload } from 'lucide-react';

export const AnalyzeView: React.FC = () => {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [useUpload, setUseUpload] = useState(false);
  const [uploadedImage, setUploadedImage] = useState<string>('');

  const handleCapture = async (img: string) => {
    if (!img) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const data = await apiService.analyze(img);
      setResult(data);
    } catch (err: any) {
      setError(err.message || "Analysis failed");
    } finally {
      setLoading(false);
    }
  };

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        const base64String = reader.result as string;
        const base64Data = base64String.split(',')[1];
        setUploadedImage(base64Data);
        handleCapture(base64Data);
      };
      reader.readAsDataURL(file);
    }
  };

  // Helper to safely access list or object from DeepFace result
  const analysis: AnalysisResult | undefined = result?.result?.[0];

  // Helper to sort all attributes (no limit)
  const getSortedAttributes = (data: Record<string, number> | undefined) => {
    if (!data) return [];
    return Object.entries(data)
      .sort(([, a], [, b]) => b - a); // Sort all, no slice
  };

  return (
    <div className="w-full max-w-6xl mx-auto p-4">
      <div className="text-center mb-10 space-y-2">
        <h2 className="text-3xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-blue-500 flex items-center justify-center gap-3">
          <ScanFace className="text-cyan-400 w-8 h-8" />
          AI Identity Analysis
        </h2>
        <p className="text-slate-400 text-sm max-w-lg mx-auto">
          Advanced biometric profiling utilizing Deep Learning to estimate age, gender, emotional state, and ethnicity from a single frame.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Left Column: Input */}
        <div className="lg:col-span-5 space-y-6">
          <div className="bg-slate-900/50 backdrop-blur-md p-1 rounded-2xl border border-slate-700/50 shadow-2xl ring-1 ring-white/5">
            <div className="bg-slate-900 rounded-xl p-4 border border-slate-800">
              <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-4 flex items-center gap-2">
                <Fingerprint className="w-4 h-4" /> Source Feed
              </h3>
              
              {/* Toggle between Webcam and Upload */}
              <div className="flex gap-2 mb-4">
                <button
                  onClick={() => { setUseUpload(false); setUploadedImage(''); }}
                  className={`flex-1 flex items-center justify-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                    !useUpload 
                      ? 'bg-indigo-600 text-white shadow-lg' 
                      : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
                  }`}
                >
                  <ScanFace size={16} />
                  Webcam
                </button>
                <button
                  onClick={() => { setUseUpload(true); setUploadedImage(''); }}
                  className={`flex-1 flex items-center justify-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                    useUpload 
                      ? 'bg-indigo-600 text-white shadow-lg' 
                      : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
                  }`}
                >
                  <Upload size={16} />
                  Upload
                </button>
              </div>

              {useUpload ? (
                // Upload Image Section
                <div className="space-y-3">
                  <div className="relative">
                    <input
                      type="file"
                      accept="image/*"
                      onChange={handleImageUpload}
                      className="hidden"
                      id="analyze-image-upload"
                    />
                    <label
                      htmlFor="analyze-image-upload"
                      className="flex flex-col items-center justify-center w-full h-64 border-2 border-dashed border-slate-600 rounded-xl cursor-pointer hover:border-indigo-500 hover:bg-slate-800/50 transition-all group"
                    >
                      {uploadedImage ? (
                        <div className="relative w-full h-full">
                          <img
                            src={`data:image/png;base64,${uploadedImage}`}
                            alt="Uploaded"
                            className="w-full h-full object-cover rounded-xl"
                          />
                          <div className="absolute top-2 right-2 bg-emerald-500 text-white px-3 py-1 rounded-full text-xs font-bold">
                            ✓ Uploaded
                          </div>
                        </div>
                      ) : (
                        <div className="flex flex-col items-center text-slate-400 group-hover:text-slate-300 transition-colors">
                          <Upload size={48} className="mb-3" />
                          <p className="text-sm font-medium">Click to upload image</p>
                          <p className="text-xs mt-1">PNG, JPG up to 10MB</p>
                        </div>
                      )}
                    </label>
                  </div>
                  
                  {uploadedImage && (
                    <button
                      onClick={() => setUploadedImage('')}
                      className="w-full bg-slate-700 hover:bg-slate-600 text-white font-medium py-2 rounded-lg transition-all"
                    >
                      Remove Image
                    </button>
                  )}
                </div>
              ) : (
                // Webcam Capture Section
                <WebcamCapture onCapture={handleCapture} label="Scan Face Now" />
              )}
            </div>
          </div>
          
          {/* Guide / Tips */}
          {!analysis && !loading && (
             <div className="bg-blue-900/10 border border-blue-500/20 p-4 rounded-xl text-sm text-blue-200/70">
               <p className="flex items-start gap-2">
                 <AlertCircle className="w-4 h-4 mt-0.5 flex-shrink-0" />
                 Ensure good lighting and look directly at the camera for the most accurate biometric estimation.
               </p>
             </div>
          )}
        </div>

        {/* Right Column: Results Dashboard */}
        <div className="lg:col-span-7 flex flex-col">
          {loading && (
            <div className="flex-1 h-full min-h-[400px] flex flex-col items-center justify-center bg-slate-800/30 rounded-2xl border border-slate-700/50 backdrop-blur-sm">
              <div className="relative">
                <div className="absolute inset-0 bg-indigo-500 blur-xl opacity-20 rounded-full animate-pulse"></div>
                <Loader2 className="animate-spin w-16 h-16 text-indigo-400 relative z-10" />
              </div>
              <p className="mt-6 text-lg font-medium text-indigo-300 animate-pulse">Analyzing facial micro-features...</p>
              <p className="text-slate-500 text-sm">Processing neural network layers</p>
            </div>
          )}

          {error && (
            <div className="flex-1 flex flex-col items-center justify-center bg-red-950/20 rounded-2xl border border-red-500/30 p-8 text-center">
              <div className="bg-red-500/10 p-4 rounded-full mb-4">
                <AlertCircle className="w-10 h-10 text-red-400" />
              </div>
              <h3 className="text-xl font-bold text-red-200 mb-2">Analysis Failed</h3>
              <p className="text-red-300/70">{error}</p>
            </div>
          )}

          {!loading && !error && !analysis && (
            <div className="flex-1 flex flex-col items-center justify-center bg-slate-800/30 rounded-2xl border border-slate-700/50 border-dashed min-h-[400px]">
              <ScanFace className="w-16 h-16 text-slate-600 mb-4" />
              <p className="text-slate-500 font-medium">Waiting for capture...</p>
            </div>
          )}

          {!loading && analysis && (
            <div className="space-y-4 animate-in slide-in-from-bottom-6 fade-in duration-700">
              
              {/* Top Row: Age & Gender */}
              <div className="grid grid-cols-2 gap-4">
                {/* Age Card */}
                <div className="bg-gradient-to-br from-slate-800 to-slate-900 p-6 rounded-2xl border border-slate-700 shadow-lg relative overflow-hidden group">
                  <div className="absolute top-0 right-0 p-3 opacity-10 group-hover:opacity-20 transition-opacity">
                    <Fingerprint className="w-24 h-24" />
                  </div>
                  <p className="text-slate-400 text-xs font-bold uppercase tracking-widest mb-1">Estimated Age</p>
                  <div className="flex items-baseline gap-2">
                    <span className="text-5xl font-black text-white tracking-tighter">
                      {analysis.age}
                    </span>
                    <span className="text-sm text-slate-500">years old</span>
                  </div>
                  <div className="mt-4 h-1.5 w-full bg-slate-700/50 rounded-full overflow-hidden">
                    <div className="h-full bg-indigo-500 rounded-full" style={{ width: `${Math.min(analysis.age, 100)}%` }}></div>
                  </div>
                </div>

                {/* Gender Card */}
                <div className="bg-gradient-to-br from-slate-800 to-slate-900 p-6 rounded-2xl border border-slate-700 shadow-lg">
                  <div className="flex items-center justify-between mb-4">
                    <p className="text-slate-400 text-xs font-bold uppercase tracking-widest">Gender</p>
                    <User className={`w-5 h-5 ${analysis.dominant_gender === 'Man' ? 'text-blue-400' : 'text-pink-400'}`} />
                  </div>
                  <p className="text-3xl font-bold text-white mb-3 capitalize">{analysis.dominant_gender}</p>
                  
                  {/* Gender Distribution */}
                  <div className="space-y-2">
                     {Object.entries(analysis.gender).map(([gender, score]) => (
                       <div key={gender} className="flex items-center gap-3 text-xs">
                         <span className="w-12 text-slate-400 capitalize">{gender}</span>
                         <div className="flex-1 h-2 bg-slate-700 rounded-full overflow-hidden">
                            <div 
                              className={`h-full rounded-full ${gender === 'Man' ? 'bg-blue-500' : 'bg-pink-500'}`} 
                              style={{ width: `${score}%` }}
                            />
                         </div>
                         <span className="w-8 text-right text-slate-300">{Math.round(score)}%</span>
                       </div>
                     ))}
                  </div>
                </div>
              </div>

              {/* Middle Row: Emotions */}
              <div className="bg-slate-900/80 p-6 rounded-2xl border border-slate-700 shadow-lg backdrop-blur-sm">
                <div className="flex items-center gap-3 mb-6">
                  <div className="p-2 bg-slate-800 rounded-lg">
                    <Smile className="w-5 h-5 text-yellow-400" />
                  </div>
                  <div>
                    <h4 className="text-lg font-bold text-white">Emotional State</h4>
                    <p className="text-xs text-slate-400">Dominant: <span className="capitalize text-indigo-400 font-medium">{analysis.dominant_emotion}</span></p>
                  </div>
                </div>
                
                <div className="space-y-3">
                  {getSortedAttributes(analysis.emotion).map(([emotion, score], index) => (
                    <div key={emotion} className="relative">
                      <div className="flex justify-between mb-1.5">
                        <span className={`text-sm font-medium capitalize ${index === 0 ? 'text-white font-semibold' : 'text-slate-400'}`}>
                          {emotion}
                        </span>
                        <span className={`text-sm ${index === 0 ? 'text-white font-semibold' : 'text-slate-400'}`}>
                          {score.toFixed(2)}%
                        </span>
                      </div>
                      <div className="h-2.5 bg-slate-800 rounded-full overflow-hidden">
                        <div 
                          className={`h-full rounded-full transition-all duration-1000 ${EMOTION_COLORS[emotion]?.replace('text-', 'bg-') || 'bg-gray-500'}`} 
                          style={{ width: `${score}%` }}
                        ></div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Bottom Row: Race */}
              <div className="bg-slate-900/80 p-6 rounded-2xl border border-slate-700 shadow-lg backdrop-blur-sm">
                <div className="flex items-center gap-3 mb-5">
                  <div className="p-2 bg-slate-800 rounded-lg">
                    <Users className="w-5 h-5 text-emerald-400" />
                  </div>
                  <div>
                    <h4 className="text-lg font-bold text-white">Ethnicity Prediction</h4>
                    <p className="text-xs text-slate-400">Dominant: <span className="capitalize text-emerald-400 font-medium">{analysis.dominant_race}</span></p>
                  </div>
                </div>
                
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-x-6 gap-y-4">
                  {getSortedAttributes(analysis.race).map(([race, score], index) => (
                    <div key={race} className="flex items-center gap-3">
                       <div className="w-1.5 h-10 bg-slate-800 rounded-full overflow-hidden">
                          <div 
                            className={`w-full rounded-full transition-all duration-1000 ${index === 0 ? 'bg-emerald-500' : 'bg-slate-600'}`}
                            style={{ height: `${score}%`, marginTop: `${100 - score}%` }}
                          ></div>
                       </div>
                       <div className="flex-1">
                          <p className={`text-sm font-medium capitalize ${index === 0 ? 'text-white font-semibold' : 'text-slate-300'}`}>
                            {race.replace('latino hispanic', 'Latino/Hispanic').replace('middle eastern', 'Middle Eastern')}
                          </p>
                          <p className={`text-xs ${index === 0 ? 'text-emerald-400' : 'text-slate-500'}`}>
                            {score.toFixed(2)}% Match
                          </p>
                       </div>
                    </div>
                  ))}
                </div>
              </div>

            </div>
          )}
        </div>
      </div>
    </div>
  );
};