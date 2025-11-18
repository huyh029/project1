
import React, { useState } from 'react';
import { ImageInput } from '../components/ImageInput';
import { apiService } from '../services/api';
import { CompareResponse } from '../types';
import { Loader2, GitCompare, CheckCircle2, XCircle, Fingerprint, ArrowRightLeft } from 'lucide-react';

export const CompareView: React.FC = () => {
  const [img1, setImg1] = useState<string>('');
  const [img2, setImg2] = useState<string>('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<CompareResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleCompare = async () => {
    if (!img1 || !img2) {
      setError("Please provide both images.");
      return;
    }
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const data = await apiService.compare(img1, img2);
      setResult(data);
    } catch (err: any) {
      setError(err.message || "Comparison failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="w-full max-w-6xl mx-auto p-4 space-y-8">
      
      {/* Header */}
      <div className="text-center space-y-2">
        <h2 className="text-3xl font-bold text-white flex items-center justify-center gap-3">
          <div className="bg-indigo-600 p-2 rounded-lg">
             <GitCompare className="text-white w-6 h-6" />
          </div>
          Biometric Verification
        </h2>
        <p className="text-slate-400 max-w-2xl mx-auto">
          Compare two facial signatures to determine if they belong to the same identity using Euclidean distance measurement.
        </p>
      </div>

      <div className="relative">
        {/* Connector Line (Desktop) */}
        <div className="hidden lg:block absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-16 h-16 bg-slate-900 rounded-full border-4 border-slate-950 z-10 !flex align-center items-center justify-center">
           <ArrowRightLeft className="text-slate-600" size={24} />
        </div>

        {/* Main Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 lg:gap-12">
          
          {/* Left Side: Reference */}
          <div className="space-y-2">
            <ImageInput 
              label="Reference Identity" 
              subLabel="Original Source / ID Card"
              onImageChange={(val) => { setImg1(val); setResult(null); }} 
            />
          </div>

          {/* Right Side: Target */}
          <div className="space-y-2">
            <ImageInput 
              label="Verification Target" 
              subLabel="Live Capture / Surveillance"
              onImageChange={(val) => { setImg2(val); setResult(null); }} 
            />
          </div>
        </div>
      </div>

      {/* Actions Area */}
      <div className="flex flex-col items-center justify-center py-6">
        <button
          onClick={handleCompare}
          disabled={loading || !img1 || !img2}
          className={`
            relative group overflow-hidden px-12 py-4 rounded-full font-bold text-lg tracking-wide transition-all duration-300
            ${loading || !img1 || !img2 
              ? 'bg-slate-800 text-slate-500 cursor-not-allowed' 
              : 'bg-indigo-600 hover:bg-indigo-500 text-white shadow-xl shadow-indigo-500/30 hover:scale-105 hover:shadow-indigo-500/50'}
          `}
        >
          <div className="relative z-10 flex items-center gap-3">
            {loading ? <Loader2 className="animate-spin w-5 h-5" /> : <Fingerprint className="w-5 h-5" />}
            {loading ? 'Processing...' : 'RUN COMPARISON'}
          </div>
          {!loading && img1 && img2 && (
            <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/10 to-transparent -translate-x-full group-hover:animate-shimmer" />
          )}
        </button>

        {error && (
           <div className="mt-4 text-red-400 bg-red-900/20 px-4 py-2 rounded-lg border border-red-500/20 animate-in fade-in">
             {error}
           </div>
        )}
      </div>

      {/* Results Section */}
      {result && (
        <div className="animate-in slide-in-from-bottom-4 fade-in duration-500">
          <div className={`
            max-w-2xl mx-auto rounded-2xl border-2 overflow-hidden shadow-2xl
            ${result.verified 
              ? 'bg-emerald-950/40 border-emerald-500/50 shadow-emerald-500/10' 
              : 'bg-red-950/40 border-red-500/50 shadow-red-500/10'}
          `}>
            <div className={`p-6 text-center border-b ${result.verified ? 'border-emerald-500/20 bg-emerald-500/10' : 'border-red-500/20 bg-red-500/10'}`}>
               <div className="flex items-center justify-center gap-3 mb-2">
                 {result.verified 
                   ? <CheckCircle2 className="w-10 h-10 text-emerald-400" />
                   : <XCircle className="w-10 h-10 text-red-400" />
                 }
                 <span className={`text-3xl font-black tracking-tighter ${result.verified ? 'text-white' : 'text-white'}`}>
                   {result.verified ? 'IDENTITY MATCHED' : 'NO MATCH FOUND'}
                 </span>
               </div>
               <p className={`text-sm font-medium ${result.verified ? 'text-emerald-200/70' : 'text-red-200/70'}`}>
                 {result.verified 
                   ? 'The biometric features indicate these images belong to the same person.' 
                   : 'Significant differences detected in facial structure.'}
               </p>
            </div>

            <div className="grid grid-cols-2 divide-x divide-slate-700/50">
               <div className="p-6 text-center hover:bg-white/5 transition-colors">
                 <span className="block text-slate-400 text-xs font-bold uppercase tracking-widest mb-1">Similarity Confidence</span>
                 <span className={`text-4xl font-bold ${result.verified ? 'text-emerald-400' : 'text-red-400'}`}>
                   {result.similarity}%
                 </span>
               </div>
               <div className="p-6 text-center hover:bg-white/5 transition-colors">
                 <span className="block text-slate-400 text-xs font-bold uppercase tracking-widest mb-1">Euclidean Distance</span>
                 <span className="text-4xl font-bold text-slate-200">
                   {result.distance?.toFixed(4)}
                 </span>
                 <span className="block text-xs text-slate-500 mt-1">Threshold: ~0.40</span>
               </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
