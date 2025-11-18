
import React, { useState, ChangeEvent } from 'react';
import { WebcamCapture } from './WebcamCapture';
import { Upload, Camera, Image as ImageIcon, X } from 'lucide-react';

interface ImageInputProps {
  label: string;
  onImageChange: (base64: string) => void;
  subLabel?: string;
}

export const ImageInput: React.FC<ImageInputProps> = ({ label, onImageChange, subLabel }) => {
  // Change default mode to 'upload' as requested
  const [mode, setMode] = useState<'camera' | 'upload'>('upload');
  const [preview, setPreview] = useState<string | null>(null);

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      const reader = new FileReader();
      reader.onloadend = () => {
        const result = reader.result as string;
        setPreview(result);
        onImageChange(result);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleCameraCapture = (base64: string) => {
    setPreview(base64);
    onImageChange(base64);
  };

  const clearImage = () => {
    setPreview(null);
    onImageChange('');
  };

  return (
    <div className="bg-slate-900/50 backdrop-blur-sm p-1 rounded-2xl border border-slate-700 shadow-xl h-full flex flex-col">
      {/* Header / Tabs */}
      <div className="bg-slate-900 rounded-xl p-1 mb-2 flex items-center justify-between border border-slate-800/50">
        <div className="px-3 py-2">
            <h3 className="text-sm font-bold text-slate-200 uppercase tracking-wider flex items-center gap-2">
              {label}
            </h3>
            {subLabel && <p className="text-xs text-slate-500">{subLabel}</p>}
        </div>
        
        <div className="flex bg-slate-950 rounded-lg p-1 border border-slate-800">
          <button
            onClick={() => setMode('upload')}
            className={`px-3 py-1.5 rounded-md text-xs font-medium transition-all flex items-center gap-1.5 ${
              mode === 'upload' ? 'bg-slate-800 text-indigo-400 shadow-sm' : 'text-slate-500 hover:text-slate-300'
            }`}
          >
            <Upload size={14} /> Upload
          </button>
          <button
            onClick={() => setMode('camera')}
            className={`px-3 py-1.5 rounded-md text-xs font-medium transition-all flex items-center gap-1.5 ${
              mode === 'camera' ? 'bg-slate-800 text-indigo-400 shadow-sm' : 'text-slate-500 hover:text-slate-300'
            }`}
          >
            <Camera size={14} /> Cam
          </button>
        </div>
      </div>

      {/* Content Area */}
      <div className="flex-1 bg-slate-950/30 rounded-xl border border-slate-800/50 overflow-hidden relative min-h-[320px] flex flex-col">
        
        {preview ? (
          <div className="relative w-full h-full flex-1 bg-black flex items-center justify-center group">
            <img src={preview} alt="Preview" className="w-full h-full object-contain" />
            
            <div className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center gap-4 backdrop-blur-sm">
               <button 
                 onClick={clearImage}
                 className="bg-red-500/20 hover:bg-red-500/40 text-red-200 p-3 rounded-full backdrop-blur-md transition-transform hover:scale-110 border border-red-500/30"
                 title="Remove Image"
               >
                 <X size={24} />
               </button>
            </div>
          </div>
        ) : (
          <div className="w-full h-full flex-1 p-4 flex flex-col">
            {mode === 'upload' ? (
              <label className="flex-1 flex flex-col items-center justify-center border-2 border-slate-700 border-dashed rounded-xl cursor-pointer hover:bg-slate-800/30 hover:border-indigo-500/50 transition-all group">
                <div className="bg-slate-800 p-4 rounded-full mb-4 group-hover:scale-110 transition-transform shadow-lg">
                  <Upload className="w-8 h-8 text-indigo-400" />
                </div>
                <p className="text-sm font-medium text-slate-300">Click or Drag image here</p>
                <p className="text-xs text-slate-500 mt-1">Supports JPG, PNG (Max 5MB)</p>
                <input type="file" className="hidden" accept="image/*" onChange={handleFileChange} />
              </label>
            ) : (
              <div className="flex-1 flex flex-col h-full">
                 <WebcamCapture onCapture={handleCameraCapture} label="Capture Face" />
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};
