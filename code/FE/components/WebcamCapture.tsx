
import React, { useRef, useState, useEffect, useCallback } from 'react';
import { Camera, RefreshCw, CheckCircle, ScanFace, AlertTriangle, Loader2 } from 'lucide-react';

interface WebcamCaptureProps {
  onCapture: (base64Image: string) => void;
  label?: string;
  allowRetake?: boolean;
  initialImage?: string | null;
}

export const WebcamCapture: React.FC<WebcamCaptureProps> = ({ 
  onCapture, 
  label = "Capture Photo", 
  allowRetake = true,
  initialImage = null
}) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null);
  // Store the latest detection result (box, score) without triggering re-renders
  const latestDetectionRef = useRef<any>(null);
  
  const [isStreaming, setIsStreaming] = useState(false);
  const [image, setImage] = useState<string | null>(initialImage);
  const [error, setError] = useState<string | null>(null);
  const [faceDetected, setFaceDetected] = useState(false);
  const [detectionScore, setDetectionScore] = useState<number>(0);
  const [modelStatus, setModelStatus] = useState<string>("Initializing...");

  const startCamera = useCallback(async () => {
    try {
      setError(null);
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { width: { ideal: 640 }, height: { ideal: 480 } } 
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        // Wait for metadata to load to prevent 0x0 dimension issues
        videoRef.current.onloadedmetadata = () => {
          setIsStreaming(true);
          videoRef.current?.play();
        };
      }
    } catch (err) {
      console.error("Error accessing webcam:", err);
      setError("Cannot access camera. Please verify permissions.");
    }
  }, []);

  const stopCamera = useCallback(() => {
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach(track => track.stop());
      videoRef.current.srcObject = null;
      setIsStreaming(false);
    }
  }, []);

  useEffect(() => {
    if (!image) {
      startCamera();
    }
    return () => {
      stopCamera();
    };
  }, [startCamera, stopCamera, image]);

  // Face Detection Logic using face-api.js
  useEffect(() => {
    if (!isStreaming || image) return;

    let animationId: number;
    let isCancelled = false;

    const loadAndDetect = async () => {
      try {
        const faceapi = (window as any).faceapi;
        if (!faceapi) {
          setModelStatus("Waiting for library...");
          setTimeout(loadAndDetect, 500);
          return;
        }

        setModelStatus("Loading models...");

        // Load TinyFaceDetector models from the official demo page (Stable)
        const MODEL_URL = 'https://justadudewhohacks.github.io/face-api.js/models';
        
        if (!faceapi.nets.tinyFaceDetector.params) {
            await faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL);
        }
        
        setModelStatus("Ready");
        if (!isCancelled) detectFrame();
      } catch (e: any) {
        console.error("Failed to load face model", e);
        setError(`Model Load Failed: ${e.message || "Network error"}`);
        setModelStatus("Error");
      }
    };

    const detectFrame = async () => {
      if (!videoRef.current || !overlayCanvasRef.current || isCancelled) return;

      const video = videoRef.current;
      const canvas = overlayCanvasRef.current;
      const faceapi = (window as any).faceapi;

      if (video.readyState === 4 && video.videoWidth > 0) {
        // Get actual video dimensions
        const displaySize = { width: video.videoWidth, height: video.videoHeight };
        
        // Ensure canvas matches video dimensions
        if (canvas.width !== displaySize.width || canvas.height !== displaySize.height) {
           faceapi.matchDimensions(canvas, displaySize);
        }

        // Detect faces
        // Using 0.5 threshold for display, but we will enforce 0.8 for capture
        const detections = await faceapi.detectAllFaces(
            video, 
            new faceapi.TinyFaceDetectorOptions({ inputSize: 416, scoreThreshold: 0.5 }) 
        );

        const resizedDetections = faceapi.resizeResults(detections, displaySize);

        const ctx = canvas.getContext('2d');
        if (ctx) {
          ctx.clearRect(0, 0, canvas.width, canvas.height);

          if (resizedDetections.length > 0) {
            // Sort by score descending to get the best face
            const bestDetection = resizedDetections.sort((a: any, b: any) => b.score - a.score)[0];
            
            setFaceDetected(true);
            latestDetectionRef.current = bestDetection; // Save for capture
            
            const score = Math.round(bestDetection.score * 100);
            setDetectionScore(score);

            const box = bestDetection.box;
            
            // Visual Feedback based on score
            const isReadyToCapture = score > 80;

            // Draw bounding box
            ctx.beginPath();
            ctx.lineWidth = 4; 
            ctx.strokeStyle = isReadyToCapture ? '#00ff00' : '#fbbf24'; // Green if ready, Yellow if low score
            ctx.rect(box.x, box.y, box.width, box.height);
            ctx.stroke();
            
          } else {
            setFaceDetected(false);
            setDetectionScore(0);
            latestDetectionRef.current = null;
          }
        }
      }
      
      animationId = requestAnimationFrame(detectFrame);
    };

    loadAndDetect();

    return () => {
      isCancelled = true;
      cancelAnimationFrame(animationId);
    };
  }, [isStreaming, image]);

  const capture = () => {
    if (videoRef.current && canvasRef.current) {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      const detection = latestDetectionRef.current;

      // Enforce 80% Threshold
      if (!detection || detection.score < 0.8) {
        setError("Face match too low. Please adjust lighting or position (>80% required).");
        // Auto clear error after 3 seconds
        setTimeout(() => setError(null), 3000);
        return;
      }

      const box = detection.box;
      
      // Calculate Crop Coordinates with Padding
      // Add 20% padding to ensure the whole head is captured, not just the tight face box
      const padding = Math.max(box.width, box.height) * 0.2; 
      
      const x = Math.max(0, box.x - padding);
      const y = Math.max(0, box.y - padding);
      const w = Math.min(video.videoWidth - x, box.width + (padding * 2));
      const h = Math.min(video.videoHeight - y, box.height + (padding * 2));

      // Set canvas size to the cropped size
      canvas.width = w;
      canvas.height = h;
      
      const ctx = canvas.getContext('2d');
      if (ctx) {
        // Draw only the cropped region
        ctx.drawImage(video, x, y, w, h, 0, 0, w, h);
        
        const dataUrl = canvas.toDataURL('image/jpeg', 0.9);
        setImage(dataUrl);
        onCapture(dataUrl);
        stopCamera();
      }
    }
  };

  const retake = () => {
    setImage(null);
    onCapture("");
    setDetectionScore(0);
    setFaceDetected(false);
    setError(null);
  };

  return (
    <div className="w-full flex flex-col items-center gap-4">
      <div className="relative w-full max-w-md bg-slate-900 rounded-lg overflow-hidden border-2 border-slate-700 shadow-inner group min-h-[300px] flex items-center justify-center">
        {error && (
          <div className="absolute inset-0 flex items-center justify-center text-red-400 text-center p-4 z-30 bg-slate-900/95 animate-in fade-in">
            <div className="flex flex-col items-center">
              <AlertTriangle className="mb-2 w-8 h-8" />
              <span className="font-semibold">{error}</span>
            </div>
          </div>
        )}
        
        {!image && (
          <div className="relative w-full h-full flex justify-center">
            <video 
              ref={videoRef} 
              autoPlay 
              playsInline 
              muted 
              className="w-full h-full object-contain transform scale-x-[-1]" 
            />
            
            <canvas 
              ref={overlayCanvasRef}
              className="absolute inset-0 w-full h-full object-contain pointer-events-none z-10 transform scale-x-[-1]" 
            />
            
            {/* Status HUD */}
            <div className="absolute bottom-4 left-0 right-0 flex flex-col items-center gap-2 z-20 pointer-events-none">
               {modelStatus !== "Ready" ? (
                 <span className="bg-slate-900/90 text-yellow-400 text-xs px-3 py-1.5 rounded-full border border-yellow-500/30 flex items-center gap-2">
                   <Loader2 className="animate-spin w-3 h-3" />
                   {modelStatus}
                 </span>
               ) : faceDetected ? (
                 <div className="flex flex-col items-center gap-1">
                    <span className={`
                      text-xs px-4 py-1.5 rounded-full border flex items-center gap-2 shadow-lg font-mono transition-colors
                      ${detectionScore > 80 
                        ? 'bg-emerald-900/90 text-emerald-400 border-emerald-500/50 shadow-emerald-500/20' 
                        : 'bg-amber-900/90 text-amber-400 border-amber-500/50 shadow-amber-500/20'}
                    `}>
                      <ScanFace size={14} /> 
                      CONFIDENCE: {detectionScore}%
                    </span>
                    {detectionScore <= 80 && (
                      <span className="text-[10px] text-amber-300 bg-black/50 px-2 py-0.5 rounded">Need &gt;80% to capture</span>
                    )}
                 </div>
               ) : (
                 <span className="bg-slate-900/80 text-slate-400 text-xs px-3 py-1 rounded-full border border-slate-500/30 backdrop-blur-sm">
                   Scanning for face...
                 </span>
               )}
            </div>
          </div>
        )}

        {image && (
          <img src={image} alt="Captured" className="w-full h-auto object-contain transform scale-x-[-1]" />
        )}
        
        {/* Hidden Canvas for capturing */}
        <canvas ref={canvasRef} className="hidden" />
      </div>

      <div className="flex gap-3">
        {!image && (
          <button
            onClick={capture}
            disabled={!isStreaming || modelStatus !== "Ready" || !faceDetected || detectionScore <= 80}
            className={`flex items-center gap-2 px-6 py-2.5 rounded-full font-medium transition-all shadow-lg ${
              faceDetected && detectionScore > 80
                ? 'bg-indigo-600 hover:bg-indigo-500 text-white shadow-indigo-500/40 transform hover:scale-105' 
                : 'bg-slate-700 text-slate-400 border border-slate-600 cursor-not-allowed opacity-70'
            }`}
          >
            <Camera size={20} />
            {label}
          </button>
        )}

        {image && allowRetake && (
          <button
            onClick={retake}
            className="flex items-center gap-2 px-6 py-2.5 bg-slate-700 hover:bg-slate-600 text-white rounded-full font-medium transition-colors"
          >
            <RefreshCw size={18} />
            Retake
          </button>
        )}
      </div>
    </div>
  );
};
