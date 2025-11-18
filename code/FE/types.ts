
export interface UserProfile {
  userName: string;
  fullName?: string;
  email?: string;
  phone?: string;
  gender?: string;
  img?: string; // base64
}

export interface AuthResponse {
  message?: string;
  token?: string;
  similarity?: number;
  error?: string;
  user?: UserProfile;
}

export interface FaceRegion {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface AnalysisResult {
  age: number;
  region: FaceRegion;
  gender: Record<string, number>; // e.g., { Man: 99.9, Woman: 0.1 }
  dominant_gender: string;
  emotion: Record<string, number>; // e.g., { happy: 90, sad: 10 }
  dominant_emotion: string;
  race: Record<string, number>; // e.g., { asian: 10, white: 90 }
  dominant_race: string;
  face_confidence?: number;
}

export interface AnalyzeResponse {
  result?: AnalysisResult[]; 
  error?: string;
}

export interface CompareResponse {
  verified?: boolean;
  distance?: number;
  similarity?: number;
  error?: string;
}

export type ViewMode = 'register' | 'login' | 'analyze' | 'compare';
