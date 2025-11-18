import { API_BASE_URL } from '../constants';
import { AuthResponse, AnalyzeResponse, CompareResponse } from '../types';

const handleResponse = async (response: Response) => {
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || `Request failed with status ${response.status}`);
  }
  return data;
};

export const apiService = {
  register: async (
    username: string, 
    password: string, 
    img: string, 
    fullName: string, 
    email: string, 
    phone: string, 
    gender: string
  ): Promise<AuthResponse> => {
    const response = await fetch(`${API_BASE_URL}/api/register`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password, img, fullName, email, phone, gender }),
    });
    return handleResponse(response);
  },

  login: async (username: string, password: string, img: string): Promise<AuthResponse> => {
    const response = await fetch(`${API_BASE_URL}/api/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password, img }),
    });
    return handleResponse(response);
  },

  analyze: async (img: string): Promise<AnalyzeResponse> => {
    const response = await fetch(`${API_BASE_URL}/api/analyze`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ img }),
    });
    return handleResponse(response);
  },

  compare: async (img1: string, img2: string): Promise<CompareResponse> => {
    const response = await fetch(`${API_BASE_URL}/api/compare`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ img1, img2 }),
    });
    return handleResponse(response);
  },
};