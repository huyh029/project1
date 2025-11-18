
import React, { useState } from 'react';
import { ViewMode, UserProfile } from './types';
import { apiService } from './services/api';
import { WebcamCapture } from './components/WebcamCapture';
import { AnalyzeView } from './views/AnalyzeView';
import { CompareView } from './views/CompareView';
import { ShieldCheck, UserPlus, LogIn, ScanFace, GitCompare, Lock, BrainCircuit, Mail, Phone, User, Users, LogOut, CheckCircle, Fingerprint } from 'lucide-react';

interface NavButtonProps {
  children?: React.ReactNode;
  active: boolean;
  onClick: () => void;
  icon: React.ReactNode;
  locked?: boolean;
}

const NavButton = ({ children, active, onClick, icon, locked }: NavButtonProps) => (
  <button
    onClick={onClick}
    disabled={locked}
    className={`flex items-center gap-2 px-4 py-2 rounded-md text-sm font-medium transition-all ${
      active 
        ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-500/25' 
        : locked 
          ? 'text-slate-600 cursor-not-allowed' 
          : 'text-slate-300 hover:bg-slate-800 hover:text-white'
    }`}
  >
    {icon}
    {children}
    {locked && <Lock size={12} className="ml-1 text-slate-600" />}
  </button>
);

function App() {
  const [view, setView] = useState<ViewMode>('login');
  
  // Auth State
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  
  // Extended Registration State
  const [fullName, setFullName] = useState('');
  const [email, setEmail] = useState('');
  const [phone, setPhone] = useState('');
  const [gender, setGender] = useState('Male');

  const [capturedImage, setCapturedImage] = useState<string>('');
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState<{ type: 'success' | 'error', text: string } | null>(null);
  
  // Updated Auth Data to include full user profile
  const [authData, setAuthData] = useState<{ token: string, similarity: number, user?: UserProfile } | null>(null);

  const handleAuth = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!capturedImage) {
      setMessage({ type: 'error', text: "Please capture your face first." });
      return;
    }

    setLoading(true);
    setMessage(null);
    setAuthData(null);

    try {
      if (view === 'register') {
        await apiService.register(username, password, capturedImage, fullName, email, phone, gender);
        setMessage({ type: 'success', text: "Registration successful! Please login." });
        setView('login');
        setCapturedImage('');
      } else {
        const res = await apiService.login(username, password, capturedImage);
        if (res.token) {
          // Login Success Logic
          setAuthData({ 
            token: res.token, 
            similarity: res.similarity || 0,
            user: res.user 
          });
          setIsLoggedIn(true);
          setMessage({ type: 'success', text: `Welcome back, ${res.user?.fullName || username}!` });
          
          // Stay on Profile view (view === 'login' maps to Profile when isLoggedIn)
          // Do NOT redirect to analyze
        }
      }
    } catch (err: any) {
      setMessage({ type: 'error', text: err.message || "Operation failed" });
    } finally {
      setLoading(false);
    }
  };

  const handleLogout = () => {
    setIsLoggedIn(false);
    setAuthData(null);
    setUsername('');
    setPassword('');
    setCapturedImage('');
    setView('login');
    setMessage({ type: 'success', text: "Logged out successfully." });
  };

  const renderAuthView = () => (
    <div className="max-w-5xl mx-auto w-full grid grid-cols-1 md:grid-cols-2 gap-8 items-start animate-in fade-in duration-500">
      <div className="order-2 md:order-1 space-y-6">
        <div className="space-y-2">
          <h2 className="text-3xl font-bold text-white tracking-tight">
            {isLoggedIn ? 'Identity Profile' : (view === 'login' ? 'Secure Access' : 'New Identity')}
          </h2>
          <p className="text-slate-400">
            {isLoggedIn 
              ? 'Verified biometric profile information.'
              : (view === 'login' 
                  ? 'Authenticate using multi-factor facial verification to access system features.' 
                  : 'Register your biometric signature securely.')}
          </p>
        </div>

        {isLoggedIn ? (
          // LOGGED IN VIEW - Full Profile
          <div className="bg-slate-800 p-8 rounded-2xl border border-slate-700 shadow-xl space-y-6 relative overflow-hidden">
             {/* Background decoration */}
             <div className="absolute top-0 right-0 w-64 h-64 bg-indigo-500/5 rounded-full blur-3xl -mr-16 -mt-16 pointer-events-none"></div>

             <div className="flex flex-col md:flex-row items-center gap-6 pb-6 border-b border-slate-700/50">
                <div className="relative">
                  <div className="w-24 h-24 rounded-full overflow-hidden border-4 border-indigo-500/30 shadow-lg bg-slate-900">
                    {authData?.user?.img ? (
                      <img 
                        src={`data:image/png;base64,${authData.user.img}`} 
                        alt="Profile" 
                        className="w-full h-full object-cover" 
                      />
                    ) : (
                      <div className="w-full h-full flex items-center justify-center text-slate-600">
                        <User size={40} />
                      </div>
                    )}
                  </div>
                  <div className="absolute bottom-0 right-0 bg-emerald-500 text-slate-900 p-1.5 rounded-full border-2 border-slate-800">
                    <CheckCircle size={14} strokeWidth={3} />
                  </div>
                </div>
                
                <div className="text-center md:text-left">
                  <h3 className="text-2xl font-bold text-white">{authData?.user?.fullName || 'User'}</h3>
                  <p className="text-slate-400 font-mono text-sm">@{authData?.user?.userName}</p>
                  <div className="mt-2 inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-indigo-900/50 text-indigo-200 border border-indigo-700/50">
                    Authenticated Session
                  </div>
                </div>
             </div>

             <div className="grid grid-cols-1 gap-4">
                <div className="bg-slate-900/50 rounded-xl p-4 border border-slate-700/50 hover:border-slate-600 transition-colors">
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-y-4 gap-x-8">
                     <div className="flex items-center gap-3">
                       <div className="p-2 bg-slate-800 rounded-lg text-slate-400"><Mail size={16} /></div>
                       <div>
                         <p className="text-xs text-slate-500 uppercase font-semibold">Email</p>
                         <p className="text-sm text-slate-200 truncate">{authData?.user?.email || 'N/A'}</p>
                       </div>
                     </div>
                     <div className="flex items-center gap-3">
                       <div className="p-2 bg-slate-800 rounded-lg text-slate-400"><Phone size={16} /></div>
                       <div>
                         <p className="text-xs text-slate-500 uppercase font-semibold">Phone</p>
                         <p className="text-sm text-slate-200">{authData?.user?.phone || 'N/A'}</p>
                       </div>
                     </div>
                     <div className="flex items-center gap-3">
                       <div className="p-2 bg-slate-800 rounded-lg text-slate-400"><Users size={16} /></div>
                       <div>
                         <p className="text-xs text-slate-500 uppercase font-semibold">Gender</p>
                         <p className="text-sm text-slate-200">{authData?.user?.gender || 'N/A'}</p>
                       </div>
                     </div>
                     <div className="flex items-center gap-3">
                       <div className="p-2 bg-slate-800 rounded-lg text-slate-400"><BrainCircuit size={16} /></div>
                       <div>
                         <p className="text-xs text-slate-500 uppercase font-semibold">Role</p>
                         <p className="text-sm text-slate-200">Standard User</p>
                       </div>
                     </div>
                  </div>
                </div>
             </div>

             {authData && (
                <div className="grid grid-cols-2 gap-4">
                  <div className="p-4 bg-emerald-900/20 border border-emerald-500/30 rounded-xl">
                    <p className="text-xs text-emerald-400 uppercase font-bold mb-1 flex items-center gap-2">
                       <ScanFace size={14} /> Biometric Match
                    </p>
                    <p className="text-3xl font-mono text-emerald-300 tracking-tight">{authData.similarity}%</p>
                    <p className="text-xs text-emerald-500/70 mt-1">Confidence Score</p>
                  </div>
                  <div className="p-4 bg-indigo-900/20 border border-indigo-500/30 rounded-xl">
                    <p className="text-xs text-indigo-400 uppercase font-bold mb-1 flex items-center gap-2">
                      <Fingerprint size={14} /> Security Token
                    </p>
                    <p className="text-xs font-mono text-indigo-300 break-all leading-tight mt-2 opacity-70">
                      {authData.token.substring(0, 16)}...
                    </p>
                    <p className="text-xs text-indigo-500/70 mt-2">Encrypted</p>
                  </div>
                </div>
             )}

             <button 
               onClick={handleLogout}
               className="w-full flex items-center justify-center gap-2 bg-slate-700 hover:bg-red-600 hover:text-white text-slate-200 font-bold py-3 rounded-lg transition-all mt-2"
             >
               <LogOut size={18} /> Sign Out
             </button>
          </div>
        ) : (
          // LOGIN / REGISTER FORM
          <form onSubmit={handleAuth} className="space-y-4 bg-slate-800 p-6 rounded-2xl border border-slate-700 shadow-xl max-h-[650px] overflow-y-auto scrollbar-thin scrollbar-thumb-slate-600 scrollbar-track-slate-800 pr-2">
            
            {/* Login/Register Common Fields */}
            <div>
              <label className="block text-sm font-medium text-slate-400 mb-1">Username</label>
              <div className="relative">
                <User className="absolute left-3 top-3.5 h-4 w-4 text-slate-500" />
                <input 
                  type="text" 
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  className="w-full bg-slate-900 border border-slate-700 rounded-lg pl-10 pr-4 py-3 text-white focus:ring-2 focus:ring-indigo-500 focus:outline-none transition-all"
                  placeholder="Enter username"
                  required 
                />
              </div>
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-400 mb-1">Password</label>
              <div className="relative">
                <Lock className="absolute left-3 top-3.5 h-4 w-4 text-slate-500" />
                <input 
                  type="password" 
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="w-full bg-slate-900 border border-slate-700 rounded-lg pl-10 pr-4 py-3 text-white focus:ring-2 focus:ring-indigo-500 focus:outline-none transition-all"
                  placeholder="••••••••"
                  required 
                />
              </div>
            </div>

            {/* Extended Fields for Registration Only */}
            {view === 'register' && (
              <div className="space-y-4 pt-2 border-t border-slate-700 animate-in slide-in-from-top-2 fade-in duration-300">
                <div>
                  <label className="block text-sm font-medium text-slate-400 mb-1">Full Name</label>
                  <div className="relative">
                    <UserPlus className="absolute left-3 top-3.5 h-4 w-4 text-slate-500" />
                    <input 
                      type="text" 
                      value={fullName}
                      onChange={(e) => setFullName(e.target.value)}
                      className="w-full bg-slate-900 border border-slate-700 rounded-lg pl-10 pr-4 py-3 text-white focus:ring-2 focus:ring-indigo-500 focus:outline-none transition-all"
                      placeholder="John Doe"
                    />
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-slate-400 mb-1">Email</label>
                    <div className="relative">
                      <Mail className="absolute left-3 top-3.5 h-4 w-4 text-slate-500" />
                      <input 
                        type="email" 
                        value={email}
                        onChange={(e) => setEmail(e.target.value)}
                        className="w-full bg-slate-900 border border-slate-700 rounded-lg pl-10 pr-2 py-3 text-white focus:ring-2 focus:ring-indigo-500 focus:outline-none transition-all text-sm"
                        placeholder="john@example.com"
                      />
                    </div>
                  </div>
                  <div>
                     <label className="block text-sm font-medium text-slate-400 mb-1">Phone</label>
                     <div className="relative">
                      <Phone className="absolute left-3 top-3.5 h-4 w-4 text-slate-500" />
                      <input 
                        type="tel" 
                        value={phone}
                        onChange={(e) => setPhone(e.target.value)}
                        className="w-full bg-slate-900 border border-slate-700 rounded-lg pl-10 pr-2 py-3 text-white focus:ring-2 focus:ring-indigo-500 focus:outline-none transition-all text-sm"
                        placeholder="+1 234 567"
                      />
                     </div>
                  </div>
                </div>

                <div>
                  <label className="block text-sm font-medium text-slate-400 mb-1">Gender</label>
                  <div className="relative">
                    <Users className="absolute left-3 top-3.5 h-4 w-4 text-slate-500" />
                    <select 
                      value={gender} 
                      onChange={(e) => setGender(e.target.value)}
                      className="w-full bg-slate-900 border border-slate-700 rounded-lg pl-10 pr-4 py-3 text-white focus:ring-2 focus:ring-indigo-500 focus:outline-none transition-all appearance-none"
                    >
                      <option value="Male">Male</option>
                      <option value="Female">Female</option>
                      <option value="Other">Other</option>
                    </select>
                  </div>
                </div>
              </div>
            )}

            {message && (
              <div className={`p-3 rounded-lg text-sm ${message.type === 'success' ? 'bg-emerald-900/50 text-emerald-200 border border-emerald-800' : 'bg-red-900/50 text-red-200 border border-red-800'}`}>
                {message.text}
              </div>
            )}

            <button 
              type="submit" 
              disabled={loading}
              className="w-full bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-500 hover:to-purple-500 text-white font-bold py-3 rounded-lg shadow-lg shadow-indigo-500/20 transition-all transform active:scale-95"
            >
              {loading ? 'Processing...' : (view === 'login' ? 'Authenticate' : 'Register Identity')}
            </button>
          </form>
        )}
        
        {!isLoggedIn && (
          <div className="text-center">
            <button 
               onClick={() => {
                 setView(view === 'login' ? 'register' : 'login');
                 setMessage(null);
                 setCapturedImage('');
                 setAuthData(null);
               }}
               className="text-slate-500 hover:text-indigo-400 text-sm font-medium transition-colors"
            >
              {view === 'login' ? "Don't have an account? Register" : "Already registered? Login"}
            </button>
          </div>
        )}
      </div>

      {/* Right Side (Camera) - Hide if logged in (showing profile instead) OR allow camera for re-auth if needed */}
      <div className="order-1 md:order-2">
         {!isLoggedIn ? (
           <div className="bg-slate-800 p-4 rounded-2xl border border-slate-700 shadow-2xl sticky top-24">
              <WebcamCapture 
                onCapture={setCapturedImage} 
                label={view === 'login' ? "Verify Face" : "Capture ID"} 
                allowRetake={true}
              />
           </div>
         ) : (
            // Decorative element when logged in
            <div className="hidden md:flex items-center justify-center h-full opacity-50">
               <ShieldCheck size={200} className="text-indigo-900/50" />
            </div>
         )}
      </div>
    </div>
  );

  // Protection Logic: If not logged in, force show auth view content (though nav remains visible but locked)
  const currentViewContent = !isLoggedIn && (view === 'analyze' || view === 'compare') 
      ? renderAuthView() // Fallback if they somehow got here
      : (
        <>
          {(view === 'login' || view === 'register') && renderAuthView()}
          {view === 'analyze' && <AnalyzeView />}
          {view === 'compare' && <CompareView />}
        </>
      );

  return (
    <div className="min-h-screen bg-slate-950 text-slate-200 font-sans selection:bg-indigo-500/30">
      {/* Navbar */}
      <nav className="bg-slate-900/80 backdrop-blur-md border-b border-slate-800 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center gap-2 cursor-pointer" onClick={() => setView(isLoggedIn ? 'login' : 'login')}>
              <div className="bg-indigo-600 p-1.5 rounded-lg">
                <BrainCircuit className="text-white w-6 h-6" />
              </div>
              <span className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-white to-slate-400">
                FaceAI<span className="text-indigo-500">.Secure</span>
              </span>
            </div>
            
            <div className="hidden md:block">
              <div className="ml-10 flex items-baseline space-x-4">
                <NavButton 
                  active={view === 'login' || view === 'register'} 
                  onClick={() => setView('login')} 
                  icon={<ShieldCheck size={16}/>}
                >
                  {isLoggedIn ? 'Profile' : 'Auth'}
                </NavButton>
                
                <NavButton 
                  active={view === 'analyze'} 
                  onClick={() => setView('analyze')} 
                  icon={<ScanFace size={16}/>}
                  locked={!isLoggedIn}
                >
                  Analyze
                </NavButton>
                
                <NavButton 
                  active={view === 'compare'} 
                  onClick={() => setView('compare')} 
                  icon={<GitCompare size={16}/>}
                  locked={!isLoggedIn}
                >
                  Compare
                </NavButton>
              </div>
            </div>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 py-12 flex flex-col items-center justify-center min-h-[calc(100vh-4rem)]">
        {currentViewContent}
      </main>
      
      {/* Mobile Nav Footer */}
      <div className="md:hidden fixed bottom-0 w-full bg-slate-900 border-t border-slate-800 p-4 flex justify-around z-50">
        <button onClick={() => setView('login')} className={`flex flex-col items-center gap-1 ${view === 'login' || view === 'register' ? 'text-indigo-400' : 'text-slate-500'}`}>
           <Lock size={20} />
           <span className="text-xs">{isLoggedIn ? 'Profile' : 'Auth'}</span>
        </button>
        <button 
          onClick={() => isLoggedIn && setView('analyze')} 
          className={`flex flex-col items-center gap-1 ${view === 'analyze' ? 'text-indigo-400' : !isLoggedIn ? 'text-slate-700' : 'text-slate-500'}`}
          disabled={!isLoggedIn}
        >
           <ScanFace size={20} />
           <span className="text-xs">Analyze {(!isLoggedIn) && '🔒'}</span>
        </button>
        <button 
          onClick={() => isLoggedIn && setView('compare')} 
          className={`flex flex-col items-center gap-1 ${view === 'compare' ? 'text-indigo-400' : !isLoggedIn ? 'text-slate-700' : 'text-slate-500'}`}
          disabled={!isLoggedIn}
        >
           <GitCompare size={20} />
           <span className="text-xs">Compare {(!isLoggedIn) && '🔒'}</span>
        </button>
      </div>
    </div>
  );
}

export default App;
