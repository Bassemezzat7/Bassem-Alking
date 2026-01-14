
import React, { useState, useEffect, useRef } from 'react';
import { createRoot } from 'react-dom/client';
import { GoogleGenAI, Modality, Type, LiveServerMessage } from '@google/genai';
import { 
  MessageSquare, 
  Image as ImageIcon, 
  Mic, 
  Send, 
  Sparkles, 
  Zap, 
  History,
  Settings,
  Volume2,
  StopCircle
} from 'lucide-react';

// --- Utilities ---
function decode(base64: string) {
  const binaryString = atob(base64);
  const bytes = new Uint8Array(binaryString.length);
  for (let i = 0; i < binaryString.length; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  return bytes;
}

function encode(bytes: Uint8Array) {
  let binary = '';
  for (let i = 0; i < bytes.byteLength; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
}

async function decodeAudioData(
  data: Uint8Array,
  ctx: AudioContext,
  sampleRate: number,
  numChannels: number,
): Promise<AudioBuffer> {
  const dataInt16 = new Int16Array(data.buffer);
  const frameCount = dataInt16.length / numChannels;
  const buffer = ctx.createBuffer(numChannels, frameCount, sampleRate);

  for (let channel = 0; channel < numChannels; channel++) {
    const channelData = buffer.getChannelData(channel);
    for (let i = 0; i < frameCount; i++) {
      channelData[i] = dataInt16[i * numChannels + channel] / 32768.0;
    }
  }
  return buffer;
}

// --- Components ---

const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'text' | 'image' | 'live'>('text');
  const [messages, setMessages] = useState<{ role: 'user' | 'model'; text: string }[]>([]);
  const [inputText, setInputText] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [generatedImage, setGeneratedImage] = useState<string | null>(null);
  const [imagePrompt, setImagePrompt] = useState('');
  
  // Live Session States
  const [isLiveActive, setIsLiveActive] = useState(false);
  const sessionRef = useRef<any>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const sourcesRef = useRef<Set<AudioBufferSourceNode>>(new Set());
  const nextStartTimeRef = useRef<number>(0);

  const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });

  // Text Generation Handler
  const handleSendMessage = async () => {
    if (!inputText.trim() || isGenerating) return;

    const userMsg = { role: 'user' as const, text: inputText };
    setMessages(prev => [...prev, userMsg]);
    setInputText('');
    setIsGenerating(true);

    try {
      const chat = ai.chats.create({
        model: 'gemini-3-pro-preview',
        config: { systemInstruction: 'أنت مساعد ذكي ومبدع تتحدث العربية بطلاقة وبأسلوب ودود.' }
      });

      const response = await chat.sendMessageStream({ message: inputText });
      let fullText = '';
      
      setMessages(prev => [...prev, { role: 'model', text: '' }]);

      for await (const chunk of response) {
        fullText += chunk.text || '';
        setMessages(prev => {
          const newMessages = [...prev];
          newMessages[newMessages.length - 1].text = fullText;
          return newMessages;
        });
      }
    } catch (error) {
      console.error(error);
      setMessages(prev => [...prev, { role: 'model', text: 'عذراً، حدث خطأ ما أثناء معالجة طلبك.' }]);
    } finally {
      setIsGenerating(false);
    }
  };

  // Image Generation Handler
  const handleGenerateImage = async () => {
    if (!imagePrompt.trim() || isGenerating) return;
    setIsGenerating(true);
    setGeneratedImage(null);

    try {
      const response = await ai.models.generateContent({
        model: 'gemini-2.5-flash-image',
        contents: { parts: [{ text: imagePrompt }] },
        config: { imageConfig: { aspectRatio: "1:1" } }
      });

      const part = response.candidates?.[0]?.content?.parts.find(p => p.inlineData);
      if (part?.inlineData) {
        setGeneratedImage(`data:image/png;base64,${part.inlineData.data}`);
      }
    } catch (error) {
      console.error(error);
    } finally {
      setIsGenerating(false);
    }
  };

  // Live Audio Handlers
  const startLiveSession = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const inputCtx = new AudioContext({ sampleRate: 16000 });
      const outputCtx = new AudioContext({ sampleRate: 24000 });
      audioContextRef.current = outputCtx;

      const sessionPromise = ai.live.connect({
        model: 'gemini-2.5-flash-native-audio-preview-12-2025',
        config: {
          responseModalities: [Modality.AUDIO],
          speechConfig: { voiceConfig: { prebuiltVoiceConfig: { voiceName: 'Puck' } } },
          systemInstruction: 'أنت رفيق صوتي ذكي ومبتهج. رد باختصار وود باللغة العربية.'
        },
        callbacks: {
          onopen: () => {
            setIsLiveActive(true);
            const source = inputCtx.createMediaStreamSource(stream);
            const scriptProcessor = inputCtx.createScriptProcessor(4096, 1, 1);
            scriptProcessor.onaudioprocess = (e) => {
              const inputData = e.inputBuffer.getChannelData(0);
              const int16 = new Int16Array(inputData.length);
              for (let i = 0; i < inputData.length; i++) int16[i] = inputData[i] * 32768;
              const pcmBlob = { data: encode(new Uint8Array(int16.buffer)), mimeType: 'audio/pcm;rate=16000' };
              sessionPromise.then(s => s.sendRealtimeInput({ media: pcmBlob }));
            };
            source.connect(scriptProcessor);
            scriptProcessor.connect(inputCtx.destination);
          },
          onmessage: async (msg: LiveServerMessage) => {
            const audioData = msg.serverContent?.modelTurn?.parts[0]?.inlineData?.data;
            if (audioData && audioContextRef.current) {
              const ctx = audioContextRef.current;
              nextStartTimeRef.current = Math.max(nextStartTimeRef.current, ctx.currentTime);
              const buffer = await decodeAudioData(decode(audioData), ctx, 24000, 1);
              const source = ctx.createBufferSource();
              source.buffer = buffer;
              source.connect(ctx.destination);
              source.start(nextStartTimeRef.current);
              nextStartTimeRef.current += buffer.duration;
              sourcesRef.current.add(source);
              source.onended = () => sourcesRef.current.delete(source);
            }
            if (msg.serverContent?.interrupted) {
              sourcesRef.current.forEach(s => s.stop());
              sourcesRef.current.clear();
              nextStartTimeRef.current = 0;
            }
          },
          onclose: () => setIsLiveActive(false),
          onerror: (e) => console.error(e)
        }
      });
      sessionRef.current = await sessionPromise;
    } catch (error) {
      console.error(error);
    }
  };

  const stopLiveSession = () => {
    if (sessionRef.current) {
      sessionRef.current.close();
      setIsLiveActive(false);
    }
  };

  return (
    <div className="flex h-screen overflow-hidden">
      {/* Sidebar */}
      <aside className="w-20 md:w-64 glass flex flex-col items-center py-8 gap-8 border-l border-white/10">
        <div className="flex items-center gap-2 mb-4 px-4">
          <div className="bg-indigo-600 p-2 rounded-xl shadow-lg shadow-indigo-500/50">
            <Sparkles className="w-6 h-6 text-white" />
          </div>
          <span className="hidden md:block font-bold text-xl tracking-tight">جيمناي برو</span>
        </div>

        <nav className="flex flex-col gap-4 w-full px-4">
          <button 
            onClick={() => setActiveTab('text')}
            className={`flex items-center gap-4 p-3 rounded-xl transition-all ${activeTab === 'text' ? 'bg-indigo-600/20 text-indigo-400 ring-1 ring-indigo-500/50' : 'hover:bg-white/5 text-gray-400'}`}
          >
            <MessageSquare className="w-6 h-6" />
            <span className="hidden md:block font-medium">محادثة نصية</span>
          </button>
          <button 
            onClick={() => setActiveTab('image')}
            className={`flex items-center gap-4 p-3 rounded-xl transition-all ${activeTab === 'image' ? 'bg-indigo-600/20 text-indigo-400 ring-1 ring-indigo-500/50' : 'hover:bg-white/5 text-gray-400'}`}
          >
            <ImageIcon className="w-6 h-6" />
            <span className="hidden md:block font-medium">توليد صور</span>
          </button>
          <button 
            onClick={() => setActiveTab('live')}
            className={`flex items-center gap-4 p-3 rounded-xl transition-all ${activeTab === 'live' ? 'bg-indigo-600/20 text-indigo-400 ring-1 ring-indigo-500/50' : 'hover:bg-white/5 text-gray-400'}`}
          >
            <Mic className="w-6 h-6" />
            <span className="hidden md:block font-medium">تفاعل صوتي</span>
          </button>
        </nav>

        <div className="mt-auto w-full px-4 space-y-4">
          <div className="hidden md:block p-4 rounded-2xl bg-indigo-950/40 border border-indigo-500/20">
            <p className="text-xs text-indigo-300/80 mb-2">استهلاك النموذج</p>
            <div className="h-1.5 bg-white/10 rounded-full overflow-hidden">
              <div className="h-full bg-indigo-500 w-2/3 shadow-[0_0_8px_rgba(99,102,241,0.8)]"></div>
            </div>
          </div>
          <button className="flex items-center gap-4 p-3 rounded-xl hover:bg-white/5 text-gray-400 w-full transition-all">
            <Settings className="w-6 h-6" />
            <span className="hidden md:block font-medium">الإعدادات</span>
          </button>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 relative flex flex-col bg-slate-950/20">
        {/* Header */}
        <header className="h-16 flex items-center justify-between px-8 border-b border-white/5 glass z-10">
          <h2 className="text-lg font-semibold flex items-center gap-2">
            {activeTab === 'text' && <><MessageSquare className="w-5 h-5 text-indigo-400" /> المحادثة الذكية</>}
            {activeTab === 'image' && <><ImageIcon className="w-5 h-5 text-purple-400" /> استوديو الصور</>}
            {activeTab === 'live' && <><Mic className="w-5 h-5 text-rose-400" /> المختبر الصوتي</>}
          </h2>
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-emerald-500/10 border border-emerald-500/20">
              <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></div>
              <span className="text-xs text-emerald-400 font-medium uppercase tracking-wider">متصل</span>
            </div>
          </div>
        </header>

        {/* Content Area */}
        <div className="flex-1 overflow-y-auto p-4 md:p-8 scrollbar-hide">
          {activeTab === 'text' && (
            <div className="max-w-4xl mx-auto space-y-6">
              {messages.length === 0 && (
                <div className="flex flex-col items-center justify-center h-[60vh] text-center space-y-4">
                  <div className="w-20 h-20 bg-indigo-600/10 rounded-3xl flex items-center justify-center border border-indigo-500/20">
                    <Zap className="w-10 h-10 text-indigo-500" />
                  </div>
                  <h1 className="text-3xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-white to-gray-500">كيف يمكنني مساعدتك اليوم؟</h1>
                  <p className="text-gray-400 max-w-md">اسألني أي شيء، من حل المشكلات البرمجية المعقدة إلى كتابة القصص الإبداعية.</p>
                </div>
              )}
              {messages.map((msg, i) => (
                <div key={i} className={`flex ${msg.role === 'user' ? 'justify-start' : 'justify-end'}`}>
                  <div className={`max-w-[85%] p-4 rounded-2xl ${msg.role === 'user' ? 'bg-indigo-600 text-white rounded-tr-none' : 'glass rounded-tl-none border-indigo-500/20'}`}>
                    <p className="whitespace-pre-wrap leading-relaxed">{msg.text || 'جاري التفكير...'}</p>
                  </div>
                </div>
              ))}
              {isGenerating && messages[messages.length-1]?.role === 'user' && (
                <div className="flex justify-end">
                  <div className="glass p-4 rounded-2xl rounded-tl-none border-indigo-500/20 animate-pulse">
                    <div className="flex gap-1">
                      <div className="w-2 h-2 bg-indigo-400 rounded-full animate-bounce"></div>
                      <div className="w-2 h-2 bg-indigo-400 rounded-full animate-bounce [animation-delay:0.2s]"></div>
                      <div className="w-2 h-2 bg-indigo-400 rounded-full animate-bounce [animation-delay:0.4s]"></div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}

          {activeTab === 'image' && (
            <div className="max-w-4xl mx-auto space-y-8">
              <div className="glass p-8 rounded-[2.5rem] border-purple-500/20 shadow-2xl">
                <div className="space-y-6 text-center">
                  <h2 className="text-2xl font-bold">حول خيالك إلى حقيقة</h2>
                  <p className="text-gray-400">صف الصورة التي تتخيلها وسأقوم بإنشائها لك باستخدام الذكاء الاصطناعي.</p>
                  
                  <div className="relative group">
                    <textarea 
                      value={imagePrompt}
                      onChange={(e) => setImagePrompt(e.target.value)}
                      placeholder="رائد فضاء يركب خيلاً في الفضاء بأسلوب الفن الرقمي..."
                      className="w-full bg-slate-900/50 border border-white/10 rounded-2xl p-6 text-lg focus:outline-none focus:ring-2 focus:ring-purple-500/50 transition-all resize-none h-32"
                    />
                    <button 
                      onClick={handleGenerateImage}
                      disabled={isGenerating || !imagePrompt.trim()}
                      className="absolute bottom-4 left-4 bg-purple-600 hover:bg-purple-700 disabled:bg-gray-600 px-6 py-2 rounded-xl font-bold transition-all shadow-lg shadow-purple-500/40 flex items-center gap-2"
                    >
                      {isGenerating ? 'جاري العمل...' : <><Zap className="w-5 h-5" /> توليد</>}
                    </button>
                  </div>
                </div>
              </div>

              {isGenerating && !generatedImage && (
                <div className="aspect-square max-w-lg mx-auto rounded-3xl glass flex flex-col items-center justify-center gap-4 animate-pulse">
                  <div className="w-16 h-16 border-4 border-purple-500 border-t-transparent rounded-full animate-spin"></div>
                  <p className="text-purple-400 font-medium">جاري رسم اللوحة الفنية...</p>
                </div>
              )}

              {generatedImage && (
                <div className="relative group max-w-lg mx-auto">
                  <img src={generatedImage} alt="Generated" className="rounded-3xl shadow-2xl border border-white/10 transition-transform group-hover:scale-[1.02]" />
                  <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity rounded-3xl flex items-center justify-center gap-4">
                    <button className="bg-white text-black px-4 py-2 rounded-lg font-bold hover:bg-gray-200">تحميل الصورة</button>
                    <button className="glass px-4 py-2 rounded-lg font-bold hover:bg-white/10">مشاركة</button>
                  </div>
                </div>
              )}
            </div>
          )}

          {activeTab === 'live' && (
            <div className="max-w-4xl mx-auto h-[70vh] flex flex-col items-center justify-center gap-12">
              <div className="relative">
                <div className={`w-48 h-48 rounded-full flex items-center justify-center transition-all duration-700 ${isLiveActive ? 'bg-rose-500/20 scale-110 shadow-[0_0_80px_rgba(244,63,94,0.3)]' : 'bg-slate-800'}`}>
                  {isLiveActive ? (
                    <div className="flex items-center gap-1">
                      {[1,2,3,4,5].map(i => (
                        <div key={i} className={`w-2 bg-rose-500 rounded-full animate-pulse-slow`} style={{ height: `${20 + Math.random() * 60}px`, animationDelay: `${i * 0.1}s` }}></div>
                      ))}
                    </div>
                  ) : (
                    <Mic className="w-20 h-20 text-slate-600" />
                  )}
                </div>
                {isLiveActive && (
                  <div className="absolute -top-4 -right-4 bg-rose-500 px-3 py-1 rounded-full text-xs font-bold animate-pulse">LIVE</div>
                )}
              </div>

              <div className="text-center space-y-4">
                <h2 className="text-2xl font-bold">{isLiveActive ? 'تحدث الآن، أنا أستمع...' : 'مختبر التفاعل الصوتي الحي'}</h2>
                <p className="text-gray-400 max-w-md mx-auto">
                  جرب التفاعل الصوتي الفوري مع جيمناي. يدعم هذا الوضع الحوار الطبيعي بزمن استجابة منخفض جداً.
                </p>
              </div>

              <button 
                onClick={isLiveActive ? stopLiveSession : startLiveSession}
                className={`flex items-center gap-3 px-8 py-4 rounded-full font-bold text-lg transition-all transform hover:scale-105 active:scale-95 shadow-xl ${isLiveActive ? 'bg-white text-black shadow-white/10' : 'bg-rose-600 text-white shadow-rose-500/40'}`}
              >
                {isLiveActive ? (
                  <><StopCircle className="w-6 h-6" /> إنهاء الجلسة</>
                ) : (
                  <><Volume2 className="w-6 h-6" /> ابدأ المحادثة الآن</>
                )}
              </button>
            </div>
          )}
        </div>

        {/* Floating Chat Bar for Text Tab */}
        {activeTab === 'text' && (
          <div className="p-4 md:p-8 bg-gradient-to-t from-slate-950/80 to-transparent">
            <div className="max-w-4xl mx-auto relative group">
              <input 
                type="text"
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleSendMessage()}
                placeholder="اكتب رسالتك هنا..."
                className="w-full bg-slate-900/80 border border-white/10 rounded-2xl py-4 pr-6 pl-16 text-lg focus:outline-none focus:ring-2 focus:ring-indigo-500/50 shadow-2xl transition-all"
              />
              <button 
                onClick={handleSendMessage}
                disabled={isGenerating || !inputText.trim()}
                className="absolute left-3 top-1/2 -translate-y-1/2 bg-indigo-600 p-2.5 rounded-xl hover:bg-indigo-700 disabled:bg-gray-700 transition-all text-white shadow-lg shadow-indigo-500/40"
              >
                <Send className="w-6 h-6" />
              </button>
            </div>
          </div>
        )}
      </main>
    </div>
  );
};

const root = createRoot(document.getElementById('root')!);
root.render(<App />);
