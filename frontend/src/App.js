import React, { useState, useRef, useEffect, useCallback } from 'react';
import './App.css';
import {
  FaMicrophone, FaStop, FaPlay, FaSpinner, FaRedo, FaVolumeUp, FaCheckCircle,
  FaVolumeMute, FaLightbulb // Kept FaLightbulb for a potential "tips" feature if needed, though not explicitly styled yet.
} from 'react-icons/fa';
import { SiGooglegemini } from "react-icons/si"; // Using SiGooglegemini for a cool icon

function App() {
  const wsRef = useRef(null);
  const [isConnected, setIsConnected] = useState(false);
  const [status, setStatus] = useState('Connecting to backend...');
  const [mediaRecorder, setMediaRecorder] = useState(null);
  const [audioChunks, setAudioChunks] = useState([]);
  const audioChunksRef = useRef([]); // Use ref for immediate access to chunks
  const [isRecording, setIsRecording] = useState(false);
  const [jobRole, setJobRole] = useState('');
  const [interviews, setInterviews] = useState([]);
  const currentQuestionIndexRef = useRef(0); // This ref will hold the index of the CURRENT question being answered/displayed
  const reconnectTimeoutRef = useRef(null);
  const reconnectInterval = useRef(1000);
  const [isLoadingQuestions, setIsLoadingQuestions] = useState(false);
  const [isProcessingAudio, setIsProcessingAudio] = useState(false);
  const interviewReviewRef = useRef(null);
  const [errorStatus, setErrorStatus] = useState('');
  const audioContextRef = useRef(null);
  const analyserRef = useRef(null);
  const [micVolume, setMicVolume] = useState(0);
  const [textInput, setTextInput] = useState('');
  const [isTextInputMode, setIsTextInputMode] = useState(false);
  const [isInterviewStarted, setIsInterviewStarted] = useState(false);
  const [hasInterviewFinished, setHasInterviewFinished] = useState(false);
  const [isMuted, setIsMuted] = useState(false);
  const [interviewSummary, setInterviewSummary] = useState(''); // New state for interview summary


  // --- WebSocket Connection Management ---
  const connectWebSocket = useCallback(() => {
    if (wsRef.current && (wsRef.current.readyState === WebSocket.OPEN || wsRef.current.readyState === WebSocket.CONNECTING)) {
      console.log("WebSocket already open or connecting.");
      return;
    }

    const ws = new WebSocket('ws://localhost:8000/ws/interview');
    wsRef.current = ws;

    ws.onopen = () => {
      console.log('WebSocket connected');
      setIsConnected(true);
      setStatus('Connected to backend.');
      setErrorStatus(''); // Clear any previous error
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
        reconnectTimeoutRef.current = null;
      }
      reconnectInterval.current = 1000; // Reset reconnect interval on successful connection
    };

    ws.onmessage = async (event) => {
      const message = JSON.parse(event.data);
      console.log("WebSocket message received:", message);

      switch (message.type) {
        case 'question':
          setStatus('AI Coach asked a question.');
          setInterviews(prev => {
            const newInterview = { question: message.data, answer: '', feedback: '' };
            // Add the new question to the end of the interviews array
            const updatedInterviews = [...prev, newInterview];
            // Update the ref to point to the index of the newly added question
            currentQuestionIndexRef.current = updatedInterviews.length - 1;
            return updatedInterviews;
          });
          setIsLoadingQuestions(false);
          setIsProcessingAudio(false); // Ensure audio processing state is off
          await speakText(message.data);
          break;

        case 'transcription':
          setStatus('You said: ' + message.data);
          setInterviews(prev => {
            const updated = [...prev];
            // Ensure currentQuestionIndexRef.current points to the correct question
            // and update its answer with the transcribed text.
            if (updated[currentQuestionIndexRef.current]) {
              updated[currentQuestionIndexRef.current].answer = message.data;
            }
            return updated;
          });
          break;

        case 'feedback':
          setStatus('Received feedback.');
          setIsProcessingAudio(false); // Feedback received, stop processing indicator

          setInterviews(prev => {
            const updatedInterviews = [...prev];
            const currentItemIndex = currentQuestionIndexRef.current; // The item that was just answered

            // Apply feedback to the current interview item
            if (updatedInterviews[currentItemIndex]) {
              updatedInterviews[currentItemIndex].feedback = message.feedback;
            }

            if (message.next_question) {
              // Add the next question to the array
              updatedInterviews.push({ question: message.next_question, answer: '', feedback: '' });
              // Update the ref to point to the newly added question
              currentQuestionIndexRef.current = updatedInterviews.length - 1;
              setStatus('AI Coach asked next question.');
            } else {
              // No next question means interview is complete
              setHasInterviewFinished(true);
              setStatus('Interview complete!');
            }
            return updatedInterviews;
          });

          // Speak feedback and next question (if any)
          if (message.next_question) {
            await speakText(message.feedback + ". " + message.next_question);
          } else {
            await speakText(message.feedback + ". Interview complete. Well done!");
          }
          break;

        case 'summary':
          setInterviewSummary(message.data);
          console.log("Interview Summary Received:", message.data);
          // Optionally speak the summary or a concluding remark after the summary is displayed
          // await speakText("Here is a summary of your interview performance.");
          break;

        case 'interview_complete':
          setHasInterviewFinished(true);
          setStatus('Interview complete!');
          setIsProcessingAudio(false);
          break;

        case 'error':
          setErrorStatus(`Error: ${message.data}`);
          setStatus('Error occurred.');
          setIsLoadingQuestions(false);
          setIsProcessingAudio(false);
          console.error("Backend Error:", message.data);
          // If a critical error occurs, might need to reset interview state
          setJobRole('');
          setInterviews([]);
          currentQuestionIndexRef.current = 0;
          setIsInterviewStarted(false);
          setHasInterviewFinished(false);
          setInterviewSummary(''); // Clear summary on error
          break;

        default:
          console.warn('Unknown message type:', message.type);
      }

      // Scroll to bottom of interview review panel
      if (interviewReviewRef.current) {
        interviewReviewRef.current.scrollTop = interviewReviewRef.current.scrollHeight;
      }
    };

    ws.onclose = (event) => {
      setIsConnected(false);
      setStatus('Disconnected from backend. Reconnecting...');
      console.log('WebSocket disconnected:', event.code, event.reason);
      if (!event.wasClean) {
        // Attempt to reconnect if disconnect was not clean (e.g., server went down)
        reconnectTimeoutRef.current = setTimeout(connectWebSocket, reconnectInterval.current);
        reconnectInterval.current = Math.min(reconnectInterval.current * 2, 30000); // Exponential backoff, max 30s
      }
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      setErrorStatus('WebSocket error. See console for details.');
      setStatus('Connection error.');
      ws.close(); // Close the socket to trigger onclose and reconnect logic
    };
  }, []); // Removed 'interviews' from dependency array to prevent unnecessary recreations

  useEffect(() => {
    connectWebSocket();
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
    };
  }, [connectWebSocket]);

  // --- Speech Synthesis ---
  const speakText = async (text) => {
    if (isMuted) {
      console.log("Speech synthesis muted.");
      return;
    }

    if ('speechSynthesis' in window) {
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.lang = 'en-US';
      utterance.pitch = 1;
      utterance.rate = 1;
      // Optional: Set voice
      const voices = window.speechSynthesis.getVoices();
      // Prefer Google voices for a better quality, or a general English voice
      const preferredVoices = voices.filter(voice =>
        voice.lang === 'en-US' && (voice.name.includes('Google') || voice.name.includes('Microsoft') || voice.name.includes('default'))
      );
      if (preferredVoices.length > 0) {
        utterance.voice = preferredVoices[0]; // Take the first preferred one
      } else {
        // Fallback to any English voice
        const enUsVoice = voices.find(voice => voice.lang === 'en-US');
        if (enUsVoice) {
          utterance.voice = enUsVoice;
        }
      }

      window.speechSynthesis.speak(utterance);
    } else {
      console.warn('Speech synthesis not supported in this browser.');
    }
  };

  // --- Audio Recording Logic ---
  const startRecording = async () => {
    setErrorStatus(''); // Clear any previous error
    setAudioChunks([]);
    audioChunksRef.current = []; // Clear the ref as well

    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        const mediaRecorderInstance = new MediaRecorder(stream, { mimeType: 'audio/webm' });

        mediaRecorderInstance.ondataavailable = (event) => {
          if (event.data.size > 0) {
            setAudioChunks((prev) => [...prev, event.data]);
            audioChunksRef.current.push(event.data);
            console.log("Audio chunk available, size:", event.data.size, "Total chunks in ref:", audioChunksRef.current.length);
          } else {
            console.log("Audio chunk available, but size is 0. Event data:", event.data);
          }
        };

        mediaRecorderInstance.onstop = async () => {
          setIsProcessingAudio(true); // Set processing state
          console.log("MediaRecorder stopped. Final audioChunksRef.current:", audioChunksRef.current);
          const audioBlob = new Blob(audioChunksRef.current, { type: mediaRecorderInstance.mimeType });
          console.log("Audio Blob created. Size:", audioBlob.size, "Type:", audioBlob.type);

          if (audioBlob.size > 0 && wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
            try {
              // Set the answer in the frontend immediately after recording stops
              // This is a placeholder, actual transcription comes from backend
              setInterviews(prev => {
                const updated = [...prev];
                if (updated[currentQuestionIndexRef.current]) {
                  updated[currentQuestionIndexRef.current].answer = 'Processing audio...'; // Indicate processing
                }
                return updated;
              });
              await wsRef.current.send(audioBlob);
              console.log("Audio Blob sent to backend.");
              setAudioChunks([]); // Clear chunks after sending
              audioChunksRef.current = []; // Clear ref as well
            } catch (error) {
              console.error("Error sending audio blob over WebSocket:", error);
              setErrorStatus("Failed to send audio. Please check your connection.");
              setIsProcessingAudio(false); // Reset processing state on send error
            }
          } else {
            const errorMessage = audioBlob.size === 0
              ? "No audio recorded or connection lost. Please ensure your microphone is working."
              : "WebSocket connection not open.";
            setErrorStatus(errorMessage);
            console.error(errorMessage);
            setIsProcessingAudio(false); // Reset processing state if not sent
          }
          // Stop all tracks in the stream
          stream.getTracks().forEach(track => track.stop());
        };

        mediaRecorderInstance.onerror = (event) => {
          console.error("MediaRecorder error:", event.error);
          setErrorStatus(`Recording error: ${event.error.name}.`);
          setIsRecording(false);
          setIsProcessingAudio(false);
          // Stop all tracks in the stream on error
          stream.getTracks().forEach(track => track.stop());
        };


        mediaRecorderInstance.start();
        console.log("MediaRecorder started. State:", mediaRecorderInstance.state);
        setMediaRecorder(mediaRecorderInstance);
        setIsRecording(true);
        setStatus('Recording...');
        setupAudioVisualizer(stream);

      } catch (err) {
        console.error('Error accessing microphone:', err);
        setErrorStatus(`Microphone access denied or error: ${err.name}. Please ensure permissions are granted.`);
        setIsRecording(false);
        setIsProcessingAudio(false);
      }
    } else {
      setErrorStatus("Cannot record: Not connected to backend.");
      console.error("Cannot record: WebSocket not open.");
    }
  };

  const stopRecording = () => {
    if (mediaRecorder) {
      mediaRecorder.stop();
      setIsRecording(false);
      setStatus('Processing audio...');
      // The onstop event handler will now manage sending the blob and setting processing state
    }
  };

  const setupAudioVisualizer = (stream) => {
    if (!audioContextRef.current) {
      audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)();
    }
    const audioContext = audioContextRef.current;

    const source = audioContext.createMediaStreamSource(stream);
    const analyser = audioContext.createAnalyser();
    analyser.fftSize = 256;
    source.connect(analyser);
    analyserRef.current = analyser;

    const dataArray = new Uint8Array(analyser.frequencyBinCount);

    const updateVolume = () => {
      analyser.getByteFrequencyData(dataArray);
      let sum = 0;
      for (let i = 0; i < dataArray.length; i++) {
        sum += dataArray[i];
      }
      let average = sum / dataArray.length;
      setMicVolume(average);
      if (isRecording) {
        requestAnimationFrame(updateVolume);
      }
    };
    updateVolume();
  };

  useEffect(() => {
    // Cleanup visualizer on unmount or recording stop
    if (!isRecording && audioContextRef.current) {
      if (audioContextRef.current.state === 'running') {
        // audioContextRef.current.close(); // Closing context can cause issues for subsequent recordings
      }
      // Ensure all tracks are stopped when recording stops
      if (mediaRecorder && mediaRecorder.stream) {
        mediaRecorder.stream.getTracks().forEach(track => track.stop());
      }
    }
  }, [isRecording, mediaRecorder]);


  // --- Interview Flow Logic ---
  const startInterview = async () => {
    if (!jobRole.trim()) {
      setErrorStatus("Please enter a job role to start the interview.");
      return;
    }
    setErrorStatus(''); // Clear any previous error
    setIsInterviewStarted(true);
    setHasInterviewFinished(false);
    setInterviews([]); // Clear previous interviews
    currentQuestionIndexRef.current = 0; // Reset index for new interview
    setInterviewSummary(''); // Clear previous summary
    setIsLoadingQuestions(true);
    setStatus('Starting interview...');

    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      try {
        await wsRef.current.send(JSON.stringify({ type: "start_interview", data: { jobRole: jobRole } }));
        console.log("Start interview message sent for role:", jobRole);
      } catch (error) {
        console.error("Failed to send start_interview message:", error);
        setErrorStatus("Failed to start interview. Please check connection.");
        setIsLoadingQuestions(false);
        setIsInterviewStarted(false);
      }
    } else {
      setErrorStatus("Not connected to backend. Please wait or refresh.");
      setIsLoadingQuestions(false);
      setIsInterviewStarted(false);
    }
  };

  const handleTextInputSubmit = async () => {
    if (!textInput.trim()) {
      setErrorStatus("Please enter your answer.");
      return;
    }
    setErrorStatus('');
    setIsProcessingAudio(true); // Use this for text input as well to show processing
    setStatus('Sending text answer for feedback...');

    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      try {
        // Update UI with the text answer immediately
        setInterviews(prev => {
          const updated = [...prev];
          if (updated[currentQuestionIndexRef.current]) {
            updated[currentQuestionIndexRef.current].answer = textInput;
          }
          return updated;
        });

        await wsRef.current.send(JSON.stringify({ type: "text_answer", data: { answer: textInput } }));
        console.log("Text answer sent:", textInput);
        setTextInput(''); // Clear input after sending
      } catch (error) {
        console.error("Failed to send text answer:", error);
        setErrorStatus("Failed to send answer. Please check your connection.");
        setIsProcessingAudio(false);
      }
    } else {
      setErrorStatus("Not connected to backend.");
      setIsProcessingAudio(false);
    }
  };

  const startNewInterview = () => {
    setJobRole('');
    setInterviews([]);
    currentQuestionIndexRef.current = 0;
    setJobRole(''); // Clear job role input
    setIsInterviewStarted(false);
    setHasInterviewFinished(false);
    setErrorStatus('');
    setStatus('Ready to start a new interview.');
    setInterviewSummary(''); // Clear summary for new interview
    connectWebSocket(); // Reconnect websocket just in case
  };

  // Scroll to the latest question/answer in the review panel
  useEffect(() => {
    if (interviewReviewRef.current) {
      // Small delay to ensure render before scroll
      setTimeout(() => {
        interviewReviewRef.current.scrollTop = interviewReviewRef.current.scrollHeight;
      }, 100);
    }
  }, [interviews]); // Trigger scroll when interviews state updates

  const toggleMute = () => {
    setIsMuted(prev => {
      const newState = !prev;
      if (newState) { // If muting
        if ('speechSynthesis' in window) {
          window.speechSynthesis.cancel(); // Stop any ongoing speech
        }
      }
      return newState;
    });
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>
          <SiGooglegemini className="gemini-icon" /> Yaseen's AI Interview Coach
        </h1>
        <p className="subtitle">Master Your Interviews with AI-Powered Feedback</p>
      </header>

      <div className="container">
        {!isInterviewStarted ? (
          <div className="panel start-panel">
            <h2>Start Your Interview</h2>
            {errorStatus && <p className="error-message">{errorStatus}</p>}
            <div className="input-group">
              <input
                type="text"
                placeholder="Enter job role (e.g., 'Software Engineer', 'Product Manager')"
                value={jobRole}
                onChange={(e) => setJobRole(e.target.value)}
                onKeyPress={(e) => {
                  if (e.key === 'Enter') {
                    startInterview();
                  }
                }}
                className="form-control"
                disabled={isLoadingQuestions}
              />
              <button
                onClick={startInterview}
                className="btn btn-primary"
                disabled={!isConnected || isLoadingQuestions || !jobRole.trim()}
              >
                {isLoadingQuestions ? (
                  <>
                    <FaSpinner className="spinner" /> Generating...
                  </>
                ) : (
                  'Start Interview'
                )}
              </button>
            </div>
            <p className="status-message">{status}</p>
          </div>
        ) : (
          <div className="interview-container">
            <div className="interview-panel panel">
              {errorStatus && <p className="error-message">{errorStatus}</p>}
              <p className="status-message">{status}</p>

              {!hasInterviewFinished ? (
                <>
                  <div className="question-display">
                    {interviews[currentQuestionIndexRef.current] && (
                      // Apply key to force re-animation when question changes
                      <h3 key={interviews[currentQuestionIndexRef.current].question}>
                        Q{currentQuestionIndexRef.current + 1}: {interviews[currentQuestionIndexRef.current].question}
                      </h3>
                    )}
                    {isLoadingQuestions && (
                      <p className="loading-indicator">
                        <FaSpinner className="spinner" /> Generating next question...
                      </p>
                    )}
                  </div>

                  <div className="controls">
                    <button
                      onClick={isRecording ? stopRecording : startRecording}
                      className={`btn ${isRecording ? 'btn-danger' : 'btn-primary'}`}
                      disabled={!isConnected || isLoadingQuestions || isProcessingAudio || !interviews[currentQuestionIndexRef.current]?.question}
                    >
                      {isRecording ? <FaStop /> : <FaMicrophone />}
                      {isRecording ? ' Stop Recording' : ' Start Recording'}
                    </button>
                    <button
                      onClick={toggleMute}
                      className="btn btn-secondary"
                      title={isMuted ? "Unmute AI Voice" : "Mute AI Voice"}
                    >
                      {isMuted ? <FaVolumeMute /> : <FaVolumeUp />} AI Voice
                    </button>
                  </div>

                  {isRecording && (
                    <div className="mic-volume-indicator">
                      <div className="volume-bar" style={{ width: `${micVolume * 1.5}px` }}></div> {/* Scale volume for better visual */}
                      <span className="recording-text-indicator">Recording audio...</span>
                    </div>
                  )}
                  {isProcessingAudio && !isRecording && (
                    <p className="processing-text-indicator">
                      <FaSpinner className="spinner" /> Processing audio and generating feedback...
                    </p>
                  )}

                  <div className="text-input-section">
                    <button
                      onClick={() => setIsTextInputMode(!isTextInputMode)}
                      className="btn btn-info toggle-text-input"
                      disabled={isLoadingQuestions || isProcessingAudio}
                    >
                      {isTextInputMode ? 'Use Microphone Instead' : 'Type Answer Instead'}
                    </button>
                    {isTextInputMode && (
                      <div className="text-input-group">
                        <textarea
                          placeholder="Type your answer here..."
                          value={textInput}
                          onChange={(e) => setTextInput(e.target.value)}
                          className="form-control text-area"
                          rows="4"
                          disabled={isProcessingAudio}
                        ></textarea>
                        <button
                          onClick={handleTextInputSubmit}
                          className="btn btn-success"
                          disabled={!textInput.trim() || isProcessingAudio}
                        >
                          Submit Answer
                        </button>
                      </div>
                    )}
                  </div>
                </>
              ) : (
                <div className="interview-complete-message">
                  <p><FaCheckCircle className="success-icon" /> All questions answered! Well done! 🎉</p>
                  {interviewSummary && (
                    <div className="interview-summary-section">
                      <h3>Interview Summary & Tips</h3>
                      <div dangerouslySetInnerHTML={{ __html: interviewSummary.replace(/\n/g, '<br/>') }} />
                    </div>
                  )}
                  <button onClick={startNewInterview} className="btn btn-success">
                    <FaRedo /> Start New Interview
                  </button>
                </div>
              )}
            </div>

            <div className="interview-review-panel panel">
              <h3>Your Interview Summary</h3>
              <div className="interview-review" ref={interviewReviewRef}>
                {interviews.map((item, index) => (
                  <div key={index} className="interview-item">
                    <h4>Q{index + 1}: {item.question}</h4>
                    <p className="your-answer">
                      <strong>Your Answer:</strong>
                      {/* Display actual answer, or processing indicator if current question and processing */}
                      {item.answer ? item.answer :
                       (index === currentQuestionIndexRef.current && isProcessingAudio ? <span className="processing-text-indicator">Processing answer...</span> : 'Not answered yet.')}
                    </p>
                    <p className="feedback">
                      <strong>Feedback:</strong>
                      {/* Display actual feedback, or generating indicator if current question and processing */}
                      {item.feedback ? item.feedback :
                       (index === currentQuestionIndexRef.current && isProcessingAudio ? <span className="processing-text-indicator">Generating feedback...</span> : 'Awaiting feedback.')}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;