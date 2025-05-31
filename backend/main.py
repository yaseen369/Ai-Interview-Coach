import os
import io
import json
import logging
import httpx
import re
from dotenv import load_dotenv

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
import speech_recognition as sr
from pydub import AudioSegment, silence
import google.generativeai as genai
from google.generativeai.types import RequestOptions

import inspect # 

# Load environment variables from .env file
load_dotenv()

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Interview Coach Backend",
    description="Backend API for an AI-powered interview coach application.",
    version="1.0.0"
)

# --- Configure CORS ---
origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Speech Recognition Initializer ---
r = sr.Recognizer()

# --- Gemini API Configuration ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.error("GEMINI_API_KEY environment variable not set. Please set it in a .env file.")
    raise ValueError("GEMINI_API_KEY environment variable not set.")

genai.configure(api_key=GEMINI_API_KEY)

# --- Session Management ---
session_states = {} # Stores conversation history and interview stage for each client

# --- Helper Functions ---
async def get_gemini_response(prompt: str, client_id: int):
    """Sends a prompt to the Gemini API and returns the response."""
    # Using 'gemini-1.5-pro' as 'gemini-pro' was not found for your API key
    # You can try 'gemini-1.5-flash' if you hit quotas with 'gemini-1.5-pro'
    model = genai.GenerativeModel('gemini-1.5-flash') # <-- CHANGE THIS LINE
    history = session_states[client_id].get("conversation_history", [])

    chat = model.start_chat(history=history)
    try:
        response = await chat.send_message_async(prompt, request_options=RequestOptions(timeout=120))
        text_response = ''.join([part.text for part in response._result.candidates[0].content.parts])

        # Update conversation history
        session_states[client_id]["conversation_history"].append({"role": "user", "parts": [prompt]})
        session_states[client_id]["conversation_history"].append({"role": "model", "parts": [text_response]})

        return text_response
    except httpx.TimeoutException:
        logger.error("Gemini API request timed out.")
        return "I'm sorry, the AI took too long to respond. Please try again."
    except Exception as e:
        logger.error(f"Error calling Gemini API: {e}", exc_info=True)
        return f"I apologize, but I encountered an error: {e}. Please try again."

def clean_gemini_response(text: str) -> str:
    """Cleans up common Gemini markdown and conversational artifacts."""
    
    cleaned_text = re.sub(r'\*\*|__|\*|_', '', text)
    
    cleaned_text = cleaned_text.strip()
    return cleaned_text

async def process_audio_and_transcribe(audio_data: bytes) -> str:
    """Processes audio blob, removes silence, and transcribes using Google Speech Recognition."""
    try:
        # Convert bytes to an in-memory file for pydub
        audio_file = io.BytesIO(audio_data)
        logger.info(f"Received audio blob of size: {len(audio_data)} bytes for processing.")

        # Load audio using pydub
        audio = AudioSegment.from_file(audio_file, format="webm")

        # --- Silence Detection (Updated for 'detect_trailing_silence' absence and 'int' return type) ---
        # Trim leading silence
        start_trim_length = 0
        leading_silence_segments = silence.detect_leading_silence(audio, silence_threshold=-35)
        # Check if it's not 0 (meaning silence was detected) AND it's a list
        if leading_silence_segments and isinstance(leading_silence_segments, list):
            start_trim_length = leading_silence_segments[0][0] # Get the length of the leading silence

        # To detect trailing silence using detect_leading_silence (as 'detect_trailing_silence' is missing):
        # 1. Reverse the audio
        reversed_audio = audio.reverse()
        # 2. Detect leading silence on the reversed audio
        trailing_silence_segments_on_reversed = silence.detect_leading_silence(reversed_audio, silence_threshold=-35)

        # Calculate the actual end_trim length for the original audio
        end_trim_length = 0
        # Check if it's not 0 (meaning silence was detected) AND it's a list
        if trailing_silence_segments_on_reversed and isinstance(trailing_silence_segments_on_reversed, list):
            # The detected leading silence on reversed audio is the trailing silence on original audio
            end_trim_length = trailing_silence_segments_on_reversed[0][0] # Get the length of the silence

        # Apply trimming
        # The audio segment is trimmed from 'start_trim_length' to 'length of audio - end_trim_length'
        trimmed_audio = audio[start_trim_length : len(audio) - end_trim_length]
        # --- End Silence Detection Updates ---

        if len(trimmed_audio) < 100: # If audio is too short after trimming (e.g., just noise)
             logger.warning("Audio too short after trimming. Using original audio for transcription.")
             trimmed_audio = audio # Use original if trimmed is too short


        # Export to WAV for SpeechRecognition
        wav_file_path = io.BytesIO()
        trimmed_audio.export(wav_file_path, format="wav")
        wav_file_path.seek(0)

        # Transcribe using Google Speech Recognition
        with sr.AudioFile(wav_file_path) as source:
            audio_listened = r.record(source)
            try:
                text = r.recognize_google(audio_listened)
                logger.info(f"Transcription successful: {text}")
                return text
            except sr.UnknownValueError:
                logger.warning("Google Speech Recognition could not understand audio")
                return "" # Return empty string if nothing understood
            except sr.RequestError as e:
                logger.error(f"Could not request results from Google Speech Recognition service; {e}")
                raise HTTPException(status_code=500, detail=f"Speech Recognition error: {e}")

    except Exception as e:
        logger.error(f"Error converting audio with pydub/ffmpeg: {e}", exc_info=True)
        ffmpeg_error_msg = str(e)
        error_detail = "Audio processing error. Ensure FFmpeg is correctly installed and in your system's PATH. Also, check browser audio settings."
        if "Decoding failed" in ffmpeg_error_msg or "EBML header parsing failed" in ffmpeg_error_msg:
            error_detail = "Audio format issue: Browser sent malformed audio or recording stopped abruptly. Try a different browser or ensure recording is not interrupted."
        raise e # Re-raise to be caught by the main websocket_endpoint try/except


# --- WebSocket Endpoint ---
MAX_INTERVIEW_QUESTIONS = 3 # Set maximum questions to 3

@app.websocket("/ws/interview")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    client_id = id(websocket)
    # Initialize session state for the client, including question count
    session_states[client_id] = {
        "stage": "initial",
        "conversation_history": [],
        "job_role": None,
        "questions_asked_count": 0
    }
    logger.info(f"WebSocket connection established: {client_id}")
    await websocket.send_json({"type": "feedback", "data": "connection open"})
    logger.info("connection open")

    # --- DEBUGGING BLOCK (can be removed later if stable) ---
    try:
        sig = inspect.signature(silence.detect_leading_silence)
        logger.info(f"DEBUG: Signature of detect_leading_silence: {sig}")
    except Exception as e:
        logger.error(f"DEBUG: Error inspecting detect_leading_silence: {e}")
    # --- END DEBUGGING BLOCK ---

    try:
        while True:
            # Use websocket.receive() to handle both text and binary messages
            message = await websocket.receive()

            if message["type"] == "websocket.receive":
                # Check if it's a text message (JSON) or a binary message (audio)
                if "text" in message:
                    raw_text_message = message["text"]
                    logger.info(f"Full raw WebSocket text message received: {raw_text_message}")
                    json_message = json.loads(raw_text_message)
                    message_type = json_message.get("type")
                    data = json_message.get("data")

                    if message_type == "start_interview":
                        job_role = data.get("jobRole")
                        if job_role:
                            session_states[client_id]["job_role"] = job_role
                            session_states[client_id]["stage"] = "interview_active"
                            session_states[client_id]["conversation_history"] = [] # Clear history for new interview
                            session_states[client_id]["questions_asked_count"] = 0 # Reset question count

                            logger.info(f"Starting interview for job role: {job_role}")
                            initial_prompt = f"You are an AI interview coach. Your task is to conduct a mock interview for a {job_role} position. Ask relevant, challenging questions. Start by asking the first interview question. Do not provide any feedback or next questions until I provide an answer to your current question. Ask only one question at a time. The interview will consist of a maximum of {MAX_INTERVIEW_QUESTIONS} questions."
                            logger.info(f"Generating first question for job role: {job_role}")
                            logger.info(f"Sending prompt to LLM: {initial_prompt[:100]}...")
                            question = await get_gemini_response(initial_prompt, client_id)
                            question = clean_gemini_response(question)
                            session_states[client_id]["questions_asked_count"] += 1 # Increment count for first question
                            await websocket.send_json({"type": "question", "data": question})
                        else:
                            await websocket.send_json({"type": "error", "data": "Job role not provided."})

                    elif message_type == "text_answer":
                        answer_text = data.get("answer")
                        if answer_text:
                            logger.info(f"Received text answer: {answer_text}")
                            # Frontend now updates its own answer display immediately.
                            # No need for simulate_text_answer from backend.

                            job_role = session_states[client_id].get("job_role", "a position")
                            questions_asked_count = session_states[client_id]["questions_asked_count"]

                            feedback_prompt_base = f"The candidate answered: '{answer_text}'. Provide concise, constructive feedback (1-3 sentences) on this answer for a {job_role} interview, focusing on clarity, completeness, and relevance."

                            if questions_asked_count >= MAX_INTERVIEW_QUESTIONS:
                                feedback_prompt = feedback_prompt_base + " This is the final question of the interview. Do NOT ask for a 'Next Question:'. Just provide the feedback, and conclude the interview. Structure your response strictly as: 'Feedback: [Your feedback here]. Interview complete! Well done!'"
                            else:
                                feedback_prompt = feedback_prompt_base + " Then, ask the next logical interview question. Structure your response strictly as: 'Feedback: [Your feedback here]. Next Question: [Your next question here]'"

                            logger.info(f"Sending feedback prompt to LLM: {feedback_prompt[:100]}...")
                            response_from_gemini = await get_gemini_response(feedback_prompt, client_id)
                            response_from_gemini = clean_gemini_response(response_from_gemini)
                            logger.info(f"Received feedback from LLM: {response_from_gemini}")

                            feedback_match = re.search(r"Feedback: (.*?)(?:Next Question: (.*)|Interview complete! (.*))", response_from_gemini, re.DOTALL)

                            if feedback_match:
                                feedback_text = feedback_match.group(1).strip()
                                next_question_text = feedback_match.group(2).strip() if feedback_match.group(2) else None
                                interview_complete_message = feedback_match.group(3).strip() if feedback_match.group(3) else None

                                if interview_complete_message:
                                    await websocket.send_json({"type": "feedback", "feedback": feedback_text, "next_question": None, "interview_complete": True})

                                    summary_prompt = f"""Based on the complete interview conversation for a {job_role} position, provide a comprehensive summary of the candidate's performance. Analyze the questions asked and the candidate's responses from the entire conversation history: {session_states[client_id]['conversation_history']}.
                                    Focus on:
                                    - **Areas for Improvement:** 2-3 specific points where the candidate could enhance their answers or approach, e.g., "Improve conciseness", "Elaborate more on technical depth".
                                    - **Considerations for Future Answers:** 2-3 key things the candidate should keep in mind when formulating responses to similar questions, e.g., "Connect answers to company values", "Provide specific examples".
                                    - **Things to Avoid:** 1-2 common pitfalls or mistakes to steer clear of, e.g., "Avoid generic statements", "Do not interrupt the interviewer".

                                    Present this information clearly with distinct headings for each section. Ensure the language is constructive, encouraging, and highly professional. Do not include any conversational remarks or personal greetings, just the structured summary sections.
                                    """
                                    logger.info(f"Generating interview summary for {client_id}")
                                    interview_summary = await get_gemini_response(summary_prompt, client_id)
                                    interview_summary = clean_gemini_response(interview_summary)
                                    logger.info(f"Generated summary: {interview_summary}")

                                    await websocket.send_json({"type": "summary", "data": interview_summary})
                                    session_states[client_id]["stage"] = "interview_complete"
                                elif next_question_text:
                                    session_states[client_id]["questions_asked_count"] += 1 # Increment for the new question
                                    await websocket.send_json({"type": "feedback", "feedback": feedback_text, "next_question": next_question_text})
                                else:
                                    # Fallback if regex doesn't fully match but feedback is present
                                    logger.warning(f"Failed to parse next question from LLM response, but feedback present: {response_from_gemini}")
                                    await websocket.send_json({"type": "feedback", "feedback": feedback_text, "next_question": "Can you elaborate further?"}) # Fallback question
                            else:
                                # If regex fails entirely, send the raw response as feedback and indicate no next question
                                logger.warning(f"Failed to parse feedback/next question from LLM response: {response_from_gemini}")
                                await websocket.send_json({"type": "feedback", "feedback": f"Raw LLM response: {response_from_gemini}. (Parsing error, please ensure LLM follows format.)", "next_question": None, "interview_complete": True}) # Assume complete on parsing failure

                        else:
                            await websocket.send_json({"type": "error", "data": "Text answer not provided."})

                elif "bytes" in message: # This branch now handles binary audio data
                    audio_data = message["bytes"] # Extract the actual bytes data
                    logger.info("Received audio blob, proceeding to process.")
                    try:
                        transcribed_text = await process_audio_and_transcribe(audio_data)
                        logger.info(f"Transcribed audio: {transcribed_text}")
                        # Frontend now updates its own answer display immediately.
                        # We still send transcription for status/logging, but not for UI update of the answer field.
                        await websocket.send_json({"type": "transcription", "data": transcribed_text})

                        job_role = session_states[client_id].get("job_role", "a position")
                        questions_asked_count = session_states[client_id]["questions_asked_count"]

                        if not transcribed_text:
                            feedback_prompt = f"The candidate provided an unclear or silent answer. Provide feedback to encourage them to speak clearly or try again, then ask them to re-answer the current question or rephrase it slightly. For example, 'I couldn't quite catch that. Could you please rephrase your answer?' or 'I didn't detect any speech. Please ensure your microphone is working and speak clearly.' Then ask the original question again or a slightly rephrased version."
                            logger.warning("Empty transcription received, asking for re-answer.")
                            response_from_gemini = await get_gemini_response(feedback_prompt, client_id)
                            response_from_gemini = clean_gemini_response(response_from_gemini)
                            logger.info(f"Received feedback for empty transcription: {response_from_gemini}")

                            feedback_match = re.search(r"Feedback: (.*?)(?:Next Question: (.*)|Interview complete! (.*))", response_from_gemini, re.DOTALL)
                            if feedback_match:
                                feedback_text = feedback_match.group(1).strip()
                                next_question_text = feedback_match.group(2).strip() if feedback_match.group(2) else None
                                await websocket.send_json({"type": "feedback", "feedback": feedback_text, "next_question": next_question_text})
                            else:
                                await websocket.send_json({"type": "feedback", "feedback": "Please speak clearly or rephrase your answer.", "next_question": "Can you please repeat your answer?"})

                        else:
                            feedback_prompt_base = f"The candidate answered: '{transcribed_text}'. Provide concise, constructive feedback (1-3 sentences) on this answer for a {job_role} interview, focusing on clarity, completeness, and relevance."

                            if questions_asked_count >= MAX_INTERVIEW_QUESTIONS:
                                feedback_prompt = feedback_prompt_base + " This is the final question of the interview. Do NOT ask for a 'Next Question:'. Just provide the feedback, and conclude the interview. Structure your response strictly as: 'Feedback: [Your feedback here]. Interview complete! Well done!'"
                            else:
                                feedback_prompt = feedback_prompt_base + " Then, ask the next logical interview question. Structure your response strictly as: 'Feedback: [Your feedback here]. Next Question: [Your next question here]'"

                            logger.info(f"Sending feedback prompt to LLM: {feedback_prompt[:100]}...")
                            response_from_gemini = await get_gemini_response(feedback_prompt, client_id)
                            response_from_gemini = clean_gemini_response(response_from_gemini)
                            logger.info(f"Received feedback from LLM: {response_from_gemini}")

                            feedback_match = re.search(r"Feedback: (.*?)(?:Next Question: (.*)|Interview complete! (.*))", response_from_gemini, re.DOTALL)

                            if feedback_match:
                                feedback_text = feedback_match.group(1).strip()
                                next_question_text = feedback_match.group(2).strip() if feedback_match.group(2) else None
                                interview_complete_message = feedback_match.group(3).strip() if feedback_match.group(3) else None

                                if interview_complete_message:
                                    await websocket.send_json({"type": "feedback", "feedback": feedback_text, "next_question": None, "interview_complete": True})

                                    summary_prompt = f"""Based on the complete interview conversation for a {job_role} position, provide a comprehensive summary of the candidate's performance. Analyze the questions asked and the candidate's responses from the entire conversation history: {session_states[client_id]['conversation_history']}.
                                    Focus on:
                                    - **Areas for Improvement:** 2-3 specific points where the candidate could enhance their answers or approach, e.g., "Improve conciseness", "Elaborate more on technical depth".
                                    - **Considerations for Future Answers:** 2-3 key things the candidate should keep in mind when formulating responses to similar questions, e.g., "Connect answers to company values", "Provide specific examples".
                                    - **Things to Avoid:** 1-2 common pitfalls or mistakes to steer clear of, e.g., "Avoid generic statements", "Do not interrupt the interviewer".

                                    Present this information clearly with distinct headings for each section. Ensure the language is constructive, encouraging, and highly professional. Do not include any conversational remarks or personal greetings, just the structured summary sections.
                                    """
                                    logger.info(f"Generating interview summary for {client_id}")
                                    interview_summary = await get_gemini_response(summary_prompt, client_id)
                                    interview_summary = clean_gemini_response(interview_summary)
                                    logger.info(f"Generated summary: {interview_summary}")

                                    await websocket.send_json({"type": "summary", "data": interview_summary})
                                    session_states[client_id]["stage"] = "interview_complete"
                                elif next_question_text:
                                    session_states[client_id]["questions_asked_count"] += 1 # Increment for the new question
                                    await websocket.send_json({"type": "feedback", "feedback": feedback_text, "next_question": next_question_text})
                                else:
                                    # Fallback if regex doesn't fully match but feedback is present
                                    logger.warning(f"Failed to parse next question from LLM response, but feedback present: {response_from_gemini}")
                                    await websocket.send_json({"type": "feedback", "feedback": feedback_text, "next_question": "Can you elaborate further?"}) # Fallback question
                            else:
                                # If regex fails entirely, send the raw response as feedback and indicate no next question
                                logger.warning(f"Failed to parse feedback/next question from LLM response: {response_from_gemini}")
                                await websocket.send_json({"type": "feedback", "feedback": f"Raw LLM response: {response_from_gemini}. (Parsing error, please ensure LLM follows format.)", "next_question": None, "interview_complete": True}) # Assume complete on parsing failure

                    except Exception as e:
                        logger.error(f"Error during audio processing or transcription: {e}", exc_info=True)
                        # Send specific error related to audio processing
                        await websocket.send_json({"type": "error", "data": f"Audio processing or transcription failed: {e}. Ensure FFmpeg is installed and browser audio settings are correct."})
                else:
                    await websocket.send_json({"type": "error", "data": "Received empty message or unexpected content type from WebSocket."})
            elif message["type"] == "websocket.disconnect":
                logger.info(f"Client disconnected from WebSocket. ID: {client_id}")
                break # Exit the loop if client disconnects
            else:
                logger.warning(f"Received unexpected WebSocket message type: {message['type']}")
                await websocket.send_json({"type": "error", "data": f"Unexpected WebSocket message type: {message['type']}"})

    except WebSocketDisconnect:
        logger.info(f"Client disconnected from WebSocket. ID: {client_id}")
        if client_id in session_states:
            del session_states[client_id] # Clean up session state
            logger.info(f"Cleaned up session state for disconnected client: {client_id}")
    except RuntimeError as e:
        logger.warning(f"RuntimeError in websocket endpoint (likely connection closed): {e}")
    except Exception as e:
        logger.critical(f"An unhandled error occurred in WebSocket endpoint: {e}", exc_info=True)
        try:
            # Send a more specific error message if possible
            if isinstance(e, json.JSONDecodeError):
                error_msg = f"Backend error: Invalid JSON format received ({e})."
            elif isinstance(e, KeyError):
                error_msg = f"Backend error: Missing expected data in message ({e}). Check frontend message format."
            else:
                error_msg = f"An unhandled backend error occurred: {e}. Check backend logs."
            await websocket.send_json({"type": "error", "data": error_msg})
        except RuntimeError:
            logger.debug("Attempted to send error on already closed WebSocket during unhandled error.")
    finally:
        logger.info("WebSocket connection handler exiting.")