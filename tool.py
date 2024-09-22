# tool.py

import time
import requests
import logging
import json
import os  # Added for environment variable handling

# Configure logging
logging.basicConfig(level=logging.INFO, filename='tool.log', 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Eden AI API key
API_KEY = os.getenv('EDEN_API_KEY')

# Base URLs
BASE_URL = 'https://api.edenai.run/v2/'

# Step 1: Transcribe audio
def transcribe_audio(file_path):
    """
    Transcribe the audio file into text with timestamps using Eden AI Speech-to-Text Async API.
    """
    url = BASE_URL + 'audio/speech_to_text_async'
    headers = {
        'Authorization': f'Bearer {API_KEY}',
        'accept': 'application/json'
    }
    data = {
        'providers': 'deepgram',
        'language': 'en-US',
        'provider_params': json.dumps({
            'deepgram': {
                'model': 'general',
                'punctuate': True,
                'diarize': True,
                'detect_topics': False,
                'auto_highlights': False,
                'smart_format': True,
                'utterances': True,
                'timestamps': True
            }
        }),
        'show_original_response': False
    }
    try:
        with open(file_path, 'rb') as f:
            files = {'file': f}
            logging.info("Submitting transcription job...")
            response = requests.post(url, headers=headers, data=data, files=files)
            response.raise_for_status()
            response_data = response.json()
            logging.info("Transcription job submitted successfully.")
            logging.debug(f"Transcription response: {response_data}")

            # Retrieve 'public_id' from response
            if 'public_id' not in response_data:
                logging.error(f"'public_id' not found in response: {response_data}")
                raise KeyError("'public_id' not found in transcription response.")

            job_id = response_data['public_id']
            logging.info(f"Job ID: {job_id}")
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to send transcription request: {e.response.content if e.response else e}")
        raise
    
    # Poll for job completion
    transcription_result = None
    while True:
        time.sleep(5)
        job_url = BASE_URL + f'audio/speech_to_text_async/{job_id}'
        try:
            job_response = requests.get(job_url, headers=headers)
            job_response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to retrieve transcription job status: {e.response.content if e.response else e}")
            raise
    
        job_data = job_response.json()
        logging.debug(f"Job status response: {json.dumps(job_data, indent=2)}")
        status = job_data.get('status')
        logging.info(f"Current job status: {status}")
    
        if status == 'finished':
            try:
                results = job_data['results']
                provider_name = list(results.keys())[0]
                transcription_data = results[provider_name]
                logging.info("Transcription job finished successfully.")
                logging.debug(f"Transcription data: {json.dumps(transcription_data, indent=2)}")
                break
            except KeyError as e:
                logging.error(f"Expected key not found in transcription result: {e}")
                logging.debug(f"Full job data: {json.dumps(job_data, indent=2)}")
                raise
        elif status == 'failed':
            logging.error('Transcription job failed.')
            raise Exception('Transcription job failed.')
        else:
            logging.info("Transcription job still in progress...")
    
    return transcription_data

# Step 2: Detect emotions
def detect_emotions(text):
    """
    Detect emotions in the transcribed text using Eden AI Emotion Detection API.
    """
    url = BASE_URL + 'text/emotion_detection'
    headers = {'Authorization': f'Bearer {API_KEY}'}
    data = {
        'providers': 'openai',
        'language': 'en',
        'text': text
    }
    try:
        logging.info("Detecting emotions...")
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to detect emotions: {e.response.content if e.response else e}")
        raise
    
    emotions_data = response.json().get('results', {}).get('openai', {}).get('items', [])
    logging.info("Emotion detection completed.")
    logging.debug(f"Emotions detected: {emotions_data}")
    return emotions_data

# Step 3: Identify specific indicators
def detect_indicators(text):
    """
    Identify specific indicators in the text using Eden AI Custom Named Entity Recognition API.
    """
    url = BASE_URL + 'text/custom_named_entity_recognition'
    headers = {'Authorization': f'Bearer {API_KEY}'}
    entities_list = [
        'Excited', 'Angry', 'Embarrassed', 'Pain', 'Goal', 'Obstacle', 
        'Workaround', 'Background', 'Feature request', 'Money', 
        'Mentioned specific person or company', 'Follow-up task'
    ]
    data = {
        'providers': 'openai',
        'text': text,
        'entities': entities_list
    }
    try:
        logging.info("Detecting indicators...")
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to detect indicators: {e.response.content if e.response else e}")
        raise
    
    entities = response.json().get('results', {}).get('openai', {}).get('items', [])
    logging.info("Indicator detection completed.")
    logging.debug(f"Indicators detected: {entities}")
    return entities

# Step 4: Merge annotations with timestamps
def merge_annotations(transcription_data, emotions, indicators):
    """
    Merge the detected indicators and emotions with the transcription data.
    """
    annotated_transcript = []
    # If segments are available, use them; otherwise, process the full text
    segments = transcription_data.get('segments', [])
    if segments:
        for segment in segments:
            segment_text = segment.get('text', '')
            segment_start = segment.get('start', 0)
            segment_end = segment.get('end', 0)
            segment_emotions = []
            segment_indicators = []

            # Check for emotions in the segment
            for emotion in emotions:
                if emotion.get('start_time', 0) >= segment_start and emotion.get('end_time', 0) <= segment_end:
                    segment_emotions.append(emotion.get('emotion', ''))

            # Check for indicators in the segment
            for entity in indicators:
                if entity.get('start', 0) >= segment_start and entity.get('end', 0) <= segment_end:
                    segment_indicators.append(entity.get('entity', ''))

            annotated_segment = {
                'timestamp': f"{segment_start}-{segment_end}",
                'text': segment_text,
                'emotions': segment_emotions,
                'indicators': segment_indicators
            }
            annotated_transcript.append(annotated_segment)
    else:
        # Process the full text as a single segment
        segment_text = transcription_data.get('text', '')
        annotated_segment = {
            'timestamp': '0-0',
            'text': segment_text,
            'emotions': emotions,
            'indicators': indicators
        }
        annotated_transcript.append(annotated_segment)
    logging.info("Annotations merged successfully.")
    return annotated_transcript

# Step 5: Format the final transcript
def format_transcript(annotated_transcript):
    """
    Generate the final formatted annotated transcript.
    """
    transcript_str = ''
    for segment in annotated_transcript:
        timestamp = segment['timestamp']
        text = segment['text']
        indicators = segment['indicators']
        emotions = segment['emotions']
        indicators_str = ' '.join([f"{get_indicator_symbol(i)} ({i})" for i in indicators])
        emotions_str = ' '.join([f"{get_emotion_symbol(e)} ({e})" for e in emotions])
        transcript_str += f"[{timestamp}]\n\n{text}\n\n{indicators_str} {emotions_str}\n\n"
    logging.info("Transcript formatted successfully.")
    logging.debug(f"Final transcript: {transcript_str}")
    return transcript_str

def get_indicator_symbol(indicator):
    """
    Get the symbol representation of an indicator.

    Args:
        indicator (str): The indicator name.

    Returns:
        str: Symbol representing the indicator.
    """
    symbols = {
        'Excited': ':)',
        'Angry': ':(',
        'Embarrassed': ':|',
        'Pain': '☇',
        'Goal': '⨅',
        'Obstacle': '☐',
        'Workaround': '⤴',
        'Background': '^',
        'Feature request': '☑',
        'Money': '＄',
        'Mentioned specific person or company': '♀',
        'Follow-up task': '☆'
    }
    return symbols.get(indicator, '')

def get_emotion_symbol(emotion):
    """
    Get the symbol representation of an emotion.

    Args:
        emotion (str): The emotion name.

    Returns:
        str: Symbol representing the emotion.
    """
    symbols = {
        'joy': ':)',
        'sadness': ':(',
        'anger': '>:{',
        'fear': 'D:',
        'disgust': ':{',
        'surprise': ':O',
        'neutral': ':|'
    }
    return symbols.get(emotion, '')

# Main function
def main():
    """
    Main function to process the audio file and generate the annotated transcript.
    """
    audio_file_path = '/Users/aadityarajesh/Library/Mobile Documents/com~apple~CloudDocs/Code/precise-customer-conversation-transcriber/Customer_Discovery_Interview.mp3'

    # Step 1: Transcribe audio
    try:
        transcription_data = transcribe_audio(audio_file_path)
    except Exception as e:
        logging.error(f"Transcription failed: {e}")
        return

    # Extract the text from transcription
    try:
        # Access the 'transcript' key, which may contain the full text
        transcribed_text = transcription_data.get('text', '')
        if not transcribed_text:
            # If 'text' key is not present, check other possible keys
            transcribed_text = transcription_data.get('transcript', '')
        if not transcribed_text:
            logging.error("Transcribed text not found in the response.")
            logging.debug(f"Transcription data: {json.dumps(transcription_data, indent=2)}")
            return
    except Exception as e:
        logging.error(f"Error extracting transcribed text: {e}")
        logging.debug(f"Transcription data: {json.dumps(transcription_data, indent=2)}")
        return

    # Step 2: Detect emotions
    try:
        emotions = detect_emotions(transcribed_text)
    except Exception as e:
        logging.error(f"Emotion detection failed: {e}")
        emotions = []

    # Step 3: Identify indicators
    try:
        indicators = detect_indicators(transcribed_text)
    except Exception as e:
        logging.error(f"Indicator detection failed: {e}")
        indicators = []

    # Step 4: Merge annotations
    try:
        annotated_transcript = merge_annotations(transcription_data, emotions, indicators)
    except Exception as e:
        logging.error(f"Merging annotations failed: {e}")
        annotated_transcript = []

    # Step 5: Format transcript
    try:
        final_transcript = format_transcript(annotated_transcript)
    except Exception as e:
        logging.error(f"Formatting transcript failed: {e}")
        final_transcript = ""

    # Save the final annotated transcript
    try:
        with open('annotated_transcript.txt', 'w') as f:
            f.write(final_transcript)
        logging.info("Annotated transcript has been saved successfully.")
    except Exception as e:
        logging.error(f"Failed to save annotated transcript: {e}")

if __name__ == '__main__':
    main()