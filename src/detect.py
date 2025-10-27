import numpy as np
import sounddevice as sd
import wave
import threading
import queue
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
import scipy.io.wavfile as wavfile

# Configuration
CHUNK = 1024  # Audio chunk size
CHANNELS = 1  # Mono audio
RATE = 44100  # Sample rate (44.1 kHz)
THRESHOLD = 0.3  # Amplitude threshold (0.0 to 1.0, adjust based on your needs)
BUFFER_DURATION = 1.0  # Seconds to keep before trigger
CAPTURE_DURATION = 2.0  # Total capture duration (1s before + 1s after)

class GunShotDetector:
    def __init__(self):
        self.running = False
        
        # Calculate buffer sizes
        self.buffer_size = int(RATE * BUFFER_DURATION)
        self.capture_size = int(RATE * CAPTURE_DURATION)
        
        # Circular buffer to store recent audio
        self.audio_buffer = deque(maxlen=self.buffer_size)
        
        # Queue for visualization
        self.viz_queue = queue.Queue(maxsize=100)
        
        # Detection state
        self.detection_lock = threading.Lock()
        self.detecting = False
        self.capture_buffer = []
        self.samples_to_capture = 0
        
    def audio_callback(self, indata, frames, time_info, status):
        """Callback function for audio stream"""
        if status:
            print(f"Status: {status}")
        
        # Convert to 1D array
        audio_data = indata[:, 0].copy()
        
        # Add to visualization queue
        try:
            self.viz_queue.put_nowait(audio_data.copy())
        except queue.Full:
            pass
        
        # Calculate amplitude (normalized 0.0 to 1.0)
        amplitude = np.max(np.abs(audio_data))
        
        with self.detection_lock:
            if self.detecting:
                # Currently capturing audio after detection
                self.capture_buffer.extend(audio_data)
                self.samples_to_capture -= len(audio_data)
                
                if self.samples_to_capture <= 0:
                    # Finished capturing
                    self.save_audio()
                    self.detecting = False
                    self.capture_buffer = []
            else:
                # Add to circular buffer
                self.audio_buffer.extend(audio_data)
                
                # Check for threshold crossing
                if amplitude > THRESHOLD:
                    print(f"🔊 DETECTION! Amplitude: {amplitude:.3f}")
                    self.trigger_capture()
        
    def start(self):
        """Start audio stream and detection"""
        self.running = True
        
        # Print available audio devices
        print("Available audio devices:")
        print(sd.query_devices())
        print()
        
        # Start audio stream
        self.stream = sd.InputStream(
            channels=CHANNELS,
            samplerate=RATE,
            blocksize=CHUNK,
            callback=self.audio_callback
        )
        self.stream.start()
        
        print("🎤 Audio detection started...")
        print(f"Threshold: {THRESHOLD} (0.0 to 1.0 scale)")
        print(f"Sample rate: {RATE} Hz")
        print(f"Monitoring for loud sounds...\n")
        
    def stop(self):
        """Stop audio stream"""
        self.running = False
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
        
    def trigger_capture(self):
        """Triggered when threshold is exceeded"""
        with self.detection_lock:
            self.detecting = True
            
            # Start capture buffer with historical data (1 second before)
            self.capture_buffer = list(self.audio_buffer)
            
            # Calculate how many more samples needed (1 second after)
            self.samples_to_capture = int(RATE * 1.0)
            
    def save_audio(self):
        """Save captured audio to WAV file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"detection_{timestamp}.wav"
        
        # Convert to numpy array
        audio_array = np.array(self.capture_buffer, dtype=np.float32)
        
        # Save as WAV (scipy handles float32 properly)
        wavfile.write(filename, RATE, audio_array)
        
        print(f"💾 Saved: {filename} ({len(audio_array)} samples, {len(audio_array)/RATE:.2f}s)")
        print(f"Amplitude range: {np.min(audio_array):.3f} to {np.max(audio_array):.3f}\n")

class AudioVisualizer:
    def __init__(self, detector):
        self.detector = detector
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Waveform display
        self.display_duration = 5.0  # Show 5 seconds
        self.display_samples = int(RATE * self.display_duration)
        self.waveform_data = deque(maxlen=self.display_samples)
        
        # Initialize with zeros
        self.waveform_data.extend([0] * self.display_samples)
        
        # Setup waveform plot
        self.time_axis = np.linspace(0, self.display_duration, self.display_samples)
        self.line1, = self.ax1.plot(self.time_axis, list(self.waveform_data))
        self.ax1.set_ylim(-1.0, 1.0)
        self.ax1.set_xlim(0, self.display_duration)
        self.ax1.set_xlabel('Time (seconds)')
        self.ax1.set_ylabel('Amplitude')
        self.ax1.set_title('Live Audio Waveform')
        self.ax1.grid(True, alpha=0.3)
        
        # Threshold line
        self.ax1.axhline(y=THRESHOLD, color='r', linestyle='--', label=f'Threshold: {THRESHOLD}')
        self.ax1.axhline(y=-THRESHOLD, color='r', linestyle='--')
        self.ax1.legend()
        
        # Setup amplitude meter
        self.amp_data = deque(maxlen=100)
        self.amp_data.extend([0] * 100)
        self.line2, = self.ax2.plot(self.amp_data)
        self.ax2.set_ylim(0, 1.0)
        self.ax2.set_xlim(0, 100)
        self.ax2.set_xlabel('Time (chunks)')
        self.ax2.set_ylabel('Max Amplitude')
        self.ax2.set_title('Amplitude History')
        self.ax2.grid(True, alpha=0.3)
        self.ax2.axhline(y=THRESHOLD, color='r', linestyle='--')
        
        # Enable interactive features
        self.ax1.set_navigate(True)
        self.ax2.set_navigate(True)
        
        plt.tight_layout()
        
    def update(self, frame):
        """Update visualization"""
        # Get new audio data from queue
        while not self.detector.viz_queue.empty():
            try:
                chunk = self.detector.viz_queue.get_nowait()
                self.waveform_data.extend(chunk)
                self.amp_data.append(np.max(np.abs(chunk)))
            except queue.Empty:
                break
        
        # Update waveform
        self.line1.set_ydata(list(self.waveform_data))
        
        # Update amplitude meter
        self.line2.set_ydata(list(self.amp_data))
        
        return self.line1, self.line2
    
    def show(self):
        """Start animation and show plot"""
        ani = FuncAnimation(self.fig, self.update, interval=50, blit=True, cache_frame_data=False)
        plt.show()

def main():
    detector = GunShotDetector()
    visualizer = AudioVisualizer(detector)
    
    # Start detection
    detector.start()
    
    # Show visualization (blocks until window closed)
    try:
        visualizer.show()
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        detector.stop()

if __name__ == "__main__":
    main()